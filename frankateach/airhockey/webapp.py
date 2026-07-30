"""Browser-based air hockey teleop server.

Replaces the pygame app: the NUC has no monitor, so the UI is a web page you
reach through an SSH tunnel. One aiohttp server on one port carries everything:

    GET  /        the control page
    GET  /video   MJPEG stream from camera_server.py (if a camera is publishing)
    WS   /ws      key state up, arm status down

Why full key-state snapshots instead of keydown/keyup events
-----------------------------------------------------------
The browser is on the far side of an SSH tunnel: measured 10.3 ms median RTT,
29 ms p99, 43 ms worst. With edge events, a single dropped or delayed `keyup`
leaves the arm driving into a wall until the next event arrives. So the client
sends the COMPLETE set of held keys ~50x/sec and the server just applies the
newest one. Any lost packet self-heals on the next tick, and if the stream
stops entirely, ArmOperator's own 100 ms watchdog freezes the arm.

The control loop itself never leaves the NUC -- only intent crosses the network.
"""

import asyncio
import json
import time
from pathlib import Path

import numpy as np
from aiohttp import WSCloseCode, WSMsgType, web

from frankateach.airhockey.operator import ArmOperator
from frankateach.constants import CAM_PORT, HOST

STATIC = Path(__file__).resolve().parent / "static"

from frankateach.airhockey.keyboard import (
    KEY_SETS as WEB_KEY_SETS,
    resolve_jog_z_keys,
    resolve_keys,
)


class CameraRelay:
    """Pulls RGB frames off the ZMQ camera bus and hands out JPEG bytes.

    Runs in a thread because ZMQCameraSubscriber is blocking. Degrades to
    'no frame' rather than failing if camera_server.py is not running.
    """

    def __init__(self, cam_id):
        self.cam_id = cam_id
        self._jpeg = None
        self._stamp = 0.0
        self._stop_event = None
        self._thread = None
        self.error = ""
        self.frames = 0

    def start(self):
        import threading

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="cam-relay")
        self._thread.start()

    def _run(self):
        import cv2

        from frankateach.network import ZMQCameraSubscriber

        try:
            sub = ZMQCameraSubscriber(HOST, CAM_PORT + self.cam_id, "RGB")
        except Exception as exc:
            self.error = f"camera subscribe failed: {type(exc).__name__}: {exc}"
            return
        try:
            while not self._stop_event.is_set():
                try:
                    image, _ = sub.recv_rgb_image()
                except Exception as exc:
                    self.error = f"camera recv failed: {type(exc).__name__}: {exc}"
                    break
                ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                if ok:
                    self._jpeg = buf.tobytes()
                    self._stamp = time.time()
                    self.frames += 1
                    self.error = ""
        finally:
            try:
                sub.stop()
            except Exception:
                pass

    def latest(self):
        return self._jpeg

    def stop(self):
        if self._stop_event is not None:
            self._stop_event.set()


def _status_payload(operators, boxes):
    out = {}
    for arm, op in operators.items():
        st = op.get_status()
        box = boxes[arm]
        out[arm] = {
            "connected": bool(st.connected),
            "error": st.error,
            "stale": bool(st.stale),
            "homing": bool(st.homing),
            "rate": round(float(st.rate), 1),
            "speed": round(float(st.speed), 3),
            "pos": [round(float(v), 4) for v in st.pos],
            # Normalised box coords in [-1, 1] so the page needs no geometry.
            "u": float(np.clip(st.box_pos[0] / box.half_extents[0], -1, 1))
            if box.half_extents[0]
            else 0.0,
            "v": float(np.clip(st.box_pos[1] / box.half_extents[1], -1, 1))
            if box.half_extents[1]
            else 0.0,
            "extents": [float(v) for v in box.half_extents],
            "yaw_deg": round(float(np.degrees(box.yaw)), 2),
        }
    return out


def build_app(operators, boxes, cfg, camera):
    app = web.Application()
    app["operators"] = operators
    app["boxes"] = boxes
    app["cfg"] = cfg
    app["camera"] = camera
    app["websockets"] = set()

    keymap = {arm: cfg["arms"][arm]["keys"] for arm in operators}

    async def index(request):
        html = (STATIC / "index.html").read_text()
        return web.Response(text=html, content_type="text/html")

    async def config_json(request):
        return web.json_response(
            {
                "arms": list(operators.keys()),
                "keys": {a: WEB_KEY_SETS[keymap[a]] for a in operators},
                "keyset_names": keymap,
                "control_hz": cfg["control_hz"],
                "speed": cfg["speed"],
                "watchdog": cfg["watchdog"],
                "has_camera": camera is not None,
            }
        )

    async def video(request):
        if camera is None:
            raise web.HTTPNotFound(text="no camera configured")
        # Wait briefly for a first frame. Without this, a configured-but-absent
        # camera leaves the <img> hanging blank forever instead of firing an
        # error, so the page never shows its "no camera" fallback.
        deadline = time.time() + 3.0
        while camera.latest() is None and time.time() < deadline:
            await asyncio.sleep(0.1)
        if camera.latest() is None:
            raise web.HTTPServiceUnavailable(
                text=camera.error or f"no frames on camera {camera.cam_id}; "
                "is camera_server.py running?"
            )
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
            },
        )
        await resp.prepare(request)
        last = None
        try:
            while True:
                jpeg = camera.latest()
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    await resp.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                await asyncio.sleep(1 / 30)
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        return resp

    async def websocket(request):
        ws = web.WebSocketResponse(heartbeat=5.0)
        await ws.prepare(request)
        request.app["websockets"].add(ws)
        print(f"[web] client connected ({len(request.app['websockets'])} total)")

        async def push_status():
            try:
                while not ws.closed:
                    await ws.send_json(
                        {"t": "status", "arms": _status_payload(operators, boxes)}
                    )
                    await asyncio.sleep(1 / 20)
            except (asyncio.CancelledError, ConnectionResetError):
                pass

        pusher = asyncio.create_task(push_status())
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if data.get("t") != "keys":
                    continue

                held = set(data.get("held") or [])
                frozen = bool(data.get("frozen", True))
                home = bool(data.get("home", False))
                for arm, op in operators.items():
                    k = WEB_KEY_SETS[keymap[arm]]
                    vx = float(k["up"] in held) - float(k["down"] in held)
                    vy = float(k["left"] in held) - float(k["right"] in held)
                    op.set_intent(vx, vy, frozen=frozen, home=home)
        finally:
            pusher.cancel()
            request.app["websockets"].discard(ws)
            # Losing the UI must stop the arms immediately; do not wait for the
            # watchdog to notice.
            for op in operators.values():
                op.set_intent(0.0, 0.0, frozen=True)
            print(f"[web] client disconnected; arms frozen")
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/config", config_json)
    app.router.add_get("/video", video)
    app.router.add_get("/ws", websocket)

    async def on_shutdown(app):
        for ws in set(app["websockets"]):
            await ws.close(code=WSCloseCode.GOING_AWAY, message=b"server shutdown")
        for op in operators.values():
            op.set_intent(0.0, 0.0, frozen=True)

    app.on_shutdown.append(on_shutdown)
    return app


def serve(arms, cfg, boxes, publish=True, port=8080, bind="127.0.0.1"):
    """Start the operators and the web server. Blocks until interrupted."""
    camera = None
    if cfg.get("view_cam_id") is not None:
        camera = CameraRelay(int(cfg["view_cam_id"]))
        camera.start()

    operators = {a: ArmOperator(a, boxes[a], cfg, publish=publish) for a in arms}
    for op in operators.values():
        op.start()

    print("Bringing arms to their box centres...")
    for arm, op in operators.items():
        if not op.wait_ready(timeout=90):
            print(f"[{arm}] did not come up in time")
    dead = [a for a, op in operators.items() if not op.get_status().connected]
    if dead:
        print(f"WARNING: {', '.join(dead)} failed to start -- see the error above.")

    app = build_app(operators, boxes, cfg, camera)
    print()
    print(f"  Air hockey UI on http://{bind}:{port}")
    print(f"  From your laptop:  ssh -L {port}:localhost:{port} franka")
    print(f"  then open          http://localhost:{port}")
    print()
    try:
        web.run_app(app, host=bind, port=port, print=None, access_log=None)
    except KeyboardInterrupt:
        pass
    finally:
        print("Parking arms...")
        for op in operators.values():
            op.stop()
        for op in operators.values():
            op.join(timeout=5)
        if camera is not None:
            camera.stop()


# ---------------------------------------------------------------------------
# Calibration mode
# ---------------------------------------------------------------------------

CORNER_NAMES = [
    "corner 1 (near-right)",
    "corner 2 (far-right)",
    "corner 3 (far-left)",
    "corner 4 (near-left)",
]


def build_calibrate_app(arm, jog, cfg):
    from frankateach.airhockey import config as ahconfig
    from frankateach.airhockey.box import box_from_corners

    # Jog with the arm's own key set, not a hardcoded wasd. Otherwise calibrating
    # the right arm uses the left arm's keys, which is exactly the sort of thing
    # you discover with a mallet in your hand.
    keyset_name = str(((cfg.get("arms") or {}).get(arm) or {}).get("keys", "wasd"))
    keys = resolve_keys(keyset_name)
    jog_z = resolve_jog_z_keys(keyset_name)

    app = web.Application()
    app["websockets"] = set()
    state = {"corners": [], "message": "", "done": False, "summary": None, "busy": False}

    async def index(request):
        return web.Response(
            text=(STATIC / "index.html").read_text(), content_type="text/html"
        )

    async def config_json(request):
        return web.json_response(
            {
                "mode": "calibrate",
                "arm": arm,
                "arms": [arm],
                "keys": {arm: keys},
                "keyset_names": {arm: keyset_name},
                "jog_z_keys": jog_z,
                "corner_names": CORNER_NAMES,
                "control_hz": cfg["control_hz"],
                "has_camera": False,
            }
        )

    def snapshot():
        st = jog.get_status()
        return {
            "t": "status",
            "pos": [round(float(v), 4) for v in st.pos],
            "rate": round(float(st.rate), 1),
            "connected": bool(st.connected),
            "stale": bool(st.stale),
            "error": st.error,
            "corners": [[round(float(v), 4) for v in c] for c in state["corners"]],
            "message": state["message"],
            "done": state["done"],
            "summary": state["summary"],
            "busy": state["busy"],
        }

    def do_finish():
        corners = np.array(state["corners"])
        st = jog.get_status()
        box, warnings = box_from_corners(
            corners,
            st.quat,
            margin=float(cfg["margin"]),
            plane_mode=str(cfg.get("plane_mode", "constant")),
            level_wrist=bool(cfg.get("level_wrist", False)),
        )
        path = ahconfig.save_box(arm, box)
        state["summary"] = {
            "center": [round(float(v), 4) for v in box.center],
            "yaw_deg": round(float(np.degrees(box.yaw)), 2),
            "half_extents_mm": [round(float(v) * 1000, 1) for v in box.half_extents],
            "play_area_m": [
                round(float(box.half_extents[0]) * 2, 3),
                round(float(box.half_extents[1]) * 2, 3),
            ],
            "plane_z": round(float(box.plane_z), 4),
            "plane_mode": str(box.plane_mode),
            "tilt_mm_per_m": round(float(np.linalg.norm(box.plane_coeffs[1:])) * 1000, 1),
            "plane_residual_mm": round(float(box.plane_residual) * 1000, 1),
            "z_spread_mm": round(float(box.z_spread) * 1000, 1),
            "quat": [round(float(v), 4) for v in box.quat],
            "warnings": warnings,
            "path": str(path),
        }
        state["done"] = True
        state["message"] = f"Saved to {path}"
        print("\n" + "=" * 68)
        print(f"{arm.upper()} ARM PLAY BOX")
        for k, v in state["summary"].items():
            if k != "warnings":
                print(f"  {k:16s}: {v}")
        for w in warnings:
            print(f"\n  !! {w}")
        print("=" * 68)
        return box

    async def websocket(request):
        ws = web.WebSocketResponse(heartbeat=5.0)
        await ws.prepare(request)
        request.app["websockets"].add(ws)

        async def push():
            try:
                while not ws.closed:
                    await ws.send_json(snapshot())
                    await asyncio.sleep(1 / 20)
            except (asyncio.CancelledError, ConnectionResetError):
                pass

        pusher = asyncio.create_task(push())
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                if data.get("t") == "keys":
                    held = set(data.get("held") or [])
                    frozen = bool(data.get("frozen", True)) or state["busy"]
                    vx = float(keys["up"] in held) - float(keys["down"] in held)
                    vy = float(keys["left"] in held) - float(keys["right"] in held)
                    vz = float(jog_z["up"] in held) - float(jog_z["down"] in held)
                    jog.set_intent(vx, vy, vz, frozen=frozen)

                elif data.get("t") == "cmd":
                    cmd = data.get("cmd")
                    if cmd == "record":
                        if len(state["corners"]) >= 4:
                            state["message"] = "Already have 4 corners."
                        elif not jog.get_status().connected:
                            state["message"] = "Arm not connected."
                        else:
                            pos = np.asarray(jog.get_status().pos, dtype=np.float64)
                            state["corners"].append(pos.copy())
                            state["message"] = (
                                f"Recorded {CORNER_NAMES[len(state['corners']) - 1]}"
                            )
                            print(f"  recorded corner {len(state['corners'])}: {pos}")
                    elif cmd == "undo":
                        if state["corners"]:
                            state["corners"].pop()
                            state["message"] = "Dropped last corner."
                            state["done"] = False
                            state["summary"] = None
                    elif cmd == "finish":
                        if len(state["corners"]) != 4:
                            state["message"] = (
                                f"Need 4 corners, have {len(state['corners'])}."
                            )
                        else:
                            try:
                                do_finish()
                            except Exception as exc:
                                state["message"] = f"{type(exc).__name__}: {exc}"
                    elif cmd == "trace":
                        if not state["done"]:
                            state["message"] = "Finish calibration first."
                        elif state["busy"]:
                            state["message"] = "Already tracing."
                        else:
                            state["busy"] = True
                            state["message"] = "Tracing perimeter..."
                            jog.set_intent(0, 0, 0, frozen=True)

                            def run_trace():
                                from frankateach.airhockey.calibrate import (
                                    trace_perimeter,
                                )

                                box = ahconfig.load_box(arm)
                                trace_perimeter(arm, box, cfg, link=jog.link)

                            try:
                                await asyncio.get_running_loop().run_in_executor(
                                    None, run_trace
                                )
                                state["message"] = "Trace complete."
                            except Exception as exc:
                                state["message"] = f"Trace failed: {exc}"
                            finally:
                                state["busy"] = False
        finally:
            pusher.cancel()
            request.app["websockets"].discard(ws)
            jog.set_intent(0.0, 0.0, 0.0, frozen=True)
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/config", config_json)
    app.router.add_get("/ws", websocket)
    return app


def serve_calibrate(arm, cfg, port=8080, bind="127.0.0.1", reset=True):
    from frankateach.airhockey.operator import JogOperator

    jog = JogOperator(arm, cfg, reset=reset)
    jog.start()
    if not jog.wait_ready(timeout=90):
        print(f"[{arm}] jog did not come up in time")
    if not jog.get_status().connected:
        print(f"ERROR: {jog.get_status().error}")
        jog.stop()
        return 1

    app = build_calibrate_app(arm, jog, cfg)
    print()
    print(f"  Calibrating {arm} arm — open http://{bind}:{port}")
    print(f"  From your laptop:  ssh -L {port}:localhost:{port} franka")
    print()
    try:
        web.run_app(app, host=bind, port=port, print=None, access_log=None)
    except KeyboardInterrupt:
        pass
    finally:
        jog.stop()
        jog.join(timeout=5)
    return 0
