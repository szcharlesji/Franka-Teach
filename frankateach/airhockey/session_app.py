"""One aiohttp app serving both play and calibrate, with live mode switching.

The existing `webapp.serve` / `serve_calibrate` each build a single-purpose app
and exit when you are done. This one is driven by `ArmSession` objects, so an arm
can be moved between play and calibrate without restarting anything below it --
franka-interface, franka_server and the browser tab all stay up.

Mode changes are pushed to the page, which reloads itself: the layout differs
enough between the two that re-rendering in place would mean maintaining a third
code path in the HTML for no benefit.
"""

import asyncio
import json
import time

import numpy as np
from aiohttp import WSMsgType, web

from frankateach.airhockey import config as ahconfig
from frankateach.airhockey.box import box_from_corners
from frankateach.airhockey.calibrate import trace_perimeter
from frankateach.airhockey.keyboard import (
    KEY_FREEZE,
    KEY_HOME,
    resolve_jog_z_keys,
    resolve_keys,
)
from frankateach.airhockey.session import CALIBRATE, PLAY
from frankateach.airhockey.webapp import CORNER_NAMES, STATIC, CameraRelay


def build_app(sessions, cfg, supervisor=None, camera=None):
    app = web.Application()
    app["websockets"] = set()

    arms = list(sessions)
    keymap = {a: str(((cfg.get("arms") or {}).get(a) or {}).get("keys", "wasd")) for a in arms}
    # Calibration is one arm at a time: two arms jogging free in 3 DoF with no
    # play box between them is the one situation nothing in this stack bounds.
    cal = {"arm": None, "corners": [], "message": "", "summary": None, "busy": False}

    # -- static + config ---------------------------------------------------
    async def index(request):
        return web.Response(
            text=(STATIC / "index.html").read_text(), content_type="text/html"
        )

    async def config_json(request):
        arm = cal["arm"]
        if arm:
            return web.json_response(
                {
                    "mode": "calibrate",
                    "arm": arm,
                    "arms": [arm],
                    "keys": {arm: resolve_keys(keymap[arm])},
                    "keyset_names": {arm: keymap[arm]},
                    "jog_z_keys": resolve_jog_z_keys(keymap[arm]),
                    "corner_names": CORNER_NAMES,
                    "control_hz": cfg["control_hz"],
                    "has_camera": False,
                    "unified": True,
                }
            )
        return web.json_response(
            {
                "mode": "play",
                "arms": arms,
                "keys": {a: resolve_keys(keymap[a]) for a in arms},
                "keyset_names": keymap,
                "jog_z_keys": None,
                "control_hz": cfg["control_hz"],
                "speed": cfg["speed"],
                "watchdog": cfg["watchdog"],
                "has_camera": camera is not None,
                "unified": True,
            }
        )

    async def video(request):
        if camera is None:
            raise web.HTTPNotFound(text="no camera configured")
        # Wait briefly for a first frame, so a configured-but-absent camera errors
        # instead of leaving the <img> blank forever with no fallback.
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

    # -- status ------------------------------------------------------------
    def supervisor_block():
        if supervisor is None:
            return {}
        return supervisor.status()

    def snapshot():
        arm = cal["arm"]
        out = {
            "t": "status",
            "mode": "calibrate" if arm else "play",
            "supervisor": supervisor_block(),
        }
        if arm:
            st = sessions[arm].get_status()
            out.update(
                {
                    "pos": [round(float(v), 4) for v in (st.pos if st else np.zeros(3))],
                    "rate": round(float(st.rate), 1) if st else 0.0,
                    "connected": bool(st.connected) if st else False,
                    "stale": bool(st.stale) if st else True,
                    "error": (st.error if st else "") or sessions[arm].error,
                    "corners": [[round(float(v), 4) for v in c] for c in cal["corners"]],
                    "message": cal["message"],
                    "done": cal["summary"] is not None,
                    "summary": cal["summary"],
                    "busy": cal["busy"],
                }
            )
            return out

        out["arms"] = {}
        for a, sess in sessions.items():
            st = sess.get_status()
            box = sess.box
            out["arms"][a] = {
                "pos": [round(float(v), 4) for v in (st.pos if st else np.zeros(3))],
                "box_pos": [
                    round(float(v), 4) for v in (st.box_pos if st else np.zeros(2))
                ],
                "speed": round(float(st.speed), 3) if st else 0.0,
                "rate": round(float(st.rate), 1) if st else 0.0,
                "connected": bool(st.connected) if st else False,
                "stale": bool(st.stale) if st else True,
                "homing": bool(st.homing) if st else False,
                "error": (st.error if st else "") or sess.error,
                "provisional": bool(sess.provisional),
                "plane_mode": None if box is None else str(box.plane_mode),
            }
        return out

    async def broadcast():
        payload = json.dumps(snapshot())
        for ws in list(app["websockets"]):
            try:
                await ws.send_str(payload)
            except ConnectionResetError:
                pass

    # -- calibration actions ----------------------------------------------
    def freeze_all():
        for sess in sessions.values():
            sess.set_intent(0.0, 0.0, frozen=True)

    async def enter_calibrate(arm):
        if arm not in sessions:
            cal["message"] = f"Unknown arm {arm}"
            return False
        freeze_all()
        # Every other arm keeps holding its pose in play mode; only this one is
        # handed to a JogOperator.
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(None, sessions[arm].set_mode, CALIBRATE)
        if ok:
            cal.update({"arm": arm, "corners": [], "message": "", "summary": None})
        else:
            cal["message"] = sessions[arm].error or "could not enter calibrate"
        return ok

    async def leave_calibrate():
        arm = cal["arm"]
        if arm is None:
            return True
        freeze_all()
        loop = asyncio.get_running_loop()
        # reload_box picks up whatever calibration was just saved.
        ok = await loop.run_in_executor(None, sessions[arm].reload_box)
        cal.update({"arm": None, "corners": [], "message": "", "summary": None})
        return ok

    def do_finish():
        arm = cal["arm"]
        st = sessions[arm].get_status()
        box, warnings = box_from_corners(
            np.array(cal["corners"]),
            st.quat,
            margin=float(cfg["margin"]),
            plane_mode=str(cfg.get("plane_mode", "constant")),
            level_wrist=bool(cfg.get("level_wrist", False)),
        )
        path = ahconfig.save_box(arm, box)
        cal["summary"] = {
            "center": [round(float(v), 4) for v in box.center],
            "yaw_deg": round(float(np.degrees(box.yaw)), 2),
            "half_extents_mm": [round(float(v) * 1000, 1) for v in box.half_extents],
            "play_area_m": [
                round(float(box.half_extents[0]) * 2, 3),
                round(float(box.half_extents[1]) * 2, 3),
            ],
            "plane_z": round(float(box.plane_z), 4),
            # What the arm will actually hold: the taught surface plus clearance.
            "play_z": round(
                float(box.plane_z) + float(cfg.get("plane_offset", 0.0)), 4
            ),
            "plane_offset_mm": round(float(cfg.get("plane_offset", 0.0)) * 1000, 1),
            "plane_mode": str(box.plane_mode),
            "tilt_mm_per_m": round(
                float(np.linalg.norm(box.plane_coeffs[1:])) * 1000, 1
            ),
            "plane_residual_mm": round(float(box.plane_residual) * 1000, 1),
            "corner_spread_mm": round(float(box.z_spread) * 1000, 1),
            "warnings": warnings,
            "saved_to": str(path),
        }
        cal["message"] = f"Saved to {path}"
        return box

    # -- websocket ---------------------------------------------------------
    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        app["websockets"].add(ws)
        print(f"[web] client connected ({len(app['websockets'])} total)")

        async def push():
            try:
                while not ws.closed:
                    await ws.send_str(json.dumps(snapshot()))
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
                    frozen = bool(data.get("frozen", True))
                    arm = cal["arm"]
                    if arm:
                        k = resolve_keys(keymap[arm])
                        z = resolve_jog_z_keys(keymap[arm])
                        vx = float(k["up"] in held) - float(k["down"] in held)
                        vy = float(k["left"] in held) - float(k["right"] in held)
                        vz = float(z["up"] in held) - float(z["down"] in held)
                        sessions[arm].set_intent(
                            vx, vy, vz, frozen=frozen or cal["busy"]
                        )
                    else:
                        home = KEY_HOME in held
                        for a, sess in sessions.items():
                            k = resolve_keys(keymap[a])
                            vx = float(k["up"] in held) - float(k["down"] in held)
                            vy = float(k["left"] in held) - float(k["right"] in held)
                            sess.set_intent(
                                vx,
                                vy,
                                frozen=frozen or KEY_FREEZE in held,
                                home=home,
                            )

                elif data.get("t") == "cmd":
                    cmd = data.get("cmd")
                    if cmd == "calibrate":
                        await enter_calibrate(data.get("arm"))
                    elif cmd == "play":
                        await leave_calibrate()
                    elif cal["arm"] is None:
                        cal["message"] = "Not in calibrate mode."
                    elif cmd == "record":
                        st = sessions[cal["arm"]].get_status()
                        if len(cal["corners"]) >= 4:
                            cal["message"] = "Already have 4 corners."
                        elif st is None or not st.connected:
                            cal["message"] = "Arm not connected."
                        else:
                            cal["corners"].append(
                                np.asarray(st.pos, dtype=np.float64).copy()
                            )
                            cal["message"] = (
                                f"Recorded {CORNER_NAMES[len(cal['corners']) - 1]}"
                            )
                            print(
                                f"  recorded corner {len(cal['corners'])}: "
                                f"{cal['corners'][-1]}"
                            )
                    elif cmd == "undo":
                        if cal["corners"]:
                            cal["corners"].pop()
                            cal["summary"] = None
                            cal["message"] = "Dropped last corner."
                    elif cmd == "finish":
                        if len(cal["corners"]) != 4:
                            cal["message"] = "Need exactly 4 corners."
                        else:
                            try:
                                do_finish()
                            except Exception as exc:
                                cal["message"] = f"{type(exc).__name__}: {exc}"
                    elif cmd == "trace":
                        if cal["summary"] is None:
                            cal["message"] = "Finish calibration first."
                        else:
                            arm = cal["arm"]
                            cal["busy"] = True
                            cal["message"] = "Tracing perimeter..."
                            await broadcast()
                            try:
                                box = ahconfig.load_box(arm)
                                loop = asyncio.get_running_loop()
                                await loop.run_in_executor(
                                    None,
                                    trace_perimeter,
                                    arm,
                                    box,
                                    cfg,
                                    sessions[arm].operator.link,
                                )
                                cal["message"] = "Trace complete."
                            except Exception as exc:
                                cal["message"] = f"Trace failed: {exc}"
                            finally:
                                cal["busy"] = False
        finally:
            pusher.cancel()
            app["websockets"].discard(ws)
            freeze_all()
            print("[web] client disconnected; arms frozen")
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/config", config_json)
    app.router.add_get("/video", video)
    app.router.add_get("/ws", ws_handler)
    return app


def serve(sessions, cfg, supervisor=None, camera=None, port=8080, bind="127.0.0.1"):
    app = build_app(sessions, cfg, supervisor=supervisor, camera=camera)
    print()
    print(f"  Air hockey — open http://{bind}:{port}")
    print(f"  From your laptop:  ssh -L {port}:localhost:{port} franka")
    print()
    try:
        web.run_app(app, host=bind, port=port, print=None, access_log=None)
    except KeyboardInterrupt:
        pass
    return 0


__all__ = ["build_app", "serve", "CameraRelay", "PLAY", "CALIBRATE", "time"]
