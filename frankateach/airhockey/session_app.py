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
from frankateach.airhockey.gamepad import LocalGamepad
from frankateach.airhockey.keyboard import (
    KEY_FREEZE,
    KEY_HOME,
    resolve_jog_z_keys,
    resolve_keys,
)
from frankateach.airhockey.session import CALIBRATE, PLAY
from frankateach.airhockey.webapp import CORNER_NAMES, STATIC, CameraRelay


def build_app(sessions, cfg, supervisor=None, camera=None, gamepad=None):
    app = web.Application()
    app["websockets"] = set()

    arms = list(sessions)
    keymap = {a: str(((cfg.get("arms") or {}).get(a) or {}).get("keys", "wasd")) for a in arms}
    # Calibration is one arm at a time: two arms jogging free in 3 DoF with no
    # play box between them is the one situation nothing in this stack bounds.
    cal = {"arm": None, "corners": [], "message": "", "summary": None, "busy": False}
    actions = {
        arm: {"restarting": False, "message": ""} for arm in sessions
    }
    controls = {"same_side": False, "revision": 0}
    local_gamepad_active = set()

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
                    "all_arms": arms,
                    "keys": {arm: resolve_keys(keymap[arm])},
                    "keyset_names": {arm: keymap[arm]},
                    "jog_z_keys": resolve_jog_z_keys(keymap[arm]),
                    "corner_names": CORNER_NAMES,
                    "control_hz": cfg["control_hz"],
                    "speed": cfg["speed"],
                    "same_side_controls": controls["same_side"],
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
                "same_side_controls": controls["same_side"],
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
            "speed_limit": float(cfg["speed"]),
            "gamepad": gamepad.status() if gamepad is not None else {},
            "same_side_controls": controls["same_side"],
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
                    "restart": actions[arm],
                }
            )
            return out

        out["arms"] = {}
        for a, sess in sessions.items():
            st = sess.get_status()
            box = sess.box
            extents = np.zeros(2) if box is None else box.half_extents
            box_pos = np.zeros(2) if st is None else st.box_pos
            out["arms"][a] = {
                "pos": [round(float(v), 4) for v in (st.pos if st else np.zeros(3))],
                "box_pos": [round(float(v), 4) for v in box_pos],
                "speed": round(float(st.speed), 3) if st else 0.0,
                "speed_limit": round(float(st.speed_limit), 3) if st else 0.0,
                "rate": round(float(st.rate), 1) if st else 0.0,
                "connected": bool(st.connected) if st else False,
                "stale": bool(st.stale) if st else True,
                "homing": bool(st.homing) if st else False,
                "error": (st.error if st else "") or sess.error,
                "provisional": bool(sess.provisional),
                "plane_mode": None if box is None else str(box.plane_mode),
                "u": float(np.clip(box_pos[0] / extents[0], -1, 1))
                if extents[0]
                else 0.0,
                "v": float(np.clip(box_pos[1] / extents[1], -1, 1))
                if extents[1]
                else 0.0,
                "extents": [float(v) for v in extents],
                "yaw_deg": round(float(np.degrees(box.yaw)), 2) if box else 0.0,
                "restart": actions[a],
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

    def player_controls(arm, vx, vy):
        """Rotate controls when both players stand along the same table side."""
        if not controls["same_side"] or cal["arm"] is not None:
            return vx, vy
        if arm == "left":
            return -vy, vx  # 90 degrees
        if arm == "right":
            return vy, -vx  # 270 degrees
        return vx, vy

    def gamepad_intent(data, arm):
        """Validated analogue [forward, left] intent, or None near centre."""
        axes = data.get("axes") or {}
        raw = axes.get(arm) if isinstance(axes, dict) else None
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return None
        try:
            intent = np.asarray([float(raw[0]), float(raw[1])], dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite(intent)):
            return None
        magnitude = np.linalg.norm(intent)
        if magnitude <= 1e-6:
            return None
        if magnitude > 1.0:
            intent /= magnitude
        return intent

    async def drive_local_gamepad():
        """Drive directly from a controller plugged into the NUC."""
        claimed = set()
        armed = {arm: False for arm in sessions}
        direction_revision = controls["revision"]
        try:
            while True:
                sticks = gamepad.intents()
                active = set()
                busy = cal["busy"] or any(s["restarting"] for s in actions.values())
                direction_changed = direction_revision != controls["revision"]
                if direction_changed:
                    direction_revision = controls["revision"]
                for arm, sess in sessions.items():
                    stick = sticks.get(arm)
                    moving = stick is not None and np.linalg.norm(stick) > 1e-6
                    if busy or not gamepad.connected or direction_changed:
                        armed[arm] = False
                        if arm in claimed:
                            sess.set_intent(0.0, 0.0, frozen=True)
                    elif not moving:
                        # Require each stick to pass through centre after startup,
                        # reconnect, calibration, or an arm-stack restart.
                        armed[arm] = True
                        if arm in claimed:
                            sess.set_intent(0.0, 0.0, frozen=True)
                    elif armed[arm]:
                        active.add(arm)
                        vx, vy = player_controls(
                            arm, float(stick[0]), float(stick[1])
                        )
                        sess.set_intent(
                            vx,
                            vy,
                            frozen=False,
                        )
                    elif arm in claimed:
                        sess.set_intent(0.0, 0.0, frozen=True)
                claimed = active
                local_gamepad_active.clear()
                local_gamepad_active.update(active)
                await asyncio.sleep(1 / 50)
        except asyncio.CancelledError:
            pass
        finally:
            local_gamepad_active.clear()
            for arm in claimed:
                sessions[arm].set_intent(0.0, 0.0, frozen=True)

    def set_speed_limit(value):
        speed = float(value)
        if not np.isfinite(speed) or not 0.01 <= speed <= 1.0:
            raise ValueError("speed limit must be between 0.01 and 1.00 m/s")
        cfg["speed"] = speed
        for sess in sessions.values():
            sess.set_speed_limit(speed)
        print(f"[web] play speed limit set to {speed:.2f} m/s")

    async def restart_server(arm):
        if supervisor is None:
            return
        if arm not in sessions or actions.get(arm, {}).get("restarting"):
            return
        freeze_all()
        state = actions[arm]
        state.update({"restarting": True, "message": "Restarting arm stack..."})
        await broadcast()
        mode = sessions[arm].mode or PLAY

        def restart():
            sessions[arm].stop()
            supervisor.restart_arm(arm)
            if not sessions[arm].start(mode):
                raise RuntimeError(sessions[arm].error or "operator failed to restart")

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, restart)
            state["message"] = "Arm stack restarted; controls are frozen."
            print(f"[web] {arm} arm-stack restart complete")
        except Exception as exc:
            state["message"] = f"Restart failed: {type(exc).__name__}: {exc}"
            print(f"[web] {arm} {state['message']}")
        finally:
            state["restarting"] = False

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
                    action_busy = any(s["restarting"] for s in actions.values())
                    arm = cal["arm"]
                    if arm:
                        if arm in local_gamepad_active:
                            continue
                        k = resolve_keys(keymap[arm])
                        z = resolve_jog_z_keys(keymap[arm])
                        stick = gamepad_intent(data, arm)
                        if stick is None:
                            vx = float(k["up"] in held) - float(k["down"] in held)
                            vy = float(k["left"] in held) - float(k["right"] in held)
                        else:
                            vx, vy = stick
                        vz = float(z["up"] in held) - float(z["down"] in held)
                        sessions[arm].set_intent(
                            vx, vy, vz, frozen=frozen or cal["busy"] or action_busy
                        )
                    else:
                        home = KEY_HOME in held
                        for a, sess in sessions.items():
                            if a in local_gamepad_active:
                                continue
                            k = resolve_keys(keymap[a])
                            stick = gamepad_intent(data, a)
                            if stick is None:
                                vx = float(k["up"] in held) - float(k["down"] in held)
                                vy = float(k["left"] in held) - float(k["right"] in held)
                            else:
                                vx, vy = stick
                            vx, vy = player_controls(a, vx, vy)
                            sess.set_intent(
                                vx,
                                vy,
                                frozen=frozen or KEY_FREEZE in held or action_busy,
                                home=home,
                            )

                elif data.get("t") == "cmd":
                    cmd = data.get("cmd")
                    if cmd == "speed":
                        try:
                            set_speed_limit(data.get("speed"))
                        except (TypeError, ValueError) as exc:
                            print(f"[web] rejected speed limit: {exc}")
                    elif cmd == "same_side_controls":
                        freeze_all()
                        controls["same_side"] = bool(data.get("enabled", False))
                        controls["revision"] += 1
                        mode = "same-side" if controls["same_side"] else "normal"
                        print(f"[web] player control layout set to {mode}")
                    elif cmd == "restart_server":
                        await restart_server(data.get("arm"))
                    elif cmd == "calibrate":
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

    if gamepad is not None:
        async def start_gamepad(app):
            gamepad.start()
            app["gamepad_task"] = asyncio.create_task(drive_local_gamepad())

        async def stop_gamepad(app):
            task = app.get("gamepad_task")
            if task is not None:
                task.cancel()
                await task
            gamepad.stop()
            gamepad.join(timeout=2)

        app.on_startup.append(start_gamepad)
        app.on_cleanup.append(stop_gamepad)
    return app


def serve(sessions, cfg, supervisor=None, camera=None, port=8080, bind="127.0.0.1"):
    gamepad = LocalGamepad()
    app = build_app(
        sessions, cfg, supervisor=supervisor, camera=camera, gamepad=gamepad
    )
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
