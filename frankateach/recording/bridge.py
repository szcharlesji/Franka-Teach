"""NUC-side, loopback-only control and telemetry bridge."""

import asyncio
import hashlib
import json
import queue
import socket
import subprocess
import time
from pathlib import Path

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
from frankateach.recording.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    jsonable,
    require_message,
    validate_hello,
    validate_keys,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root):
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except Exception as exc:
        return {"revision": None, "dirty": None, "error": str(exc)}


class TelemetryHub:
    """A bounded handoff that never makes an ArmOperator wait on networking."""

    def __init__(self, maxsize=4096):
        self.queue = queue.Queue(maxsize=maxsize)
        self.subscribed = False

    def publish(self, event):
        if not self.subscribed:
            return True
        try:
            self.queue.put_nowait(event)
            return True
        except queue.Full:
            return False

    def clear(self):
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return


class RobotBridge:
    def __init__(self, sessions, cfg, supervisor, telemetry):
        self.sessions = sessions
        self.cfg = cfg
        self.supervisor = supervisor
        self.telemetry = telemetry
        self.active_ws = None
        self.cal = {
            "arm": None,
            "corners": [],
            "message": "",
            "summary": None,
            "busy": False,
        }
        self.keymap = {
            arm: resolve_keys(str(cfg.get("arms", {}).get(arm, {}).get("keys", "wasd")))
            for arm in sessions
        }
        repo_root = Path(__file__).resolve().parents[2]
        serialized_cfg = json.dumps(jsonable(cfg), sort_keys=True).encode("utf-8")
        arm_configs = {}
        for arm, stack in supervisor.stacks.items():
            client_path = repo_root / "frankateach" / "configs" / stack.deoxys_config
            arm_configs[arm] = {
                "client": str(client_path),
                "client_sha256": _sha256(client_path),
                "nuc": str(stack.nuc_config),
                "nuc_sha256": _sha256(stack.nuc_config),
                "control_freq": stack.control_freq,
                "num_steps": stack.num_steps,
            }
        self.host_metadata = {
            "hostname": socket.gethostname(),
            "git": _git_state(repo_root),
            "configuration": {
                "runtime_sha256": hashlib.sha256(serialized_cfg).hexdigest(),
                "control_hz": float(cfg["control_hz"]),
                "arms": arm_configs,
            },
        }

    def freeze_all(self):
        for session in self.sessions.values():
            if session.mode == CALIBRATE:
                session.set_intent(0.0, 0.0, 0.0, frozen=True)
            else:
                session.set_intent(0.0, 0.0, frozen=True)

    def snapshot(self):
        arms = {}
        for arm, session in self.sessions.items():
            status = session.get_status()
            box = session.box
            arms[arm] = {
                "mode": session.mode,
                "connected": bool(status.connected) if status else False,
                "stale": bool(status.stale) if status else True,
                "error": (status.error if status else "") or session.error,
                "rate_hz": float(status.rate) if status else 0.0,
                "pos": jsonable(status.pos) if status else [0.0, 0.0, 0.0],
                "box_pos": jsonable(getattr(status, "box_pos", np.zeros(2))),
                "speed": float(getattr(status, "speed", 0.0)),
                "homing": bool(getattr(status, "homing", False)),
                "provisional": bool(session.provisional),
                "half_extents": jsonable(box.half_extents) if box is not None else None,
                "yaw": float(box.yaw) if box is not None else None,
            }
        return {
            "t": "status",
            "protocol_version": PROTOCOL_VERSION,
            "nuc_mono_ns": time.perf_counter_ns(),
            "control_hz": float(self.cfg["control_hz"]),
            "arms": arms,
            "supervisor": self.supervisor.status(),
            "calibration": jsonable(self.cal),
            "host": self.host_metadata,
        }

    def handle_keys(self, raw):
        data = validate_keys(raw)
        held = set(data["held"])
        if self.cal["arm"]:
            arm = self.cal["arm"]
            keys = self.keymap[arm]
            zkeys = resolve_jog_z_keys(
                str(self.cfg.get("arms", {}).get(arm, {}).get("keys", "wasd"))
            )
            vx = float(keys["up"] in held) - float(keys["down"] in held)
            vy = float(keys["left"] in held) - float(keys["right"] in held)
            vz = float(zkeys["up"] in held) - float(zkeys["down"] in held)
            self.sessions[arm].set_intent(vx, vy, vz, frozen=data["frozen"])
            return

        for arm, session in self.sessions.items():
            keys = self.keymap[arm]
            vx = float(keys["up"] in held) - float(keys["down"] in held)
            vy = float(keys["left"] in held) - float(keys["right"] in held)
            session.set_intent(
                vx,
                vy,
                frozen=data["frozen"] or KEY_FREEZE in held,
                home=data["home"] or KEY_HOME in held,
                sequence=data["sequence"],
                source_stamp_ns=data["discovery_mono_ns"],
            )

    async def handle_admin(self, data):
        command = str(data.get("command") or "")
        self.freeze_all()
        if command == "speed":
            speed = float(data["speed"])
            if not 0.01 <= speed <= 1.0:
                raise ValueError("speed must be in [0.01, 1.0] m/s")
            self.cfg["speed"] = speed
            for session in self.sessions.values():
                session.set_speed_limit(speed)
            return {"message": f"speed set to {speed:.2f} m/s"}
        if command == "restart":
            arm = str(data.get("arm"))
            if arm not in self.sessions:
                raise ValueError(f"unknown arm {arm!r}")
            mode = self.sessions[arm].mode or PLAY

            def restart():
                self.sessions[arm].stop()
                self.supervisor.restart_arm(arm)
                if not self.sessions[arm].start(mode):
                    raise RuntimeError(self.sessions[arm].error)

            await asyncio.to_thread(restart)
            return {"message": f"{arm} stack restarted; controls remain frozen"}
        if command == "calibrate":
            arm = str(data.get("arm"))
            if arm not in self.sessions:
                raise ValueError(f"unknown arm {arm!r}")
            ok = await asyncio.to_thread(self.sessions[arm].set_mode, CALIBRATE)
            if not ok:
                raise RuntimeError(self.sessions[arm].error)
            self.cal.update(
                {"arm": arm, "corners": [], "message": "", "summary": None, "busy": False}
            )
            return {"message": f"calibrating {arm}"}
        if command == "play":
            arm = self.cal["arm"]
            if arm is not None and not await asyncio.to_thread(self.sessions[arm].reload_box):
                raise RuntimeError(self.sessions[arm].error)
            self.cal.update(
                {"arm": None, "corners": [], "message": "", "summary": None, "busy": False}
            )
            return {"message": "play mode; controls remain frozen"}
        if self.cal["arm"] is None:
            raise ValueError("not in calibration mode")
        arm = self.cal["arm"]
        if command == "record_corner":
            if len(self.cal["corners"]) >= 4:
                raise ValueError("already have four corners")
            status = self.sessions[arm].get_status()
            if status is None or not status.connected:
                raise RuntimeError("arm is not connected")
            self.cal["corners"].append(np.asarray(status.pos, dtype=np.float64).tolist())
            return {"message": f"recorded corner {len(self.cal['corners'])}"}
        if command == "undo_corner":
            if self.cal["corners"]:
                self.cal["corners"].pop()
            self.cal["summary"] = None
            return {"message": "removed last corner"}
        if command == "finish_calibration":
            if len(self.cal["corners"]) != 4:
                raise ValueError("exactly four corners are required")
            status = self.sessions[arm].get_status()
            box, warnings = box_from_corners(
                np.asarray(self.cal["corners"], dtype=np.float64),
                status.quat,
                margin=float(self.cfg["margin"]),
                plane_mode=str(self.cfg.get("plane_mode", "constant")),
                level_wrist=bool(self.cfg.get("level_wrist", False)),
            )
            path = ahconfig.save_box(arm, box)
            self.cal["summary"] = {
                "box": box.to_dict(),
                "warnings": warnings,
                "saved_to": str(path),
            }
            return {
                "message": f"saved calibration to {path}",
                "summary": self.cal["summary"],
            }
        if command == "trace":
            if self.cal["summary"] is None:
                raise ValueError("finish calibration before tracing")
            self.cal["busy"] = True
            try:
                box = ahconfig.load_box(arm)
                await asyncio.to_thread(
                    trace_perimeter,
                    arm,
                    box,
                    self.cfg,
                    self.sessions[arm].operator.link,
                )
            finally:
                self.cal["busy"] = False
            return {"message": "trace complete"}
        raise ValueError(f"unknown admin command {command!r}")

    async def websocket(self, request):
        if self.active_ws is not None and not self.active_ws.closed:
            raise web.HTTPConflict(text="a Discovery client already owns this bridge")
        ws = web.WebSocketResponse(heartbeat=5)
        await ws.prepare(request)
        self.active_ws = ws
        self.telemetry.clear()
        self.telemetry.subscribed = True

        try:
            first = await ws.receive(timeout=5)
            if first.type != WSMsgType.TEXT:
                raise ProtocolError("first message must be hello")
            session = validate_hello(json.loads(first.data))
            await ws.send_json(
                {"t": "hello", "protocol_version": PROTOCOL_VERSION, "session": session}
            )
        except Exception as exc:
            await ws.send_json({"t": "error", "message": str(exc)})
            await ws.close()
            self.telemetry.subscribed = False
            self.freeze_all()
            self.active_ws = None
            return ws

        async def push():
            next_status = 0.0
            try:
                while not ws.closed:
                    batch = []
                    for _ in range(64):
                        try:
                            item = self.telemetry.queue.get_nowait()
                        except queue.Empty:
                            break
                        session_obj = self.sessions.get(item["arm"])
                        item["provisional"] = (
                            bool(session_obj.provisional) if session_obj else True
                        )
                        batch.append(item)
                    if batch:
                        await ws.send_json({"t": "telemetry_batch", "items": batch})
                    now = time.perf_counter()
                    if now >= next_status:
                        await ws.send_json(self.snapshot())
                        next_status = now + 0.1
                    await asyncio.sleep(0.002 if batch else 0.01)
            except (asyncio.CancelledError, ConnectionError):
                pass

        pusher = asyncio.create_task(push())
        try:
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    continue
                remote_recv_ns = time.perf_counter_ns()
                data = {}
                try:
                    data = json.loads(message.data)
                    kind = require_message(data)
                    if kind == "keys":
                        self.handle_keys(data)
                    elif kind == "clock_probe":
                        remote_send_ns = time.perf_counter_ns()
                        await ws.send_json(
                            {
                                "t": "clock_reply",
                                "probe_id": data.get("probe_id"),
                                "local_send_ns": data.get("local_send_ns"),
                                "remote_recv_ns": remote_recv_ns,
                                "remote_send_ns": remote_send_ns,
                            }
                        )
                    elif kind == "heartbeat":
                        await ws.send_json(
                            {
                                "t": "heartbeat_ack",
                                "probe_id": data.get("probe_id"),
                                "local_send_ns": data.get("local_send_ns"),
                                "remote_mono_ns": time.perf_counter_ns(),
                            }
                        )
                    elif kind == "admin":
                        result = await self.handle_admin(data)
                        await ws.send_json(
                            {
                                "t": "admin_result",
                                "request_id": data.get("request_id"),
                                **result,
                            }
                        )
                    else:
                        raise ProtocolError(f"unsupported message {kind!r}")
                except Exception as exc:
                    await ws.send_json(
                        {
                            "t": "error",
                            "request_id": data.get("request_id")
                            if isinstance(data, dict)
                            else None,
                            "message": f"{type(exc).__name__}: {exc}",
                        }
                    )
        finally:
            pusher.cancel()
            await asyncio.gather(pusher, return_exceptions=True)
            self.telemetry.subscribed = False
            self.telemetry.clear()
            self.freeze_all()
            self.active_ws = None
        return ws

    def app(self):
        app = web.Application()
        app.router.add_get("/health", lambda _: web.json_response(self.snapshot()))
        app.router.add_get("/ws", self.websocket)

        async def cleanup(_):
            self.telemetry.subscribed = False
            self.freeze_all()

        app.on_cleanup.append(cleanup)
        return app
