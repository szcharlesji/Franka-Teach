"""Unified run.py WebUI commands, using in-memory sessions and supervisor."""

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from frankateach.airhockey.box import PlayBox  # noqa: E402
from frankateach.airhockey.operator import ArmStatus  # noqa: E402
from frankateach.airhockey.session import PLAY  # noqa: E402
from frankateach.airhockey.session_app import build_app  # noqa: E402

fails = []


def check(name, condition, extra=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name} {extra}")
    if not condition:
        fails.append(name)


class FakeSession:
    def __init__(self, arm, cfg):
        self.arm = arm
        self.cfg = cfg
        self.mode = PLAY
        self.provisional = False
        self.error = ""
        self.speed_limits = []
        self.intents = []
        self.stops = 0
        self.starts = 0
        self.box = PlayBox(
            center=np.array([0.4, 0.0]),
            yaw=0.0,
            half_extents=np.array([0.1, 0.2]),
            plane_z=0.2,
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.status = ArmStatus(
            pos=np.array([0.4, 0.0, 0.2]),
            box_pos=np.zeros(2),
            speed_limit=cfg["speed"],
            connected=True,
            stale=True,
        )

    def get_status(self):
        return self.status

    def set_intent(self, *args, **kwargs):
        self.intents.append((args, kwargs))

    def set_speed_limit(self, speed):
        self.speed_limits.append(float(speed))
        self.status.speed_limit = float(speed)

    def stop(self):
        self.stops += 1
        self.mode = None

    def start(self, mode):
        self.starts += 1
        self.mode = mode
        return True


class FakeSupervisor:
    def __init__(self):
        self.restarts = []

    def restart_server(self, arm):
        self.restarts.append(arm)

    def status(self):
        return {
            "left": {
                "interface": "up",
                "server": "up",
                "port_bound": True,
                "port": 9001,
            }
        }


async def newest_status(ws, timeout=2.0):
    latest = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=0.1)
        except asyncio.TimeoutError:
            if latest is not None:
                return latest
            continue
        if msg.type == aiohttp.WSMsgType.TEXT:
            data = json.loads(msg.data)
            if data.get("t") == "status":
                latest = data
    return latest


async def main():
    cfg = {
        "control_hz": 50,
        "speed": 0.35,
        "watchdog": 0.1,
        "arms": {"left": {"keys": "wasd"}},
    }
    session = FakeSession("left", cfg)
    supervisor = FakeSupervisor()
    app = build_app({"left": session}, cfg, supervisor=supervisor)
    client = TestClient(TestServer(app))
    await client.start_server()

    response = await client.get("/")
    html = await response.text()
    check("unified page includes speed control", "Apply speed" in html)
    check("unified page includes server restart", "data-restart" in html)

    ws = await client.ws_connect("/ws")
    await ws.send_json({"t": "cmd", "cmd": "speed", "speed": 0.12})
    status = await newest_status(ws)
    check("speed command updates config", cfg["speed"] == 0.12)
    check("speed command reaches live session", session.speed_limits == [0.12])
    check("status publishes current cap", status["speed_limit"] == 0.12)

    await ws.send_json({"t": "cmd", "cmd": "speed", "speed": 2.0})
    await asyncio.sleep(0.1)
    check("out-of-range speed is rejected", cfg["speed"] == 0.12)

    await ws.send_json({"t": "cmd", "cmd": "restart_server", "arm": "left"})
    deadline = time.time() + 2.0
    restarted = None
    while time.time() < deadline:
        restarted = await newest_status(ws, timeout=0.2)
        action = restarted and restarted["arms"]["left"]["restart"]
        if action and "restarted" in action["message"].lower():
            break
    check("restart recycles the selected server", supervisor.restarts == ["left"])
    check("restart rebuilds the arm session", session.stops == 1 and session.starts == 1)
    check(
        "restart leaves controls frozen",
        bool(session.intents and session.intents[0][1].get("frozen")),
    )

    await ws.close()
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
    print()
    print("FAILED:", fails if fails else "none")
    sys.exit(1 if fails else 0)
