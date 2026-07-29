"""End-to-end test of the browser teleop stack, without a robot or a browser.

Starts the real aiohttp app against tests/fake_server.py, then drives it over a
real websocket exactly as the page does. Covers the routes, the full-key-state
protocol, and the two network safety properties that matter now that the UI is
remote: a dropped keyup must not leave the arm running, and a dropped
*connection* must freeze it.

    python3 tests/test_webapp.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from frankateach.airhockey.box import box_from_corners  # noqa: E402
from frankateach.airhockey.operator import ArmOperator  # noqa: E402
from frankateach.airhockey.webapp import build_app  # noqa: E402
from tests.fake_server import FakeServer  # noqa: E402

CFG = {
    "control_hz": 50,
    "speed": 0.35,
    "accel_time": 0.08,
    "max_lead": 0.05,
    "watchdog": 0.1,
    "view_cam_id": None,
    "arms": {"left": {"keys": "wasd"}, "right": {"keys": "arrows"}},
}
CORNERS = np.array(
    [[0.30, -0.24, 0.12], [0.62, -0.24, 0.12], [0.62, 0.24, 0.12], [0.30, 0.24, 0.12]]
)

fails = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        fails.append(name)


async def main():
    box, _ = box_from_corners(CORNERS, [1.0, 0.0, 0.0, 0.0], margin=0.015)
    server = FakeServer("left")
    server.start()
    server.ready.wait(2)

    op = ArmOperator("left", box, CFG, publish=False)
    op.start()
    check("operator ready", op.wait_ready(30))

    app = build_app({"left": op}, {"left": box}, CFG, None)
    client = TestClient(TestServer(app))
    await client.start_server()

    # -- routes ------------------------------------------------------------
    r = await client.get("/")
    html = await r.text()
    check("GET / serves the page", r.status == 200 and "FRANKA AIR HOCKEY" in html)

    r = await client.get("/config")
    cfg_json = await r.json()
    check("GET /config lists arms", cfg_json["arms"] == ["left"], f"({cfg_json['arms']})")
    check(
        "config exposes wasd for the left arm",
        cfg_json["keys"]["left"]["up"] == "KeyW",
        f"({cfg_json['keys']['left']})",
    )

    r = await client.get("/video")
    check("GET /video 404s with no camera", r.status == 404, f"({r.status})")

    # -- websocket ---------------------------------------------------------
    ws = await client.ws_connect("/ws")

    async def recv_status(timeout=2.0):
        """Newest status, draining anything the server queued while we drove.

        The server pushes at 20 Hz whether or not we are reading, so the first
        message waiting is stale by however long we were busy.
        """
        latest = None
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=0.05)
            except asyncio.TimeoutError:
                if latest is not None:
                    return latest
                continue
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("t") == "status":
                    latest = data
        return latest

    st = await recv_status()
    check("websocket pushes status", st is not None and "left" in st["arms"])
    check("status reports connected", st["arms"]["left"]["connected"])

    async def drive(held, seconds, frozen=False, home=False):
        deadline = time.time() + seconds
        while time.time() < deadline:
            await ws.send_json({"t": "keys", "held": held, "frozen": frozen, "home": home})
            await asyncio.sleep(0.02)

    # Drive up-table and confirm the arm actually moves.
    start = server.pos.copy()
    await drive(["KeyW"], 1.5)
    moved = np.linalg.norm(server.pos[:2] - start[:2])
    check("held key moves the arm", moved > 0.05, f"(moved {moved * 1000:.0f} mm)")

    st = await recv_status()
    check("status u tracks up-table motion", st["arms"]["left"]["u"] > 0.1,
          f"(u={st['arms']['left']['u']:.3f})")

    # Drive into the corner; the box must still hold over the network path.
    await drive(["KeyW", "KeyA"], 2.5)
    local = box.to_box(server.pos[:2])
    check(
        "box still clips when driven over the websocket",
        box.contains(server.pos[:2], tol=1e-6),
        f"(box coords {local}, extents {box.half_extents})",
    )

    # -- THE network safety property: a lost keyup must not run away --------
    # Simulate it exactly: stop sending anything at all while a key was held.
    # By design the arm keeps moving for up to `watchdog` seconds before it
    # gives up, so there are two things to assert: the coast is bounded, and
    # once the window passes the arm is genuinely stopped.
    at_silence = server.pos.copy()
    await ws.send_json({"t": "keys", "held": ["KeyS"], "frozen": False, "home": False})
    await asyncio.sleep(0.35)  # radio silence, well past the 100 ms watchdog
    coast = np.linalg.norm(server.pos[:2] - at_silence[:2])
    bound = CFG["speed"] * CFG["watchdog"] + 0.005
    check(
        "coast after the link dies is bounded by speed x watchdog",
        coast <= bound,
        f"(coasted {coast * 1000:.1f} mm, bound {bound * 1000:.1f} mm)",
    )

    settled = server.pos.copy()
    await asyncio.sleep(0.6)  # still silent
    drift = np.linalg.norm(server.pos[:2] - settled[:2])
    check(
        "arm is fully stopped once the watchdog window passes",
        drift < 0.001,
        f"(drifted {drift * 1000:.3f} mm)",
    )
    check(
        "coasting never left the box",
        box.contains(server.pos[:2], tol=1e-6),
        f"(box coords {box.to_box(server.pos[:2])})",
    )

    # -- dropping the connection must freeze, not coast --------------------
    await drive(["KeyS"], 0.4)
    await ws.close()
    await asyncio.sleep(0.1)
    before = server.pos.copy()
    await asyncio.sleep(0.6)
    drift = np.linalg.norm(server.pos[:2] - before[:2])
    check(
        "closing the websocket freezes the arm",
        drift < 0.004,
        f"(drifted {drift * 1000:.2f} mm)",
    )

    # -- reconnect works ---------------------------------------------------
    ws2 = await client.ws_connect("/ws")
    got = None
    deadline = time.time() + 2
    while time.time() < deadline and got is None:
        msg = await asyncio.wait_for(ws2.receive(), timeout=2)
        if msg.type == aiohttp.WSMsgType.TEXT:
            d = json.loads(msg.data)
            if d.get("t") == "status":
                got = d
    check("reconnect resumes status", got is not None and got["arms"]["left"]["connected"])
    await ws2.close()

    rate = op.get_status().rate
    check("loop held ~50 Hz throughout", 44 <= rate <= 56, f"({rate:.1f} Hz)")

    await client.close()
    op.stop()
    op.join(timeout=5)
    server.stop.set()
    server.join(timeout=3)


if __name__ == "__main__":
    asyncio.run(main())
    print()
    print("FAILED:", fails if fails else "none")
    sys.exit(1 if fails else 0)
