"""Hardware-free integration tests for the air hockey teleop loop.

Runs the real ArmOperator against tests/fake_server.py, so it covers the
REQ/REP protocol, reset, homing, ramping, box clipping, the leash, the
watchdog, loop pacing, state publishing, and two-arm thread independence --
everything except deoxys and the physical robot. Run directly:

    python3 tests/test_teleop.py
"""

import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import yaml
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frankateach.airhockey import config as ahconfig  # noqa: E402
from frankateach.airhockey.box import box_from_corners  # noqa: E402
from frankateach.airhockey.operator import ArmOperator  # noqa: E402
from frankateach.constants import arm_ports  # noqa: E402
from frankateach.network import ZMQKeypointSubscriber  # noqa: E402
from tests.fake_server import FakeServer  # noqa: E402

CFG = {
    "control_hz": 50,
    "speed": 0.35,
    "accel_time": 0.08,
    "max_lead": 0.05,
    "watchdog": 0.1,
}
CORNERS = np.array(
    [[0.30, -0.24, 0.12], [0.62, -0.24, 0.12], [0.62, 0.24, 0.12], [0.30, 0.24, 0.12]]
)
QUAT = [1.0, 0.0, 0.0, 0.0]

fails = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        fails.append(name)


def make_box():
    box, _ = box_from_corners(CORNERS, QUAT, margin=0.015)
    return box


def test_single_arm():
    print("\n-- single arm --")
    box = make_box()
    server = FakeServer("right")
    server.start()
    server.ready.wait(2)
    sub = ZMQKeypointSubscriber("localhost", arm_ports("right")[1], "robot_state")
    op = ArmOperator("right", box, CFG, publish=True)
    op.start()

    check("operator became ready", op.wait_ready(30))
    time.sleep(0.3)
    check("operator connected", op.get_status().connected, f"({op.get_status().error})")
    check("performed a joint reset", server.resets == 1, f"(resets={server.resets})")
    check(
        "startup and play never command the gripper",
        bool(server.gripper_commands)
        and all(command is None for command in server.gripper_commands),
    )
    check(
        "homed to the box centre",
        np.allclose(server.pos[:2], box.center, atol=0.01),
        f"(pos={server.pos[:2]})",
    )
    check("z pinned to the plane", abs(server.pos[2] - box.plane_z) < 0.005)

    published = None
    for _ in range(50):
        published = sub.recv_keypoints(flags=zmq.NOBLOCK)
        if published is not None:
            break
        time.sleep(0.02)
    check("publishes robot_state for collect_data.py", published is not None)

    # Drive hard into a corner for 3 s.
    n0 = len(server.commands)
    deadline = time.time() + 3.0
    while time.time() < deadline:
        op.set_intent(1.0, 1.0)
        time.sleep(0.02)
    cmds = np.array(server.commands[n0:])
    inside = [box.contains(c[:2], tol=1e-6) for c in cmds]
    check(
        "every command stayed inside the box",
        all(inside),
        f"({sum(inside)}/{len(inside)})",
    )
    check("z never left the plane", np.allclose(cmds[:, 2], box.plane_z, atol=1e-9))
    local = box.to_box(server.pos[:2])
    check(
        "reached the corner it was driven at",
        np.allclose(np.abs(local), box.half_extents, atol=0.01),
        f"(box coords {local})",
    )

    # Watchdog: go silent for longer than `watchdog` and the arm must stop.
    op.set_intent(-1.0, 0.0)
    time.sleep(0.15)
    before = server.pos.copy()
    time.sleep(0.8)  # no set_intent calls at all
    moved = np.linalg.norm(server.pos[:2] - before[:2])
    check("watchdog froze the arm on stale intent", moved < 0.004, f"({moved * 1000:.2f} mm)")
    check("status reports stale", op.get_status().stale)

    op.set_intent(0.0, 0.0, frozen=True)
    time.sleep(0.6)
    rate = op.get_status().rate
    check("loop holds ~50 Hz", 45 <= rate <= 55, f"({rate:.1f} Hz)")

    deadline = time.time() + 4.0
    while time.time() < deadline:
        op.set_intent(0.0, 0.0, home=True)
        time.sleep(0.02)
    check(
        "H returns to the box centre",
        np.allclose(server.pos[:2], box.center, atol=0.01),
        f"(pos={server.pos[:2]})",
    )

    # Diagonals must not be faster than straight moves.
    def peak_speed(vx, vy, seconds=1.2):
        op.set_intent(0.0, 0.0, home=True)
        time.sleep(1.5)
        peak = 0.0
        deadline = time.time() + seconds
        while time.time() < deadline:
            op.set_intent(vx, vy)
            peak = max(peak, op.get_status().speed)
            time.sleep(0.01)
        return peak

    straight = peak_speed(1.0, 0.0)
    diagonal = peak_speed(1.0, 1.0)
    check(
        "diagonal is not faster than straight",
        abs(diagonal - straight) < 0.02,
        f"(straight {straight:.3f} m/s, diagonal {diagonal:.3f} m/s)",
    )
    check(
        "speed reaches the configured maximum",
        abs(straight - CFG["speed"]) < 0.02,
        f"({straight:.3f} vs {CFG['speed']})",
    )

    # The run.py WebUI changes this while the operator is live. Lowering it is
    # a hard cap, including when the arm was already moving faster.
    op.set_speed_limit(0.08)
    op.set_intent(1.0, 0.0)
    time.sleep(0.08)
    limited = op.get_status()
    check(
        "live speed-limit update takes effect immediately",
        limited.speed <= 0.081 and limited.speed_limit == 0.08,
        f"(speed={limited.speed:.3f}, limit={limited.speed_limit:.3f})",
    )

    op.stop()
    op.join(timeout=5)
    check("operator thread exited", not op.is_alive())
    sub.stop()
    server.stop.set()
    server.join(timeout=3)  # must fully release the port before the next test binds it


def test_bimanual():
    print("\n-- two arms + render load --")
    boxes, servers, ops = {}, {}, {}
    for arm in ("right", "left"):
        boxes[arm] = make_box()
        # Give the left arm a deliberately sluggish server.
        servers[arm] = FakeServer(arm, work=0.004 if arm == "left" else 0.0)
        servers[arm].start()
        servers[arm].ready.wait(2)
        ops[arm] = ArmOperator(arm, boxes[arm], CFG, publish=False)
        ops[arm].start()
    for arm, op in ops.items():
        check(f"{arm} ready", op.wait_ready(30))

    stop = threading.Event()

    def render():  # stand in for the pygame thread
        while not stop.is_set():
            np.linalg.svd(np.random.rand(60, 60))
            time.sleep(1 / 60)

    threading.Thread(target=render, daemon=True).start()

    deadline = time.time() + 5.0
    while time.time() < deadline:
        ops["right"].set_intent(1.0, 0.0)
        ops["left"].set_intent(0.0, 1.0)
        time.sleep(0.01)

    for arm in ("right", "left"):
        st = ops[arm].get_status()
        check(f"{arm} holds rate under load", 44 <= st.rate <= 56, f"({st.rate:.1f} Hz)")
        check(f"{arm} stayed in its box", boxes[arm].contains(st.pos[:2], tol=1e-6))

    nr, nl = len(servers["right"].commands), len(servers["left"].commands)
    check(
        "a slow arm does not starve the other",
        abs(nr - nl) < 0.15 * max(nr, nl),
        f"(right={nr}, left={nl})",
    )

    stop.set()
    for op in ops.values():
        op.stop()
    for op in ops.values():
        op.join(timeout=5)
    for s in servers.values():
        s.stop.set()
    for s in servers.values():
        s.join(timeout=3)


def test_config_roundtrip():
    print("\n-- config --")
    src = Path(__file__).resolve().parent.parent / "configs" / "airhockey.yaml"
    tmp = tempfile.mktemp(suffix=".yaml")
    shutil.copy(src, tmp)
    try:
        # Blank the arm first. This used to just assume the right arm was
        # uncalibrated in the real config, which quietly became false the moment
        # someone calibrated it -- the test then failed for a reason that had
        # nothing to do with the code under test.
        blanked = yaml.safe_load(open(tmp))
        blanked["arms"]["right"] = {"keys": "ijkl", "center": None}
        with open(tmp, "w") as f:
            yaml.safe_dump(blanked, f, sort_keys=False)
        try:
            ahconfig.load_box("right", tmp)
            check("uncalibrated arm raises a helpful error", False)
        except RuntimeError as exc:
            check("uncalibrated arm raises a helpful error", "--calibrate" in str(exc))

        box = make_box()
        ahconfig.save_box("right", box, tmp)
        rt = ahconfig.load_box("right", tmp)
        check(
            "saved box round trips",
            np.allclose(rt.center, box.center)
            and np.allclose(rt.half_extents, box.half_extents)
            and abs(rt.plane_z - box.plane_z) < 1e-12
            and abs(rt.yaw - box.yaw) < 1e-12,
        )
        after = yaml.safe_load(open(tmp))
        check("the other arm's block is untouched", after["arms"]["left"]["keys"] == "wasd")
        check(
            "top-level settings preserved",
            after["control_hz"] == 50 and after["speed"] == 0.35,
        )
    finally:
        os.unlink(tmp)


if __name__ == "__main__":
    test_single_arm()
    test_bimanual()
    test_config_roundtrip()
    print()
    print("FAILED:", fails if fails else "none")
    sys.exit(1 if fails else 0)
