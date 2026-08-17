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
    telemetry = []
    op = ArmOperator(
        "right", box, CFG, publish=True, telemetry_callback=telemetry.append
    )
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
    op.set_intent(1.0, 0.0, sequence=23, source_stamp_ns=456)
    time.sleep(0.06)
    check(
        "recording observes the applied intent sequence",
        bool(telemetry) and telemetry[-1]["intent_sequence"] == 23,
    )
    check(
        "recording observes post-ramp absolute EE targets",
        bool(telemetry)
        and len(telemetry[-1]["commanded_pos"]) == 3
        and len(telemetry[-1]["commanded_box_xy"]) == 2,
    )

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


def test_inward_speed_scale():
    """Inward (-x, toward the base) must be capped at the configured fraction.

    Measured as a steady-state speed through the real loop rather than by
    reading the intent back, so it covers the ramp, the box clip and the leash
    as well -- the cap is worthless if any of those undo it.
    """
    print("\n-- inward speed scale --")
    scale = 0.6
    box = make_box()
    server = FakeServer("right")
    server.start()
    op = ArmOperator("right", box, dict(CFG, inward_speed_scale=scale), publish=False)
    op.start()
    try:
        if not op.wait_ready(timeout=20):
            check("operator came up", False)
            return

        def sweep(vx, seconds=1.2):
            """Mean commanded speed along box x while holding one key."""
            # Park against the opposite edge first, so the sweep has full travel.
            deadline = time.time() + 1.5
            while time.time() < deadline:
                op.set_intent(-1.0 if vx > 0 else 1.0, 0.0)
                time.sleep(0.02)
            n0 = len(server.commands)
            t0 = time.time()
            deadline = t0 + seconds
            while time.time() < deadline:
                op.set_intent(vx, 0.0)
                time.sleep(0.02)
            cmds = np.array(server.commands[n0:])
            local = np.array([box.to_box(c[:2]) for c in cmds])
            # Ignore the ramp: take the fastest sustained window.
            dx = np.abs(np.diff(local[:, 0]))
            return float(np.mean(np.sort(dx)[-len(dx) // 3 :])) * CFG["control_hz"]

        outward = sweep(+1.0)
        inward = sweep(-1.0)
        ratio = inward / outward if outward else 0.0
        check(
            "outward sweep runs at the configured speed",
            abs(outward - CFG["speed"]) < 0.05,
            f"({outward:.3f} m/s vs {CFG['speed']})",
        )
        check(
            f"inward sweep is capped at {scale}x",
            abs(ratio - scale) < 0.08,
            f"(ratio {ratio:.3f}, inward {inward:.3f} m/s)",
        )
    finally:
        op.stop()
        op.join(timeout=5)
        server.stop.set()
        server.join(timeout=3)

    # A typo that would speed the arm up must not be silently accepted.
    for bad in (0.0, -0.5, 6.0):
        try:
            ArmOperator("right", box, dict(CFG, inward_speed_scale=bad), publish=False)
            check(f"inward_speed_scale={bad} rejected", False)
        except ValueError:
            check(f"inward_speed_scale={bad} rejected", True)


def test_lateral_speed_scale():
    """Left/right must be capped symmetrically, and independently of inward.

    Both y directions are pure joint-1 rotation, so unlike the inward cap this
    one is not sign-dependent -- a scale that only bit one way would leave the
    asymmetry it exists to remove.
    """
    print("\n-- lateral speed scale --")
    scale = 0.5
    box = make_box()
    server = FakeServer("right")
    server.start()
    op = ArmOperator(
        "right",
        box,
        # inward left at 1.0 so a lateral cap leaking into x shows up.
        dict(CFG, lateral_speed_scale=scale, inward_speed_scale=1.0),
        publish=False,
    )
    op.start()
    try:
        if not op.wait_ready(timeout=20):
            check("operator came up", False)
            return

        def sweep(vx, vy, seconds=1.2):
            """Mean commanded speed along the driven box axis."""
            axis = 0 if vx else 1
            # Park against the opposite edge first, so the sweep has full travel.
            deadline = time.time() + 1.5
            while time.time() < deadline:
                op.set_intent(-vx, -vy)
                time.sleep(0.02)
            n0 = len(server.commands)
            deadline = time.time() + seconds
            while time.time() < deadline:
                op.set_intent(vx, vy)
                time.sleep(0.02)
            cmds = np.array(server.commands[n0:])
            local = np.array([box.to_box(c[:2]) for c in cmds])
            # Ignore the ramp: take the fastest sustained window.
            step = np.abs(np.diff(local[:, axis]))
            return float(np.mean(np.sort(step)[-len(step) // 3 :])) * CFG["control_hz"]

        outward = sweep(+1.0, 0.0)
        left = sweep(0.0, +1.0)
        right = sweep(0.0, -1.0)
        check(
            "outward sweep is untouched by the lateral cap",
            abs(outward - CFG["speed"]) < 0.05,
            f"({outward:.3f} m/s vs {CFG['speed']})",
        )
        for name, got in (("+y", left), ("-y", right)):
            ratio = got / outward if outward else 0.0
            check(
                f"{name} sweep is capped at {scale}x",
                abs(ratio - scale) < 0.08,
                f"(ratio {ratio:.3f}, {got:.3f} m/s)",
            )
    finally:
        op.stop()
        op.join(timeout=5)
        server.stop.set()
        server.join(timeout=3)

    for bad in (0.0, -0.5, 6.0):
        try:
            ArmOperator("right", box, dict(CFG, lateral_speed_scale=bad), publish=False)
            check(f"lateral_speed_scale={bad} rejected", False)
        except ValueError:
            check(f"lateral_speed_scale={bad} rejected", True)


def test_joint1_budget():
    """The joint-1 budget must bind near the base and stay out of the way far out.

    CORNERS spans x = 0.30..0.62, so the ceiling (budget * r) more than doubles
    across the box -- the whole reason a single `speed` cannot be right
    everywhere.
    """
    print("\n-- joint-1 budget --")
    box = make_box()
    fraction = 0.7
    gain = 9.35
    budget = fraction * 2.175
    # max_lead is chosen to sit between the inner and outer leash caps, so the
    # tightening is observable in both directions on this box.
    max_lead = 0.07
    cfg = dict(
        CFG,
        joint1_speed_fraction=fraction,
        lead_speed_gain=gain,
        speed=0.6,
        max_lead=max_lead,
    )
    op = ArmOperator("right", box, cfg, publish=False)

    # Geometry only -- no robot needed for the two helpers.
    inner = box.to_world([-box.half_extents[0], 0.0])
    outer = box.to_world([box.half_extents[0], 0.0])
    ceil_in = op._joint1_ceiling(inner)
    ceil_out = op._joint1_ceiling(outer)
    check(
        "ceiling rises with radius",
        ceil_out > ceil_in,
        f"(inner {ceil_in:.3f} < outer {ceil_out:.3f} m/s)",
    )
    check(
        "ceiling equals budget * radius",
        abs(ceil_in - budget * np.hypot(*inner)) < 1e-9,
    )
    # The leash must shrink where the ceiling is low. That is the part that caps
    # PEAK speed; clamping the commanded velocity alone would not.
    check(
        "leash is tightened at the inner edge",
        op._lead_limit(inner) < max_lead - 1e-9,
        f"({op._lead_limit(inner) * 1000:.1f} mm vs {max_lead * 1000:.0f} mm)",
    )
    check(
        "leash is untouched where there is room",
        abs(op._lead_limit(outer) - max_lead) < 1e-9,
        f"({op._lead_limit(outer) * 1000:.1f} mm)",
    )
    # Tangential motion is what turns joint 1; radial motion barely does.
    tangential = np.array([0.0, 0.6])
    radial = np.array([0.6, 0.0])
    check(
        "tangential velocity is clamped at the inner edge",
        np.linalg.norm(op._joint1_clamp(tangential, inner)) < 0.6 - 1e-6,
        f"({np.linalg.norm(op._joint1_clamp(tangential, inner)):.3f} m/s)",
    )
    check(
        "radial velocity is left alone",
        np.allclose(op._joint1_clamp(radial, inner), radial),
    )
    check(
        "clamped tangential speed equals the ceiling",
        abs(np.linalg.norm(op._joint1_clamp(tangential, inner)) - ceil_in) < 1e-6,
    )
    # Disabled by default, so existing configs are unaffected.
    plain = ArmOperator("right", box, CFG, publish=False)
    check(
        "limiter is off unless configured",
        plain._lead_limit(inner) == CFG["max_lead"]
        and plain._joint1_ceiling(inner) == float("inf"),
    )
    for bad in (-0.1, 1.5):
        try:
            ArmOperator("right", box, dict(CFG, joint1_speed_fraction=bad), publish=False)
            check(f"joint1_speed_fraction={bad} rejected", False)
        except ValueError:
            check(f"joint1_speed_fraction={bad} rejected", True)

    # End to end: sweep y at each edge and compare achieved commanded speed.
    server = FakeServer("right")
    server.start()
    op.start()
    try:
        if not op.wait_ready(timeout=20):
            check("operator came up", False)
            return

        def sweep_y_at(x_local):
            # Park at the requested x, then sweep y and measure.
            deadline = time.time() + 2.5
            while time.time() < deadline:
                op.set_intent(1.0 if x_local > 0 else -1.0, -1.0)
                time.sleep(0.02)
            n0 = len(server.commands)
            deadline = time.time() + 1.5
            while time.time() < deadline:
                op.set_intent(0.0, 1.0)
                time.sleep(0.02)
            c = np.array(server.commands[n0:])
            hz = CFG["control_hz"]
            dy = np.abs(np.diff([box.to_box(p[:2])[1] for p in c]))
            speed = float(np.mean(np.sort(dy)[-len(dy) // 3 :])) * hz
            # Peak joint-1 rate along the swept path: w = (x*vy - y*vx)/r^2,
            # evaluated in the base frame where joint 1 actually rotates.
            v = np.diff(c[:, :2], axis=0) * hz
            x, y = c[:-1, 0], c[:-1, 1]
            w = np.abs(x * v[:, 1] - y * v[:, 0]) / (x * x + y * y)
            return speed, float(w.max())

        v_out, w_out = sweep_y_at(+1.0)
        v_in, w_in = sweep_y_at(-1.0)
        check(
            "y sweep is slower at the inner edge",
            v_in < v_out - 0.02,
            f"(inner {v_in:.3f} < outer {v_out:.3f} m/s)",
        )
        # The speed ceiling is not constant along a y sweep -- r grows with |y|,
        # so the fastest samples legitimately sit above the y=0 ceiling. The
        # invariant that actually matters is the joint-1 rate itself.
        check(
            "commanded joint-1 rate never exceeds the budget (inner edge)",
            w_in <= budget * 1.02,
            f"({w_in:.3f} vs budget {budget:.3f} rad/s)",
        )
        check(
            "commanded joint-1 rate never exceeds the budget (outer edge)",
            w_out <= budget * 1.02,
            f"({w_out:.3f} vs budget {budget:.3f} rad/s)",
        )
    finally:
        op.stop()
        op.join(timeout=5)
        server.stop.set()
        server.join(timeout=3)


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
        # Compared against the real file rather than literals: this is asserting
        # that save_box() preserves what it does not own, so hardcoding values
        # here only made it fail whenever someone legitimately retuned a knob.
        expected = {k: v for k, v in yaml.safe_load(open(src)).items() if k != "arms"}
        differing = sorted(k for k, v in expected.items() if after.get(k) != v)
        check(
            "top-level settings preserved",
            not differing,
            f"({len(expected)} keys checked" + (f", {differing} differ)" if differing else ")"),
        )
    finally:
        os.unlink(tmp)


if __name__ == "__main__":
    test_single_arm()
    test_bimanual()
    test_inward_speed_scale()
    test_lateral_speed_scale()
    test_joint1_budget()
    test_config_roundtrip()
    print()
    print("FAILED:", fails if fails else "none")
    sys.exit(1 if fails else 0)
