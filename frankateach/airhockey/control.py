"""Shared control primitives for air hockey teleop."""

import pickle
import time

import numpy as np
import zmq

from frankateach.constants import (
    GRIPPER_CLOSE,
    HOST,
    ROBOT_WORKSPACE_MAX,
    ROBOT_WORKSPACE_MIN,
    arm_ports,
)
from frankateach.messages import FrankaAction, FrankaState
from frankateach.network import create_request_socket


def assert_sole_client(arm):
    """Refuse to proceed if something else is already driving this arm.

    FrankaServer is a synchronous REQ/REP loop, so a second client is not
    rejected -- it is served, alternately with the first. The arm then sits
    between the two commanded poses, and any measurement taken is a blend of the
    two. That reads exactly like a controller fault: a run.py session holding
    plane_z while a diagnostic commanded a different height produced a very
    convincing -66 mm 'gravity sag' that was nothing of the sort.

    Used by the diagnostic scripts. The live stack enforces the same rule
    structurally, via ArmSession owning one operator at a time.

    On loopback each connection appears twice in `ss` (both endpoints), so two
    rows is one client.
    """
    import subprocess

    port = arm_ports(arm)[0]
    try:
        out = subprocess.run(
            ["ss", "-tnH", "state", "established"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        print(f"WARNING: could not check for other clients on port {port}.")
        return
    clients = sum(1 for line in out.splitlines() if f":{port}" in line) // 2
    if clients:
        raise SystemExit(
            f"\nRefusing to run: {clients} client(s) already connected to the {arm} "
            f"arm on port {port}.\n"
            "run.py (or airhockey.py, or another script) is driving this arm. Both\n"
            "would be served alternately and every measurement would be a blend of\n"
            "the two commanded poses. Stop it first, then start a bare server:\n\n"
            f"    python3 franka_server.py arm={arm} "
            f"deoxys_config_path=deoxys_{arm}_fast.yml control_freq=50 num_steps=1\n"
        )


class RateLimiter:
    """Sleep-based loop pacing that actually hits the requested rate.

    Deliberately NOT frankateach.utils.FrequencyTimer: that one busy-waits in
    end_loop(), which holds the GIL and would starve the other arm's thread. We
    run one control thread per arm in a single process, so the wait must yield.

    Two corrections make a plain time.sleep() good enough at 50 Hz:

    1. Bias compensation. time.sleep() systematically overshoots -- measured at
       ~4.7 ms for a 20 ms request, and ~9 ms with a busy sibling thread, which
       is exactly our situation. We track that overshoot and ask for
       correspondingly less.
    2. Bounded catch-up. Targets are absolute, so a late tick is made up by
       shortening the next one. We only give up and resync once we are more
       than `max_drift` periods behind, which keeps a genuine stall from
       producing a burst of back-to-back commands at the robot.
    """

    def __init__(self, hz, max_drift=2.0, bias_alpha=0.1):
        self.period = 1.0 / hz
        self.max_drift = max_drift * self.period
        self.bias_alpha = bias_alpha
        self._bias = 0.0
        self._next = time.perf_counter()

    def sleep(self):
        self._next += self.period
        now = time.perf_counter()
        remaining = self._next - now

        if remaining > 0:
            requested = max(0.0, remaining - self._bias)
            if requested > 0:
                before = time.perf_counter()
                time.sleep(requested)
                overshoot = (time.perf_counter() - before) - requested
                # Only ever compensate, never anticipate.
                self._bias += self.bias_alpha * (max(0.0, overshoot) - self._bias)
        elif -remaining > self.max_drift:
            self._next = now

        return remaining


def ramp(current, target, accel_time, dt, full_scale):
    """Move `current` toward `target` at a rate that spans full_scale in accel_time."""
    if accel_time <= 0:
        return np.array(target, dtype=np.float64)
    step = full_scale * dt / accel_time
    delta = np.asarray(target, dtype=np.float64) - current
    dist = np.linalg.norm(delta)
    if dist <= step:
        return np.array(target, dtype=np.float64)
    return current + delta / dist * step


class ArmLink:
    """REQ-socket wrapper around one arm's FrankaServer.

    Every command is an absolute EE pose with a fixed orientation and a closed
    gripper -- the mallet is bolted on and never rotates.
    """

    def __init__(self, arm, quat=None, gripper=GRIPPER_CLOSE, timeout_ms=5000):
        self.arm = arm
        self.control_port, self.state_port, self.commanded_port = arm_ports(arm)
        self.socket = create_request_socket(HOST, self.control_port)
        # Without this, a franka_server that is up but stuck in "Waiting for the
        # robot to connect" (no franka-interface, or the robot latched in Reflex)
        # leaves every caller blocked in recv() forever with no output at all.
        # Fail loudly instead -- the REQ socket is unusable after a timeout anyway.
        self.timeout_ms = int(timeout_ms)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.quat = None if quat is None else np.asarray(quat, dtype=np.float64)
        self.gripper = gripper
        self.last_state = None

    def _recv(self):
        try:
            return self.socket.recv()
        except zmq.Again as exc:
            raise RuntimeError(
                f"{self.arm} franka_server did not reply. Its port is bound but it "
                "is not serving -- usually franka-interface is not running, or the "
                "robot is latched in Reflex after hitting something. Check Desk for "
                "an error to acknowledge, then restart franka-interface."
            ) from exc

    def get_state(self) -> FrankaState:
        self.socket.send(b"get_state")
        reply = self._recv()
        if reply == b"state_error":
            raise RuntimeError(f"{self.arm} arm returned state_error")
        self.last_state = pickle.loads(reply)
        return self.last_state

    def reset(self) -> FrankaState:
        """Joint-space reset to the server's configured ready pose."""
        action = FrankaAction(
            pos=np.zeros(3),
            quat=np.zeros(4),
            gripper=self.gripper,
            reset=True,
            timestamp=time.time(),
        )
        self.socket.send(pickle.dumps(action, protocol=-1))
        # A joint reset is a real motion and FrankaServer.reset_joints allows
        # itself 7 s, so the normal per-request timeout would fire spuriously.
        self.socket.setsockopt(zmq.RCVTIMEO, max(self.timeout_ms, 30000))
        try:
            self.last_state = pickle.loads(self._recv())
        finally:
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        if self.quat is None:
            self.quat = np.asarray(self.last_state.quat, dtype=np.float64)
        return self.last_state

    def send_pose(self, pos, quat=None) -> FrankaState:
        """Command one absolute pose; returns the resulting state."""
        pos = np.clip(
            np.asarray(pos, dtype=np.float64), ROBOT_WORKSPACE_MIN, ROBOT_WORKSPACE_MAX
        )
        quat = self.quat if quat is None else np.asarray(quat, dtype=np.float64)
        action = FrankaAction(
            pos=pos.flatten().astype(np.float32),
            quat=quat.flatten().astype(np.float32),
            gripper=self.gripper,
            reset=False,
            timestamp=time.time(),
        )
        self.socket.send(pickle.dumps(action, protocol=-1))
        self.last_state = pickle.loads(self._recv())
        return self.last_state, action

    def glide_to(self, target, speed=0.08, hz=50.0, tol=0.003, timeout=15.0):
        """Move to a pose in a straight line at a bounded speed.

        Used for homing and for the calibration perimeter trace -- anywhere the
        arm must cover distance without being yanked there by a step command.
        """
        target = np.asarray(target, dtype=np.float64)
        limiter = RateLimiter(hz)
        deadline = time.perf_counter() + timeout
        state = self.get_state()
        pos = np.asarray(state.pos, dtype=np.float64)
        step = speed / hz

        while time.perf_counter() < deadline:
            delta = target - pos
            dist = np.linalg.norm(delta)
            if dist < tol:
                break
            pos = pos + delta / dist * min(step, dist)
            self.send_pose(pos)
            limiter.sleep()

        # Settle on the exact target.
        for _ in range(int(0.2 * hz)):
            self.send_pose(target)
            limiter.sleep()
        return self.last_state

    def close(self):
        self.socket.close()
