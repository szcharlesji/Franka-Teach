"""Per-arm control loops for air hockey teleop.

ArmOperator is the 50 Hz play loop, one per arm, each on its own thread with its
own REQ socket so a slow osc_move on one arm cannot stall the other. JogOperator
is the 3-DoF version used to teach the play box during calibration.

The UI only ever writes an intent; everything safety-relevant (clip, leash,
watchdog) lives in here, so a hung, crashed, or *disconnected* UI freezes the
arm rather than running it. That matters more now that the UI is a web page on
the far side of a network link than it did when it was a local pygame window.
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from frankateach.airhockey.control import ArmLink, RateLimiter, ramp
from frankateach.constants import HOST
from frankateach.network import ZMQKeypointPublisher


@dataclass
class ArmIntent:
    """What the keyboard wants, in box axes. Written by the input thread."""

    vx: float = 0.0
    vy: float = 0.0
    stamp: float = field(default_factory=time.perf_counter)
    home: bool = False
    frozen: bool = False


@dataclass
class ArmStatus:
    """What the control thread is actually doing. Read by the renderer."""

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    box_pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    speed: float = 0.0
    rate: float = 0.0
    connected: bool = False
    stale: bool = False
    homing: bool = False
    error: str = ""


class ArmOperator(threading.Thread):
    def __init__(self, arm, box, cfg, publish=True):
        super().__init__(daemon=True, name=f"arm-{arm}")
        self.arm = arm
        self.box = box
        self.hz = float(cfg["control_hz"])
        self.speed = float(cfg["speed"])
        self.accel_time = float(cfg["accel_time"])
        self.max_lead = float(cfg["max_lead"])
        self.watchdog = float(cfg["watchdog"])
        self.home_speed = float(cfg.get("home_speed", 0.15))
        self.publish = publish

        self.intent = ArmIntent(frozen=True)
        self.status = ArmStatus()
        self._lock = threading.Lock()
        # NOT self._stop -- threading.Thread has an internal _stop() method that
        # join() calls during teardown; shadowing it breaks every clean shutdown.
        self._stop_event = threading.Event()
        self._ready = threading.Event()
        self.link = None
        self._state_pub = None
        self._cmd_pub = None

    # -- called from the input thread --------------------------------------
    def set_intent(self, vx, vy, frozen=False, home=False):
        with self._lock:
            self.intent = ArmIntent(
                vx=vx, vy=vy, stamp=time.perf_counter(), home=home, frozen=frozen
            )

    def get_status(self):
        with self._lock:
            return self.status

    def wait_ready(self, timeout=60.0):
        return self._ready.wait(timeout)

    def stop(self):
        self._stop_event.set()

    # -- control thread -----------------------------------------------------
    def run(self):
        try:
            self.link = ArmLink(self.arm, quat=self.box.quat)
            if self.publish:
                self._state_pub = ZMQKeypointPublisher(HOST, self.link.state_port)
                self._cmd_pub = ZMQKeypointPublisher(HOST, self.link.commanded_port)

            print(f"[{self.arm}] resetting to ready pose...")
            self.link.reset()
            print(f"[{self.arm}] moving to box centre {self.box.home}...")
            self.link.glide_to(self.box.home, speed=0.08, hz=self.hz)

            target = np.asarray(self.link.last_state.pos, dtype=np.float64).copy()
            target[:2] = self.box.clip(target[:2])
            target[2] = self.box.plane_z
            vel = np.zeros(2)

            self._ready.set()
            print(f"[{self.arm}] live at {self.hz:.0f} Hz")

            limiter = RateLimiter(self.hz)
            dt = 1.0 / self.hz
            last_tick = time.perf_counter()
            # Smoothed, because a single tick's period is far too jittery to
            # read off a HUD -- the instantaneous value swings by tens of Hz.
            mean_period = dt

            while not self._stop_event.is_set():
                with self._lock:
                    intent = self.intent

                age = time.perf_counter() - intent.stamp
                stale = age > self.watchdog
                # Watchdog: a hung input thread must not leave the last
                # velocity applied. Freeze in place, keep holding the pose.
                if stale or intent.frozen:
                    desired = np.zeros(2)
                else:
                    # Normalise diagonals: pressing two keys must not make the
                    # mallet 41% faster than pressing one.
                    v = np.array([intent.vx, intent.vy], dtype=np.float64)
                    mag = np.linalg.norm(v)
                    if mag > 1.0:
                        v /= mag
                    desired = self.box.rotate_intent(v[0], v[1]) * self.speed

                if intent.home:
                    vel = np.zeros(2)
                    delta = self.box.center - target[:2]
                    dist = np.linalg.norm(delta)
                    step = self.home_speed * dt
                    target[:2] = (
                        self.box.center if dist <= step else target[:2] + delta / dist * step
                    )
                else:
                    vel = ramp(vel, desired, self.accel_time, dt, self.speed)
                    target[:2] = target[:2] + vel * dt

                target[:2] = self.box.clip(target[:2])
                target[2] = self.box.plane_z

                # Leash: never let the command outrun where the arm actually is.
                actual = np.asarray(self.link.last_state.pos, dtype=np.float64)
                delta = target - actual
                dist = np.linalg.norm(delta)
                if dist > self.max_lead:
                    target = actual + delta / dist * self.max_lead
                    target[2] = self.box.plane_z

                state, action = self.link.send_pose(target)

                if self.publish:
                    state.start_teleop = not (stale or intent.frozen)
                    self._state_pub.pub_keypoints(state, "robot_state")
                    self._cmd_pub.pub_keypoints(action, "commanded_robot_state")

                now = time.perf_counter()
                measured = now - last_tick
                last_tick = now
                if measured > 0:
                    mean_period += 0.05 * (measured - mean_period)
                with self._lock:
                    pos = np.asarray(state.pos, dtype=np.float64)
                    self.status = ArmStatus(
                        pos=pos,
                        box_pos=self.box.to_box(pos[:2]),
                        speed=float(np.linalg.norm(vel)),
                        rate=1.0 / mean_period if mean_period > 0 else 0.0,
                        connected=True,
                        stale=stale or intent.frozen,
                        homing=intent.home,
                    )
                limiter.sleep()
        except Exception as exc:  # keep one arm's failure off the other's thread
            with self._lock:
                self.status = ArmStatus(connected=False, error=f"{type(exc).__name__}: {exc}")
            print(f"[{self.arm}] control loop died: {type(exc).__name__}: {exc}")
            self._ready.set()
        finally:
            self._park()

    def _park(self):
        """Hold the last pose briefly, then drop the sockets."""
        try:
            if self.link is not None and self.link.last_state is not None:
                pos = np.asarray(self.link.last_state.pos, dtype=np.float64)
                for _ in range(int(0.2 * self.hz)):
                    self.link.send_pose(pos)
                    time.sleep(1.0 / self.hz)
        except Exception:
            pass
        for sock in (self._state_pub, self._cmd_pub):
            try:
                if sock is not None:
                    sock.stop()
            except Exception:
                pass
        try:
            if self.link is not None:
                self.link.close()
        except Exception:
            pass
        print(f"[{self.arm}] parked.")


@dataclass
class JogStatus:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    rate: float = 0.0
    connected: bool = False
    stale: bool = False
    error: str = ""


class JogOperator(threading.Thread):
    """Free 3-DoF jogging for calibration.

    Same intent seam and watchdog as ArmOperator, but no play box (there isn't
    one yet -- that's what you're teaching) and a z axis so you can find the
    play height. Only the global ROBOT_WORKSPACE clip in ArmLink.send_pose
    bounds it, so jog slowly.
    """

    def __init__(self, arm, cfg):
        super().__init__(daemon=True, name=f"jog-{arm}")
        self.arm = arm
        self.hz = float(cfg["control_hz"])
        self.speed = float(cfg["jog_speed"])
        self.z_speed = float(cfg["jog_z_speed"])
        self.accel_time = float(cfg["accel_time"])
        self.max_lead = float(cfg["max_lead"])
        self.watchdog = float(cfg["watchdog"])

        self.intent = ArmIntent(frozen=True)
        self.vz = 0.0
        self.status = JogStatus()
        self._lock = threading.Lock()
        # NOT self._stop -- shadows threading.Thread._stop(), breaking join().
        self._stop_event = threading.Event()
        self._ready = threading.Event()
        self.link = None

    def set_intent(self, vx, vy, vz=0.0, frozen=False):
        with self._lock:
            self.intent = ArmIntent(
                vx=vx, vy=vy, stamp=time.perf_counter(), frozen=frozen
            )
            self.vz = vz

    def get_status(self):
        with self._lock:
            return self.status

    def wait_ready(self, timeout=90.0):
        return self._ready.wait(timeout)

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            self.link = ArmLink(self.arm)
            print(f"[{self.arm}] resetting to ready pose for calibration...")
            self.link.reset()
            target = np.asarray(self.link.last_state.pos, dtype=np.float64).copy()
            vel = np.zeros(3)
            self._ready.set()
            print(f"[{self.arm}] jog ready at {self.hz:.0f} Hz")

            limiter = RateLimiter(self.hz)
            dt = 1.0 / self.hz
            last_tick = time.perf_counter()
            mean_period = dt

            while not self._stop_event.is_set():
                with self._lock:
                    intent, vz = self.intent, self.vz

                stale = (time.perf_counter() - intent.stamp) > self.watchdog
                if stale or intent.frozen:
                    desired = np.zeros(3)
                else:
                    v = np.array([intent.vx, intent.vy], dtype=np.float64)
                    mag = np.linalg.norm(v)
                    if mag > 1.0:
                        v /= mag
                    desired = np.array(
                        [v[0] * self.speed, v[1] * self.speed, vz * self.z_speed]
                    )

                vel = ramp(vel, desired, self.accel_time, dt, self.speed)
                target = target + vel * dt

                actual = np.asarray(self.link.last_state.pos, dtype=np.float64)
                delta = target - actual
                dist = np.linalg.norm(delta)
                if dist > self.max_lead:
                    target = actual + delta / dist * self.max_lead

                state, _ = self.link.send_pose(target)

                now = time.perf_counter()
                measured = now - last_tick
                last_tick = now
                if measured > 0:
                    mean_period += 0.05 * (measured - mean_period)
                with self._lock:
                    self.status = JogStatus(
                        pos=np.asarray(state.pos, dtype=np.float64),
                        quat=np.asarray(state.quat, dtype=np.float64),
                        rate=1.0 / mean_period if mean_period > 0 else 0.0,
                        connected=True,
                        stale=stale or intent.frozen,
                    )
                limiter.sleep()
        except Exception as exc:
            with self._lock:
                self.status = JogStatus(
                    connected=False, error=f"{type(exc).__name__}: {exc}"
                )
            print(f"[{self.arm}] jog loop died: {type(exc).__name__}: {exc}")
            self._ready.set()
        finally:
            try:
                if self.link is not None and self.link.last_state is not None:
                    pos = np.asarray(self.link.last_state.pos, dtype=np.float64)
                    for _ in range(int(0.2 * self.hz)):
                        self.link.send_pose(pos)
                        time.sleep(1.0 / self.hz)
            except Exception:
                pass
            print(f"[{self.arm}] jog stopped.")
