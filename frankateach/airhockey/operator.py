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

from frankateach.airhockey.control import ArmLink, RateLimiter, StatePublisher, ramp
from frankateach.constants import HOST, JOINT1_VELOCITY_LIMIT
from frankateach.network import ZMQKeypointPublisher


@dataclass
class ArmIntent:
    """What the keyboard wants, in box axes. Written by the input thread."""

    vx: float = 0.0
    vy: float = 0.0
    stamp: float = field(default_factory=time.perf_counter)
    home: bool = False
    frozen: bool = False
    sequence: int = 0
    source_stamp_ns: int = 0


@dataclass
class ArmStatus:
    """What the control thread is actually doing. Read by the renderer."""

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    box_pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    speed: float = 0.0
    speed_limit: float = 0.0
    rate: float = 0.0
    connected: bool = False
    stale: bool = False
    homing: bool = False
    error: str = ""


class ArmOperator(threading.Thread):
    def __init__(
        self,
        arm,
        box,
        cfg,
        publish=True,
        speed=None,
        reset=True,
        telemetry_callback=None,
    ):
        super().__init__(daemon=True, name=f"arm-{arm}")
        self.arm = arm
        self.box = box
        # reset=False holds the current pose instead of joint-resetting and gliding
        # to the box centre. Used for a provisional box, which is anchored on the
        # pose the arm is already in -- resetting would move an arm whose play area
        # nobody has verified.
        self.reset = reset
        self.hz = float(cfg["control_hz"])
        # speed is overridable so the launcher can cap an uncalibrated arm.
        self.speed = float(cfg["speed"] if speed is None else speed)
        # Inward (-x in box axes, toward the robot base) is capped below the
        # outward speed. Joint 1 carries all of the box-y travel, and its
        # tangential ceiling is omega_limit * radius -- so the inner edge of a
        # box centred 0.31 m out has ~0.48 m/s of headroom against ~0.86 m/s at
        # the outer edge. Slowing the inward half keeps the same fraction of
        # that ceiling across the box instead of only at the far end.
        # 1.0 restores the original symmetric behaviour.
        self.inward_scale = float(cfg.get("inward_speed_scale", 1.0))
        if not 0.0 < self.inward_scale <= 1.0:
            raise ValueError(
                "inward_speed_scale must be in (0, 1] -- it only ever reduces "
                f"speed, and 0 would trap the arm at the far edge. Got {self.inward_scale}"
            )
        # Joint-1 budget, in rad/s. See _joint1_ceiling / _lead_limit.
        self.j1_fraction = float(cfg.get("joint1_speed_fraction", 0.0))
        if not 0.0 <= self.j1_fraction <= 1.0:
            raise ValueError(
                f"joint1_speed_fraction must be in [0, 1], got {self.j1_fraction}"
            )
        self.j1_budget = self.j1_fraction * JOINT1_VELOCITY_LIMIT
        # Peak EE speed the OSC controller drives per metre of leash.
        self.lead_speed_gain = float(cfg.get("lead_speed_gain", 9.35))
        if self.lead_speed_gain <= 0.0:
            raise ValueError(
                f"lead_speed_gain must be positive, got {self.lead_speed_gain}"
            )
        self.accel_time = float(cfg["accel_time"])
        self.max_lead = float(cfg["max_lead"])
        self.watchdog = float(cfg["watchdog"])
        self.home_speed = float(cfg.get("home_speed", 0.15))
        # Base-frame [x, y] gains, metres of z per m/s. See configs/airhockey.yaml.
        self.z_ff = np.array(
            [float(cfg.get("z_feedforward_x", 0.0)), float(cfg.get("z_feedforward_y", 0.0))]
        )
        self.z_ff_limit = abs(float(cfg.get("z_feedforward_limit", 0.01)))
        # In-plane counterpart of z_ff. See _xy_lead. Base-frame [x, y] metres.
        # Zero (the default) disables it and leaves the command bit-identical.
        self.xy_ff = np.array(
            [
                float(cfg.get("xy_feedforward_x", 0.0)),
                float(cfg.get("xy_feedforward_y", 0.0)),
            ]
        )
        self.xy_ff_limit = abs(float(cfg.get("xy_feedforward_limit", 0.02)))
        # Speed at which the offset reaches full magnitude. Without it, a
        # direction-dependent offset would step by 2*xy_ff the instant velocity
        # crosses zero -- precisely at a reversal, which is where
        # joint_velocity_violation already lives.
        self.xy_ff_knee = abs(float(cfg.get("xy_feedforward_knee", 0.05)))
        if self.xy_ff_knee <= 0.0:
            raise ValueError("xy_feedforward_knee must be positive")
        self.publish = publish
        # Recording observes the control loop here. The callback must be
        # non-blocking; a false return means its bounded queue overflowed.
        self.telemetry_callback = telemetry_callback
        self._telemetry_drops = 0

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
        self._publisher = None

    # -- called from the input thread --------------------------------------
    def set_intent(
        self,
        vx,
        vy,
        frozen=False,
        home=False,
        sequence=0,
        source_stamp_ns=0,
    ):
        with self._lock:
            self.intent = ArmIntent(
                vx=vx,
                vy=vy,
                stamp=time.perf_counter(),
                home=home,
                frozen=frozen,
                sequence=int(sequence),
                source_stamp_ns=int(source_stamp_ns),
            )

    def set_speed_limit(self, speed):
        """Change the play-speed cap without restarting the control loop."""
        speed = float(speed)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("speed limit must be a positive finite number")
        with self._lock:
            self.speed = speed

    def get_status(self):
        with self._lock:
            return self.status

    def wait_ready(self, timeout=60.0):
        return self._ready.wait(timeout)

    def stop(self):
        self._stop_event.set()

    def _joint1_ceiling(self, xy):
        """Base-frame EE speed at which joint 1 saturates, at position `xy`.

        Joint 1 rotates about the base z axis, so it alone carries motion
        tangential to the base radius: w = (x*vy - y*vx) / r^2. The worst case
        at a point is pure tangential motion, needing w = v/r, so the ceiling is
        simply `budget * r` -- and it collapses as the arm comes in toward the
        base. For a box centred 0.31 m out that is ~0.86 m/s at the outer edge
        against ~0.48 m/s at the inner one, a 1.8x spread across 17 cm of
        travel, which is why a single `speed` number cannot be right everywhere.

        Returns inf when the limiter is disabled or the arm is on the base axis,
        where the notion degenerates; ROBOT_WORKSPACE_MIN keeps us far from it.
        """
        if self.j1_budget <= 0.0:
            return float("inf")
        r = float(np.hypot(xy[0], xy[1]))
        if r < 1e-6:
            return float("inf")
        return self.j1_budget * r

    def _joint1_clamp(self, vel, xy):
        """Scale a commanded velocity so joint 1 stays inside its budget.

        Uses the exact rate for the commanded direction rather than the
        worst-case tangential bound, so purely radial motion -- which barely
        turns joint 1 at all -- is not penalised.
        """
        if self.j1_budget <= 0.0:
            return vel
        r2 = float(xy[0] ** 2 + xy[1] ** 2)
        if r2 < 1e-12:
            return vel
        w = abs(xy[0] * vel[1] - xy[1] * vel[0]) / r2
        return vel * (self.j1_budget / w) if w > self.j1_budget else vel

    def _lead_limit(self, xy):
        """Leash length allowed at `xy`, tightened where joint 1 has less room.

        The leash, not `speed`, is what sets *peak* speed. The OSC controller
        drives toward roughly sqrt(Kp)/2 m/s for every metre of lead -- 9.35 1/s
        at Kp=350 -- so an 80 mm leash permits ~0.75 m/s no matter how low
        `speed` is. Recorded episodes bear this out: commanding 0.35 m/s, the
        measured EE still peaked at 0.66 m/s, which is a harmless 77% of joint 1
        at the box's outer edge but would be 139% of it at the inner edge.

        So capping the commanded velocity alone does not prevent the reflex; the
        leash has to shrink too. That is the whole reason this method exists.
        """
        if self.j1_budget <= 0.0:
            return self.max_lead
        return min(self.max_lead, self._joint1_ceiling(xy) / self.lead_speed_gain)

    def _xy_lead(self, vel):
        """In-plane feedforward for the friction that makes x lag more than y.

        The OSC controller settles to

            e = (Kd/Kp)*v  +  (1/Kp) * Lambda^-1 * F_uncompensated

        scripts/wobble_test.py already reports both halves per axis: its
        "lag along" column against "predicted", where predicted is the first
        term, -2*v/sqrt(Kp). Whatever "lag along" shows on top of "predicted"
        is F_uncompensated, and that excess is what this cancels.

        It is far larger on x than on y. x is driven by the shoulder and elbow,
        which rotate about horizontal axes while carrying the arm's weight; y is
        the base yaw joint, whose vertical axis neither fights gravity nor loads
        its bearings the same way. The same asymmetry is already visible in
        z_feedforward_x (0.0105) against z_feedforward_y (-0.0015).

        Coulomb-shaped and per axis: the offset follows the *direction* of
        travel rather than its magnitude, ramped in over `xy_ff_knee` so a
        reversal does not step the command. Add a speed-proportional term only
        if wobble_test shows the excess growing with speed -- flat means dry
        friction, which this shape already matches.
        """
        if not self.xy_ff.any():
            return np.zeros(2)
        offset = self.xy_ff * np.clip(
            np.asarray(vel, dtype=np.float64) / self.xy_ff_knee, -1.0, 1.0
        )
        norm = float(np.linalg.norm(offset))
        if norm > self.xy_ff_limit:
            offset = offset * (self.xy_ff_limit / norm)
        return offset

    def _z_lead(self, vel):
        """Vertical feedforward for the commanded xy velocity `vel` (base frame).

        The controller does not compensate joint friction, so the EE rides off the
        play plane while it translates -- by an amount proportional to velocity and
        with a sign that flips with direction. We cannot fix that from here, but we
        can command the opposite offset so the arm ends up where the box says.

        `vel` is the commanded velocity, not the measured one, so this leads rather
        than chases; that is the point. It is zero while homing and while frozen,
        because `vel` is zeroed there.

        Deliberately NOT applied in JogOperator: calibration has to measure the
        real surface, and a feedforward would bias the taught corner heights by
        exactly the error it is correcting for.
        """
        if not self.z_ff.any():
            return 0.0
        return float(np.clip(self.z_ff @ vel, -self.z_ff_limit, self.z_ff_limit))

    # -- control thread -----------------------------------------------------
    def run(self):
        try:
            self.link = ArmLink(self.arm, quat=self.box.quat)
            if self.publish:
                self._state_pub = ZMQKeypointPublisher(HOST, self.link.state_port)
                self._cmd_pub = ZMQKeypointPublisher(HOST, self.link.commanded_port)
                self._publisher = StatePublisher(
                    (
                        (self._state_pub, "robot_state"),
                        (self._cmd_pub, "commanded_robot_state"),
                    )
                )
                self._publisher.start()

            if self.reset:
                print(f"[{self.arm}] resetting to ready pose...")
                self.link.reset()
                print(f"[{self.arm}] moving to box centre {self.box.home}...")
                self.link.glide_to(self.box.home, speed=0.08, hz=self.hz)
            else:
                state = self.link.get_state()
                print(
                    f"[{self.arm}] holding current pose "
                    f"z={state.pos[2]:.4f} (no reset, no glide)"
                )

            target = np.asarray(self.link.last_state.pos, dtype=np.float64).copy()
            target[:2] = self.box.clip(target[:2])
            target[2] = self.box.z_at(target[:2])
            vel = np.zeros(2)

            # Publish connected=True *before* signalling ready. connected is
            # otherwise first set at the end of the first loop iteration, and
            # callers that wait_ready() then immediately read the status win that
            # race against a real arm's ~20 ms round trip (FakeServer answers in
            # microseconds, which is why this only shows up on hardware).
            with self._lock:
                self.status = ArmStatus(
                    pos=np.asarray(self.link.last_state.pos, dtype=np.float64),
                    box_pos=self.box.clip(
                        np.asarray(self.link.last_state.pos, dtype=np.float64)[:2]
                    ),
                    connected=True,
                    stale=True,
                    speed_limit=self.speed,
                )
            self._ready.set()
            print(f"[{self.arm}] live at {self.hz:.0f} Hz")

            limiter = RateLimiter(self.hz)
            dt = 1.0 / self.hz
            last_tick = time.perf_counter()
            # Smoothed, because a single tick's period is far too jittery to
            # read off a HUD -- the instantaneous value swings by tens of Hz.
            mean_period = dt

            tick_sequence = 0
            while not self._stop_event.is_set():
                with self._lock:
                    intent, speed_limit = self.intent, self.speed

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
                    # After the diagonal normalisation, so the inward cap is a
                    # real speed limit rather than something a diagonal dilutes.
                    if v[0] < 0.0:
                        v[0] *= self.inward_scale
                    desired = self.box.rotate_intent(v[0], v[1]) * speed_limit

                if intent.home:
                    vel = np.zeros(2)
                    delta = self.box.center - target[:2]
                    dist = np.linalg.norm(delta)
                    step = self.home_speed * dt
                    target[:2] = (
                        self.box.center if dist <= step else target[:2] + delta / dist * step
                    )
                else:
                    vel = ramp(vel, desired, self.accel_time, dt, speed_limit)
                    # A newly lowered limit is a hard cap, not merely a new ramp
                    # target. This makes the WebUI control take effect on this tick.
                    vel_mag = np.linalg.norm(vel)
                    if vel_mag > speed_limit:
                        vel *= speed_limit / vel_mag
                    # Cruise cap: keeps joint 1 inside its budget in steady
                    # state. The leash below handles the transient peaks.
                    vel = self._joint1_clamp(vel, target[:2])
                    target[:2] = target[:2] + vel * dt

                target[:2] = self.box.clip(target[:2])

                # Leash: never let the command outrun where the arm actually is.
                # Measured in the plane only. z is not ours to give away -- it is
                # dictated by the play surface -- and letting a z sag count against
                # the budget would tighten the xy leash for no reason.
                actual = np.asarray(self.link.last_state.pos, dtype=np.float64)
                delta_xy = target[:2] - actual[:2]
                dist_xy = np.linalg.norm(delta_xy)
                # Anchored on where the arm *is*, not on the target: the leash
                # bounds the peak speed the controller will drive to, so it has
                # to reflect the joint-1 headroom at the arm's own position.
                lead_limit = self._lead_limit(actual[:2])
                if dist_xy > lead_limit:
                    target[:2] = actual[:2] + delta_xy / dist_xy * lead_limit
                # The pose actually sent. Deliberately NOT written back into
                # `target`: target is the integrator, so folding the feedforward
                # into it would accumulate the offset every tick instead of
                # applying it once.
                command_xy = target[:2]
                offset = self._xy_lead(vel)
                if offset.any():
                    command_xy = self.box.clip(target[:2] + offset)
                    # The feedforward does not get to buy extra lead. The
                    # joint-1 budget reasons about total command-to-arm
                    # distance, so that bound has to hold for what is sent.
                    delta_ff = command_xy - actual[:2]
                    dist_ff = np.linalg.norm(delta_ff)
                    if dist_ff > lead_limit:
                        command_xy = actual[:2] + delta_ff / dist_ff * lead_limit
                command = np.array(
                    [
                        command_xy[0],
                        command_xy[1],
                        self.box.z_at(command_xy) + self._z_lead(vel),
                    ]
                )

                command_mono_ns = time.perf_counter_ns()
                state, action = self.link.send_pose(command)
                state_mono_ns = time.perf_counter_ns()

                if self.publish:
                    state.start_teleop = not (stale or intent.frozen)
                    self._publisher.publish((state, action))

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
                        speed_limit=speed_limit,
                        rate=1.0 / mean_period if mean_period > 0 else 0.0,
                        connected=True,
                        stale=stale or intent.frozen,
                        homing=intent.home,
                    )
                tick_sequence += 1
                if self.telemetry_callback is not None:
                    event = {
                        "t": "telemetry",
                        "arm": self.arm,
                        "tick_sequence": tick_sequence,
                        "intent_sequence": int(intent.sequence),
                        "intent_source_mono_ns": int(intent.source_stamp_ns),
                        "command_mono_ns": command_mono_ns,
                        "state_mono_ns": state_mono_ns,
                        "commanded_pos": np.asarray(action.pos, dtype=np.float64).tolist(),
                        "commanded_box_xy": self.box.to_box(action.pos[:2]).tolist(),
                        "commanded_box_velocity": self.box.to_box(
                            self.box.center + vel
                        ).tolist(),
                        "measured_pos": pos.tolist(),
                        "measured_quat": np.asarray(state.quat, dtype=np.float64).tolist(),
                        "speed": float(np.linalg.norm(vel)),
                        "rate_hz": 1.0 / mean_period if mean_period > 0 else 0.0,
                        "connected": True,
                        "stale": bool(stale),
                        "frozen": bool(intent.frozen),
                        "homing": bool(intent.home),
                        "error": "",
                        "telemetry_drops": self._telemetry_drops,
                    }
                    try:
                        accepted = self.telemetry_callback(event)
                    except Exception:
                        accepted = False
                    if accepted is False:
                        self._telemetry_drops += 1
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
        # Stop the worker before the sockets it publishes on.
        try:
            if self._publisher is not None:
                self._publisher.close()
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

    def __init__(self, arm, cfg, reset=True):
        super().__init__(daemon=True, name=f"jog-{arm}")
        self.arm = arm
        self.reset = reset
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
            if self.reset:
                print(f"[{self.arm}] resetting to ready pose for calibration...")
                self.link.reset()
            else:
                # Jog from wherever the arm already is -- e.g. a height set by
                # hand-guiding in Desk. The reset pose can sit low enough to
                # graze the table, and Q/E only moves at jog_z_speed.
                state = self.link.get_state()
                # reset() is what normally latches the held orientation; without
                # it ArmLink.quat stays None and send_pose would fail.
                self.link.quat = np.asarray(state.quat, dtype=np.float64)
                print(
                    f"[{self.arm}] no reset; jogging from current pose "
                    f"z={state.pos[2]:.4f}"
                )
            target = np.asarray(self.link.last_state.pos, dtype=np.float64).copy()
            vel = np.zeros(3)
            # See the matching comment in ArmOperator.run: connected must be
            # visible before _ready fires, or serve_calibrate reports a bare
            # "ERROR:" (connected False, error still "") on a healthy arm.
            with self._lock:
                self.status = JogStatus(
                    pos=np.asarray(self.link.last_state.pos, dtype=np.float64),
                    quat=np.asarray(self.link.last_state.quat, dtype=np.float64),
                    connected=True,
                    stale=True,
                )
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

                # Leash xy and z independently. A single 3-vector clip pulls the
                # commanded height toward wherever the arm has sagged to, so lag in
                # the plane made WASD wander in z -- which then got baked into the
                # taught corner heights.
                actual = np.asarray(self.link.last_state.pos, dtype=np.float64)
                delta_xy = target[:2] - actual[:2]
                dist_xy = np.linalg.norm(delta_xy)
                if dist_xy > self.max_lead:
                    target[:2] = actual[:2] + delta_xy / dist_xy * self.max_lead
                dz = target[2] - actual[2]
                if abs(dz) > self.max_lead:
                    target[2] = actual[2] + np.sign(dz) * self.max_lead

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
