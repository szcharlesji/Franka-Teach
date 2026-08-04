"""Why does the EE deviate vertically while translating horizontally?

The symptom is a z deviation during travel that *mirrors* when the direction
reverses -- forward the path sags in the middle, backward it lifts. This script
separates the three mechanisms that can produce that, none of which is visible
in scripts/rate_test.py's single Euclidean error.

The controller is deoxys' OSC_POSE (franka-interface/src/controllers/
osc_impedance.cpp). At constant velocity it settles to

    Lambda_ctrl * (Kp*e - Kd*v) = F_uncompensated
      =>  e = (Kd/Kp)*v  +  (1/Kp) * Lambda^-1 * F_uncompensated

The first term is lag *along* the direction of travel and has no z component,
because we command v_z = 0. So every millimetre of z deviation is the second
term, and it is classified by how it behaves under two transforms:

  reverse direction   even  -> Coriolis (C(q,qd)qd is quadratic in qd; computed
                               at osc_impedance.cpp:123 and never added to tau_d)
                               or a gravity/payload model error
                      odd   -> joint friction (-f*sign(qd)); libfranka's torque
                               interface compensates gravity but NOT friction,
                               and deoxys models none

  double the speed    flat  -> Coulomb (dry) friction
                      x2    -> viscous friction
                      x4    -> Coriolis

So the script drives a trapezoidal out-and-back stroke on both horizontal axes
at two speeds, and reports the z deviation during the constant-velocity cruise
decomposed into its symmetric (even) and antisymmetric (odd) parts. The
antisymmetric part is the mirrored sag you are seeing; its speed scaling names
the cause.

Separately, a steady offset and a *ringing* want different fixes, so the cruise
trace is also FFT'd. Expected peaks:

  ~sqrt(Kp)/2pi (~3.0 Hz at Kp=350)  underdamped task-space loop. Kd = 2*sqrt(Kp)
                                     is critical damping ONLY when Lambda_ctrl
                                     equals the true task-space inertia, and
                                     osc_impedance.cpp:120 deliberately breaks
                                     that with residual_mass_vec.
  10-30 Hz                           elbow / nullspace mode. The nullspace term
                                     (line 219) is a pure spring with no damping
                                     at all, and the projector it acts through is
                                     built from the same corrupted M, so it does
                                     not actually cancel at the EE.
  low, irregular, worse when slow    stick-slip.

Sampling is capped at the 50 Hz command round trip (franka_server has no state
publisher), so Nyquist is 25 Hz -- anything faster aliases. Broadband ripple with
no clean peak is itself the signature of an above-Nyquist mode.

Run it raised well clear of the table; see the module docstring of hold_test.py
for the same caution. Nothing here needs the mallet off.
"""

import argparse
import csv
import time

import numpy as np

from frankateach.airhockey.control import ArmLink, RateLimiter, assert_sole_client
from frankateach.constants import ROBOT_WORKSPACE_MAX, ROBOT_WORKSPACE_MIN

AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0])}


class Trapezoid:
    """Distance/velocity along a stroke: ramp up, cruise, ramp down.

    The cruise phase is the whole point -- a pure sine (what rate_test.py uses)
    never holds a constant velocity, so it cannot separate a velocity-dependent
    offset from an acceleration-dependent one.
    """

    def __init__(self, dist, speed, ramp):
        ramp = max(ramp, 1e-6)
        if dist < speed * ramp:  # too short to reach cruise; go triangular
            speed = dist / ramp
        self.dist = dist
        self.v = speed
        self.ramp = ramp
        self.d_ramp = 0.5 * speed * ramp
        self.d_cruise = dist - 2 * self.d_ramp
        self.t1 = ramp
        self.t2 = self.t1 + (self.d_cruise / speed if speed > 0 else 0.0)
        self.t3 = self.t2 + ramp

    def at(self, t):
        v, ramp = self.v, self.ramp
        if t < self.t1:
            return 0.5 * v / ramp * t * t, v / ramp * t
        if t < self.t2:
            return self.d_ramp + v * (t - self.t1), v
        if t < self.t3:
            dt = t - self.t2
            return (
                self.d_ramp + self.d_cruise + v * dt - 0.5 * v / ramp * dt * dt,
                v - v / ramp * dt,
            )
        return self.dist, 0.0


def _stroke(link, start, unit, dist, speed, ramp, hz, dwell, rows, tag):
    """One trapezoidal traverse, sampled at `hz`. Appends to `rows`."""
    prof = Trapezoid(dist, speed, ramp)
    limiter = RateLimiter(hz)
    t0 = time.perf_counter()
    while True:
        t = time.perf_counter() - t0
        if t > prof.t3 + dwell:
            break
        s, v = prof.at(t)
        target = start + unit * s
        state, _ = link.send_pose(target)
        meas = np.asarray(state.pos, dtype=np.float64)
        rows.append(
            {
                "axis": tag[0],
                "speed": tag[1],
                "dir": tag[2],
                "t": t,
                "v": v,
                "cruise": prof.v,
                "t1": prof.t1,  # cruise starts here; used to discard settling
                "cmd": target.copy(),
                "meas": meas,
                # Deviation perpendicular to travel. Commanded z is constant, so
                # this is pure tracking error.
                "dz": meas[2] - target[2],
                # Lag along travel. Should be ~ -(Kd/Kp)*v; a sanity check that
                # our model of the controller matches the hardware.
                "along": float((meas - target) @ unit),
            }
        )
        limiter.sleep()
    return prof


def _spectrum(dev, dt):
    """Dominant frequency and RMS of a detrended cruise segment."""
    n = len(dev)
    if n < 16 or dt <= 0:
        return None
    dev = dev - np.polyval(np.polyfit(np.arange(n), dev, 1), np.arange(n))
    rms = float(np.sqrt(np.mean(dev**2)))
    crossings = int(np.sum(np.diff(np.sign(dev - dev.mean())) != 0))
    win = np.hanning(n)
    mag = np.abs(np.fft.rfft(dev * win)) * 2.0 / win.sum()
    freq = np.fft.rfftfreq(n, dt)
    band = freq >= 0.5  # ignore DC and the residual trend
    if not band.any():
        return None
    k = int(np.argmax(mag[band]))
    return {
        "rms": rms,
        "peak_hz": float(freq[band][k]),
        "peak_mm": float(mag[band][k] * 1000),
        "nyquist": float(0.5 / dt),
        # A short cruise cannot resolve 3 Hz from 5 Hz. Report the bin width so a
        # peak is never read more precisely than it was measured.
        "resolution_hz": float(1.0 / (n * dt)),
        "crossings": crossings,
        "n": n,
    }


def _group(rows, axis, speed, direction):
    return [r for r in rows if r["axis"] == axis and r["speed"] == speed and r["dir"] == direction]


def run(arm, hz, dz, amplitude, speeds, ramp, axes, cycles, dwell, kp, csv_path,
        floor_mm=0.05, z_abs=None, settle_s=None):
    assert_sole_client(arm)
    link = ArmLink(arm)
    state = link.get_state()
    # No reset: work from wherever the arm is, exactly like hold_test.py.
    link.quat = np.asarray(state.quat, dtype=np.float64)
    origin = np.asarray(state.pos, dtype=np.float64).copy()
    print(f"Starting from {np.round(origin, 4)}")

    rows = []
    try:
        if z_abs is not None and dz:
            raise SystemExit("Pass --z or --dz, not both.")
        if z_abs is not None:
            target_z = float(z_abs)
        elif dz:
            target_z = origin[2] + dz
        else:
            target_z = None
        if target_z is not None:
            raised = origin.copy()
            raised[2] = target_z
            print(f"Moving to z={raised[2]:.4f} ({raised[2] - origin[2]:+.4f} m) ...")
            link.glide_to(raised, speed=0.05, hz=hz)
            origin = raised

        # send_pose() clips to the workspace silently, which would corrupt the
        # measurement rather than fail it. Check up front instead.
        for name in axes:
            for sign in (+1, -1):
                pt = origin + AXES[name] * sign * amplitude
                if np.any(pt < ROBOT_WORKSPACE_MIN) or np.any(pt > ROBOT_WORKSPACE_MAX):
                    raise SystemExit(
                        f"Stroke end {np.round(pt, 4)} is outside the workspace "
                        f"[{ROBOT_WORKSPACE_MIN}, {ROBOT_WORKSPACE_MAX}].\n"
                        "Lower --amplitude or change --dz."
                    )

        print(
            f"\nPlan: axes {'+'.join(axes)}, +/-{amplitude * 100:.0f} cm about the "
            f"start, speeds {speeds} m/s, {cycles} cycle(s) each, ramp {ramp}s.\n"
            "Ctrl-C stops; the arm holds its last commanded pose.\n"
        )

        for name in axes:
            unit = AXES[name]
            for speed in speeds:
                start = origin - unit * amplitude
                link.glide_to(start, speed=0.06, hz=hz)
                print(f"  {name}-axis @ {speed:.2f} m/s ...", flush=True)
                for _ in range(cycles):
                    prof = _stroke(
                        link, start, unit, 2 * amplitude, speed, ramp, hz, dwell,
                        rows, (name, speed, +1),
                    )
                    far = start + unit * 2 * amplitude
                    _stroke(
                        link, far, -unit, 2 * amplitude, speed, ramp, hz, dwell,
                        rows, (name, speed, -1),
                    )
                if prof.d_cruise <= 0:
                    print(
                        f"    NOTE: no cruise phase at {speed} m/s "
                        f"(stroke too short for ramp={ramp}s) -- raise --amplitude."
                    )
            link.glide_to(origin, speed=0.06, hz=hz)

    except KeyboardInterrupt:
        print("\nInterrupted; arm holds its last commanded pose.")
    finally:
        link.close()

    if not rows:
        return
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["axis", "speed", "dir", "t", "v", "cmd_x", "cmd_y", "cmd_z",
                        "meas_x", "meas_y", "meas_z", "dz", "along"])
            for r in rows:
                w.writerow([r["axis"], r["speed"], r["dir"], f"{r['t']:.4f}",
                            f"{r['v']:.4f}", *[f"{c:.6f}" for c in r["cmd"]],
                            *[f"{c:.6f}" for c in r["meas"]],
                            f"{r['dz']:.6f}", f"{r['along']:.6f}"])
        print(f"\nWrote {len(rows)} samples to {csv_path}")

    report(rows, axes, speeds, kp, floor_mm, settle_s)


def report(rows, axes, speeds, kp, floor_mm=0.05, settle_s=None):
    print("\n" + "=" * 78)
    print("CRUISE-PHASE DEVIATION  (constant velocity only; ramps discarded)")
    print("=" * 78)
    print(f"{'axis':>4} {'speed':>6} {'dir':>4} {'n':>5} {'z dev':>10} {'ripple':>9} "
          f"{'lag along':>11} {'predicted':>10}")
    print(f"{'':>4} {'m/s':>6} {'':>4} {'':>5} {'mm':>10} {'mm rms':>9} "
          f"{'mm':>11} {'mm':>10}")

    stats = {}
    for name in axes:
        for speed in speeds:
            for direction in (+1, -1):
                g = _group(rows, name, speed, direction)
                if not g:
                    continue
                # Reaching cruise velocity is not the same as having settled: the
                # closed loop needs ~4/sqrt(Kp) to converge, and at high speed that
                # is most of the cruise window. Including it inflates the apparent
                # speed scaling and can turn a Coulomb signature into a viscous one.
                settle = 4.0 / np.sqrt(kp) if settle_s is None else settle_s
                cruise = [
                    r for r in g
                    if r["cruise"] > 0
                    and r["v"] >= 0.98 * r["cruise"]
                    and r["t"] >= r["t1"] + settle
                ]
                if len(cruise) < 4:
                    continue
                dev = np.array([r["dz"] for r in cruise])
                along = np.array([r["along"] for r in cruise])
                # (Kd/Kp)*v with Kd = 2*sqrt(Kp) -> 2*v/sqrt(Kp), trailing = negative.
                pred = -2.0 * speed / np.sqrt(kp)
                stats[(name, speed, direction)] = {
                    "dev": float(dev.mean()),
                    "ripple": float(dev.std()),
                    "n": len(cruise),
                    "cruise": cruise,
                }
                print(f"{name:>4} {speed:>6.2f} {direction:>+4d} {len(dev):>5} "
                      f"{dev.mean() * 1000:>+10.2f} {dev.std() * 1000:>9.2f} "
                      f"{along.mean() * 1000:>+11.2f} {pred * 1000:>+10.2f}")

    print("\n" + "=" * 78)
    print("SYMMETRY DECOMPOSITION   dev = sym + anti,  sym = even in v, anti = odd")
    print("=" * 78)
    print(f"{'axis':>4} {'speed':>6} {'sym (mm)':>12} {'anti (mm)':>12}   dominant")
    decomp = {}
    for name in axes:
        for speed in speeds:
            f = stats.get((name, speed, +1))
            b = stats.get((name, speed, -1))
            if not (f and b):
                continue
            sym = (f["dev"] + b["dev"]) / 2
            anti = (f["dev"] - b["dev"]) / 2
            decomp[(name, speed)] = (sym, anti)
            which = "ODD (friction)" if abs(anti) > abs(sym) else "EVEN (Coriolis/gravity)"
            print(f"{name:>4} {speed:>6.2f} {sym * 1000:>+12.2f} {anti * 1000:>+12.2f}   {which}")

    print("\n" + "=" * 78)
    print("SPEED SCALING            what the ratio between the two speeds means")
    print("=" * 78)
    if len(speeds) >= 2:
        lo, hi = min(speeds), max(speeds)
        r = hi / lo
        print(f"speed ratio {hi:.2f}/{lo:.2f} = {r:.2f}x  ->  expect  "
              f"Coulomb 1.0x,  viscous {r:.2f}x,  Coriolis {r * r:.2f}x")
        for name in axes:
            a = decomp.get((name, lo))
            b = decomp.get((name, hi))
            if not (a and b):
                continue
            for label, i in (("sym ", 0), ("anti", 1)):
                if abs(a[i]) < 1e-5:
                    print(f"  {name} {label}: too small at {lo:.2f} m/s to form a ratio")
                    continue
                got = b[i] / a[i]
                if label == "anti":
                    guess = ("Coulomb (dry) friction" if got < (1 + r) / 2
                             else "viscous friction")
                else:
                    guess = ("Coriolis" if got > (r + r * r) / 2
                             else "gravity/payload model error")
                print(f"  {name} {label}: {got:>5.2f}x  -> {guess}")
    else:
        print("Only one speed was run; pass two to --speeds to get this.")

    print("\n" + "=" * 78)
    print("SPECTRUM OF THE CRUISE TRACE   (steady offset vs ringing)")
    print("=" * 78)
    ring_hz = np.sqrt(kp) / (2 * np.pi)
    print(f"task-loop natural frequency at Kp={kp:g}: {ring_hz:.2f} Hz\n")
    for key, s in stats.items():
        c = s["cruise"]
        t = np.array([r["t"] for r in c])
        dt = float(np.median(np.diff(t))) if len(t) > 2 else 0.0
        spec = _spectrum(np.array([r["dz"] for r in c]), dt)
        name, speed, direction = key
        if spec is None:
            print(f"  {name} {speed:.2f} {direction:+d}: cruise too short to transform "
                  f"({s['n']} samples) -- raise --amplitude or lower the speed")
            continue
        print(f"  {name} {speed:.2f} {direction:+d}: peak {spec['peak_hz']:>5.2f} Hz "
              f"at {spec['peak_mm']:>5.2f} mm, ripple {spec['rms'] * 1000:>5.2f} mm rms "
              f"(+/-{spec['resolution_hz']:.1f} Hz bins, Nyquist {spec['nyquist']:.1f} Hz)")
        # Measurement noise always produces *some* argmax. Naming a mechanism for
        # a 0.01 mm peak would invent an elbow mode out of encoder quantisation.
        if spec["peak_mm"] < floor_mm:
            print(f"      below the {floor_mm:.2f} mm floor -- no ringing; the "
                  "deviation is a steady offset")
            continue
        # An FFT reports a frequency whether or not one exists. A single smooth
        # arc across the stroke -- which is what a position-modulated offset looks
        # like -- has its argmax in the lowest non-DC bin every time, and reading
        # that as a resonance is how "0.53 Hz stick-slip" gets invented. Demand
        # that the trace actually crosses its own mean a few times first.
        if spec["crossings"] < 4:
            print(f"      only {spec['crossings']} mean-crossing(s) -- this is a single "
                  "arc, not a ringing.\n      The peak is a segment-length artifact; "
                  "treat it as a position-modulated offset.")
            continue
        tol = max(1.0, spec["resolution_hz"])
        if abs(spec["peak_hz"] - ring_hz) <= tol:
            tag = "~task-loop natural freq -> underdamped (residual_mass_vec)"
        elif spec["peak_hz"] > ring_hz + tol:
            tag = "above the task loop -> nullspace/elbow or structural"
        else:
            tag = "below the task loop -> stick-slip?"
        print(f"      {tag}")

    print("\n" + "=" * 78)
    print("HOW TO ACT ON THIS")
    print("=" * 78)
    print("""\
  anti >> sym, and anti flat with speed
      Coulomb joint friction. Not fixable in config -- needs friction
      compensation in osc_impedance.cpp (rebuild). Meanwhile raise z stiffness:
      Kp.translation: [350.0, 350.0, 700.0] halves it, since dev ~ 1/Kp.

  anti >> sym, and anti scales with speed
      Viscous friction / a velocity-linear term. Same remedies; the z-stiffness
      bump helps proportionally.

  sym >> anti, and sym scales ~4x when speed doubles
      Coriolis. osc_impedance.cpp:123 computes it and never uses it; adding
      `tau_d << tau_d + coriolis;` is a one-line change plus a NUC rebuild.

  ripple concentrated near the task-loop natural frequency
      Lambda_ctrl != Lambda_true, i.e. residual_mass_vec. Try
      residual_mass_vec: [0,0,0,0,0,0,0] first -- config only, no rebuild. It
      also restores the true nullspace projector, so the undamped posture spring
      at line 219 stops leaking into the EE. PInverse already guards the
      singular case (control_utils.h:15), so nothing needs the padding.

  ripple broadband, or peaking near Nyquist
      Faster than this 50 Hz sampling can resolve. Suspect the nullspace spring;
      osc_position_impedance.cpp:240 has that exact line commented out already.

  lag along vs predicted
      If these disagree badly, the controller is not behaving like the model
      above and the rest of this report is on sand -- check Kp actually reached
      the NUC before trusting anything else.""")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="left", choices=["right", "left"])
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--dz", type=float, default=0.0,
                   help="metres to raise RELATIVE to the current pose. Not repeatable: "
                        "the arm parks where the last run left it, so consecutive runs "
                        "climb. Use --z to compare two runs.")
    p.add_argument("--z", type=float, default=None, dest="z_abs",
                   help="absolute base-frame z to test at. Prefer this over --dz for "
                        "any A/B comparison -- the deviation depends on arm posture, "
                        "so two runs at different heights are not comparable.")
    p.add_argument("--amplitude", type=float, default=0.12,
                   help="metres, half-stroke; needs to be long enough to cruise "
                        "(at 0.35 m/s and ramp=0.15 this is ~27 cruise samples)")
    p.add_argument("--speeds", default="0.08,0.35",
                   help="comma-separated m/s; two speeds are what makes the scaling test work")
    p.add_argument("--ramp", type=float, default=0.15, help="accel/decel seconds")
    p.add_argument("--axes", default="xy", help="subset of 'xy'")
    p.add_argument("--cycles", type=int, default=2, help="out-and-back pairs per axis/speed")
    p.add_argument("--dwell", type=float, default=0.3, help="seconds at rest between strokes")
    p.add_argument("--kp", type=float, default=350.0,
                   help="Kp.translation from osc-pose-controller.yml")
    p.add_argument("--csv", default=None, help="write the raw samples here")
    p.add_argument("--ripple-floor", type=float, default=0.05,
                   help="mm; spectral peaks below this are called noise, not a mode")
    p.add_argument("--settle", type=float, default=None,
                   help="seconds of each cruise window to discard as settling. "
                        "Default 4/sqrt(Kp), which isolates steady state. Pass 0 to "
                        "keep everything -- correct when you want the error under "
                        "PLAY conditions, where strokes are shorter than the settling "
                        "time and the arm never reaches steady state anyway.")
    a = p.parse_args()
    run(
        a.arm, a.hz, a.dz, a.amplitude,
        [float(s) for s in a.speeds.split(",")],
        a.ramp, [c for c in a.axes if c in AXES], a.cycles, a.dwell, a.kp, a.csv,
        a.ripple_floor, a.z_abs, a.settle,
    )
