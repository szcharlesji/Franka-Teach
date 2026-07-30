"""Does the arm hold the pose it is told to? Isolates force errors from lag.

scripts/rate_test.py sweeps a sine and reports one Euclidean error, which cannot
tell a *constant sag* (gravity/payload compensation is wrong) from a *lag* (the
controller is too soft or too slow). This holds still instead, so anything left is
a steady-state force error.

    python3 scripts/hold_test.py --arm left --dz 0.12

Three phases:

1. HOLD    - command one fixed pose and measure the steady-state error per axis.
             A non-zero z error while stationary is compensation error, not lag.
             The implied force is Kp_translation * error, so with Kp=150 N/m a
             33 mm sag is about 5 N -- roughly a 0.5 kg unmodelled tool.
2. STEP    - step up, settle, step down, settle. Gravity error is asymmetric in z
             (it fights you one way and helps the other); pure lag is symmetric.
3. STIFFNESS - repeat the hold at the same pose after you change Kp. If the sag
             scales as 1/Kp it is a force error. If it does not move, it is not.

Nothing here needs the mallet off, and in fact it is more informative WITH the
mallet on, since an unmodelled payload is the thing you are looking for.
"""

import argparse
import time

import numpy as np

from frankateach.airhockey.control import ArmLink, RateLimiter, assert_sole_client


def settle_and_measure(link, target, hz, hold_s, label):
    """Command `target` for hold_s seconds; report error over the last half."""
    limiter = RateLimiter(hz)
    errs = []
    t0 = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= hold_s:
            break
        state, _ = link.send_pose(target)
        if elapsed > hold_s / 2:  # discard the transient
            errs.append(np.asarray(state.pos, dtype=np.float64) - target)
        limiter.sleep()

    errs = np.array(errs)
    mean = errs.mean(axis=0)
    std = errs.std(axis=0)
    print(f"\n{label}")
    print(f"  commanded      : {np.round(target, 4)}")
    print(f"  measured - cmd : x {mean[0] * 1000:+7.2f}  y {mean[1] * 1000:+7.2f}  "
          f"z {mean[2] * 1000:+7.2f}   (mm, mean)")
    print(f"  jitter (std)   : x {std[0] * 1000:7.2f}  y {std[1] * 1000:7.2f}  "
          f"z {std[2] * 1000:7.2f}   (mm)")
    return mean


def run(arm, hz, dz, step, hold_s, kp):
    assert_sole_client(arm)
    link = ArmLink(arm)
    state = link.get_state()
    # No reset: hold wherever the arm is. link.quat is normally latched by reset().
    link.quat = np.asarray(state.quat, dtype=np.float64)
    origin = np.asarray(state.pos, dtype=np.float64).copy()
    print(f"Starting from {np.round(origin, 4)}")

    try:
        if dz:
            raised = origin.copy()
            raised[2] += dz
            print(f"Raising {dz * 100:+.1f} cm to z={raised[2]:.4f} ...")
            link.glide_to(raised, speed=0.05, hz=hz)
            origin = raised

        hold = settle_and_measure(link, origin, hz, hold_s, "1. HOLD (stationary)")

        up = origin.copy()
        up[2] += step
        down = origin.copy()
        down[2] -= step
        link.glide_to(up, speed=0.05, hz=hz)
        e_up = settle_and_measure(link, up, hz, hold_s, f"2a. STEP UP  (+{step * 100:.0f} cm)")
        link.glide_to(down, speed=0.05, hz=hz)
        e_down = settle_and_measure(link, down, hz, hold_s, f"2b. STEP DOWN (-{step * 100:.0f} cm)")
        link.glide_to(origin, speed=0.05, hz=hz)

        sag = hold[2]
        print("\n" + "=" * 68)
        print(f"steady-state z error while stationary : {sag * 1000:+.2f} mm")
        print(f"implied unmodelled force at Kp={kp:g}    : {abs(sag) * kp:.2f} N "
              f"(~{abs(sag) * kp / 9.81:.2f} kg)")
        asym = abs(e_up[2] - e_down[2])
        print(f"up/down z error asymmetry             : {asym * 1000:.2f} mm")
        print("=" * 68)
        print("\nReading it:")
        if abs(sag) > 0.005:
            print(f"  * {sag * 1000:+.1f} mm of steady-state z error with the arm NOT moving is a")
            print("    force error, not lag. Set the end-effector load in Desk (mass and")
            print("    CoM) if anything is bolted to the flange, then re-run.")
        else:
            print("  * z holds within 5 mm while stationary: gravity/payload compensation")
            print("    is close. Any error in rate_test.py is then lag, not sag.")
        if asym > 0.005:
            print(f"  * {asym * 1000:.1f} mm up/down asymmetry points at gravity specifically;")
            print("    a pure delay would be symmetric.")
        print("  * Then double Kp translation in frankateach/configs/osc-pose-controller.yml")
        print("    and re-run: a force error halves, a delay does not move.")
    finally:
        link.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="left", choices=["right", "left"])
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--dz", type=float, default=0.0, help="raise this far before testing")
    p.add_argument("--step", type=float, default=0.05, help="metres for the step test")
    p.add_argument("--hold", type=float, default=3.0, help="seconds per hold")
    p.add_argument(
        "--kp",
        type=float,
        default=150.0,
        help="Kp translation from osc-pose-controller.yml, for the force estimate",
    )
    a = p.parse_args()
    run(a.arm, a.hz, a.dz, a.step, a.hold, a.kp)
