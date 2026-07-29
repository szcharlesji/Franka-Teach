"""Phase 0 gate: can one arm sustain a 50 Hz command round trip?

Drives a slow sine in x around the current EE position and reports, for each
`num_steps` setting, the achieved round-trip rate and how far the measured EE
lags the commanded target. Nothing else in the air hockey stack should be
trusted until this passes.

Run with the arm's franka_server already up, e.g.

    python3 franka_server.py arm=right deoxys_config_path=deoxys_right_fast.yml \
        control_freq=50 num_steps=1
    python3 scripts/rate_test.py --arm right --hz 50

The server's num_steps is fixed at startup, so to sweep it restart the server
once per value. This script reports what it measured against whatever the
server is currently running.
"""

import argparse
import pickle
import time

import numpy as np

from frankateach.constants import GRIPPER_CLOSE, HOST, arm_ports
from frankateach.messages import FrankaAction, FrankaState
from frankateach.network import create_request_socket


def run(arm, hz, seconds, amplitude, period, do_reset):
    control_port, _, _ = arm_ports(arm)
    sock = create_request_socket(HOST, control_port)
    dt = 1.0 / hz

    try:
        if do_reset:
            print("Resetting to ready pose...")
            sock.send(
                pickle.dumps(
                    FrankaAction(
                        pos=np.zeros(3),
                        quat=np.zeros(4),
                        gripper=GRIPPER_CLOSE,
                        reset=True,
                        timestamp=time.time(),
                    ),
                    protocol=-1,
                )
            )
            pickle.loads(sock.recv())

        sock.send(b"get_state")
        state: FrankaState = pickle.loads(sock.recv())
        origin = np.array(state.pos, dtype=np.float64)
        quat = np.array(state.quat, dtype=np.float64)
        print(f"Origin pos={origin}, holding quat={quat}")
        print(f"Sweeping +/-{amplitude * 100:.0f} cm in x, period {period}s\n")

        periods, errors = [], []
        t0 = time.perf_counter()
        next_tick = t0
        last = t0

        while True:
            now = time.perf_counter()
            elapsed = now - t0
            if elapsed >= seconds:
                break

            target = origin.copy()
            target[0] += amplitude * np.sin(2 * np.pi * elapsed / period)

            sock.send(
                pickle.dumps(
                    FrankaAction(
                        pos=target.astype(np.float32),
                        quat=quat.astype(np.float32),
                        gripper=GRIPPER_CLOSE,
                        reset=False,
                        timestamp=time.time(),
                    ),
                    protocol=-1,
                )
            )
            state = pickle.loads(sock.recv())

            done = time.perf_counter()
            periods.append(done - last)
            last = done
            errors.append(np.linalg.norm(np.array(state.pos) - target))

            # Pace to the requested rate without busy-waiting.
            next_tick += dt
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # We are already behind; resync so lateness does not compound.
                next_tick = time.perf_counter()

        periods = np.array(periods[1:])
        errors = np.array(errors[1:])
        rates = 1.0 / periods

        print(f"ticks              : {len(periods)}")
        print(f"achieved rate mean : {rates.mean():6.2f} Hz")
        print(f"achieved rate p5   : {np.percentile(rates, 5):6.2f} Hz  (worst case)")
        print(f"round trip mean    : {periods.mean() * 1000:6.2f} ms")
        print(f"round trip p95     : {np.percentile(periods, 95) * 1000:6.2f} ms")
        print(f"budget at {hz} Hz    : {dt * 1000:6.2f} ms")
        print(f"tracking err mean  : {errors.mean() * 1000:6.2f} mm")
        print(f"tracking err max   : {errors.max() * 1000:6.2f} mm")

        ok_rate = np.percentile(rates, 5) >= hz * 0.95
        ok_err = errors.mean() < 0.005
        print()
        print(f"rate gate  (p5 >= {hz * 0.95:.1f} Hz) : {'PASS' if ok_rate else 'FAIL'}")
        print(f"error gate (mean < 5 mm)   : {'PASS' if ok_err else 'FAIL'}")
        if not (ok_rate and ok_err):
            print(
                "\nIf the rate gate fails, drop the server to num_steps=1, confirm "
                "POLICY_RATE matches control_freq in BOTH the client yml and the "
                "NUC-side config, then retry. If it still fails, lower control_hz."
            )
        return ok_rate and ok_err
    finally:
        sock.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="right", choices=["right", "left"])
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--amplitude", type=float, default=0.05, help="metres, half-stroke")
    p.add_argument("--period", type=float, default=4.0, help="seconds per sine cycle")
    p.add_argument("--no-reset", action="store_true", help="skip the joint reset")
    a = p.parse_args()
    run(a.arm, a.hz, a.seconds, a.amplitude, a.period, not a.no_reset)
