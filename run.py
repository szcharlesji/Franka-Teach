"""One command to bring up air hockey: interfaces, servers, control loops, web UI.

Replaces the four-terminal dance of auto_arm.sh + franka_server.py per arm +
airhockey.py. Everything is a child of this process, so one Ctrl-C takes the whole
stack down in the right order.

    python3 run.py                       # both arms, play mode
    python3 run.py --arms left           # one arm
    python3 run.py --calibrate left      # come up straight into calibrate
    python3 run.py --no-interface        # franka-interface already running in tmux

Calibration is a *mode*, not a separate program: hit "Calibrate" for an arm in the
web UI and that arm switches to a jog operator; finish and it reloads the box and
returns to play. Nothing below the control loop restarts.

Prerequisites this cannot do for you: brakes released and FCI enabled in Desk for
each arm. See RUNBOOK.md.
"""

import argparse
import signal
import sys

from frankateach.airhockey import config as ahconfig
from frankateach.airhockey.session import CALIBRATE, PLAY, ArmSession
from frankateach.airhockey.session_app import CameraRelay, serve
from frankateach.airhockey.supervisor import Supervisor


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arms", default="left,right", help="comma list of arms to bring up")
    p.add_argument(
        "--calibrate",
        metavar="ARM",
        choices=["left", "right"],
        help="start with this arm in calibrate mode",
    )
    p.add_argument("--port", type=int, default=8080, help="web UI port")
    p.add_argument(
        "--bind",
        default="127.0.0.1",
        help="bind address; keep the default and use an SSH tunnel",
    )
    p.add_argument("--speed", type=float, help="override configs/airhockey.yaml speed")
    p.add_argument("--control-hz", type=float, help="override control_hz")
    p.add_argument(
        "--num-steps",
        type=int,
        default=1,
        help="OSC deltas per request in franka_server (1 for 50 Hz play)",
    )
    p.add_argument(
        "--no-interface",
        action="store_true",
        help="do not start auto_arm.sh; use franka-interface instances already running",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="do not start camera_server.py and skip the video pane",
    )
    p.add_argument(
        "--no-publish",
        action="store_true",
        help="skip state publishing (disables collect_data.py recording)",
    )
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.calibrate and args.calibrate not in arms:
        p.error(f"--calibrate {args.calibrate} but --arms is {arms}")

    cfg = ahconfig.load()
    if args.speed is not None:
        cfg["speed"] = args.speed
    if args.control_hz is not None:
        cfg["control_hz"] = args.control_hz

    want_camera = not args.no_video and cfg.get("view_cam_id") is not None
    supervisor = Supervisor(
        arms,
        cfg,
        control_freq=int(float(cfg["control_hz"])),
        num_steps=args.num_steps,
        with_interface=not args.no_interface,
        with_camera=want_camera,
    )
    sessions = {}
    camera = None

    def shutdown(*_):
        print("\nShutting down...")
        for sess in sessions.values():
            sess.stop()
        if camera is not None:
            camera.stop()
        supervisor.stop()

    # SIGTERM must behave like Ctrl-C. web.run_app installs its own signal
    # handling once the loop is running, so a plain signal.signal() handler here
    # gets replaced and `kill <pid>` leaves franka-interface, the servers and the
    # camera all running. Raising KeyboardInterrupt instead lands in the same path
    # aiohttp already unwinds cleanly, and our finally: shutdown() then runs.
    def _sigterm(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm)

    try:
        supervisor.start()

        for arm in arms:
            sess = ArmSession(arm, cfg, publish=not args.no_publish)
            mode = CALIBRATE if args.calibrate == arm else PLAY
            if not sess.start(mode):
                raise RuntimeError(f"{arm}: {sess.error}")
            sessions[arm] = sess
            print(f"[{arm}] {mode} ready")

        if want_camera:
            # camera_server.py was started by the supervisor; this is the relay that
            # subscribes to its ZMQ bus and re-encodes for the page.
            camera = CameraRelay(cfg["view_cam_id"])
            camera.start()

        return serve(
            sessions,
            cfg,
            supervisor=supervisor,
            camera=camera,
            port=args.port,
            bind=args.bind,
        )
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        return 1
    finally:
        shutdown()


if __name__ == "__main__":
    sys.exit(main())
