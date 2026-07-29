"""Bimanual keyboard air hockey teleop, driven from a browser.

The UI is a web page rather than a desktop window because the NUC has no
monitor. Everything runs on the NUC; you reach the page through an SSH tunnel,
exactly like the Franka Desk panels:

    ssh -L 8080:localhost:8080 franka        # then open http://localhost:8080

Only key intent crosses the network. Each arm's 50 Hz control loop stays on the
NUC talking to deoxys over localhost, and freezes itself if the browser goes
quiet (see the watchdog in operator.py).

Prerequisites:
  1. deoxys franka-interface running on the NUC for each arm you want
  2. one franka_server.py per arm, e.g.
       python3 franka_server.py arm=left deoxys_config_path=deoxys_left_fast.yml \
           control_freq=50 num_steps=1
  3. python3 camera_server.py   (optional, for the video pane)

Then:
  python3 airhockey.py --calibrate --arm left     # teach the left box
  python3 airhockey.py --verify --arm left        # trace the perimeter
  python3 airhockey.py --arms left                # play (single arm)
  python3 airhockey.py                            # play (both arms)
"""

import argparse
import sys

from frankateach.airhockey import config as ahconfig
from frankateach.airhockey.calibrate import trace_perimeter
from frankateach.airhockey.webapp import serve, serve_calibrate


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--calibrate", action="store_true", help="teach one arm's play box")
    p.add_argument("--verify", action="store_true", help="trace a calibrated perimeter")
    p.add_argument("--arm", choices=["right", "left"], help="arm for --calibrate/--verify")
    p.add_argument("--arms", default="right,left", help="arms to play with")
    p.add_argument("--port", type=int, default=8080, help="web UI port")
    p.add_argument(
        "--bind",
        default="127.0.0.1",
        help="bind address; keep the default and use an SSH tunnel",
    )
    p.add_argument(
        "--no-publish",
        action="store_true",
        help="skip state publishing (disables collect_data.py recording)",
    )
    args = p.parse_args()

    cfg = ahconfig.load()

    if args.calibrate or args.verify:
        if not args.arm:
            p.error("--calibrate/--verify need --arm right|left")
        if args.calibrate:
            return serve_calibrate(args.arm, cfg, port=args.port, bind=args.bind)
        box = ahconfig.load_box(args.arm)
        print(
            f"{args.arm} box: centre {box.center}, yaw {box.yaw:.4f} rad, "
            f"half extents {box.half_extents}, plane_z {box.plane_z:.4f}"
        )
        trace_perimeter(args.arm, box, cfg)
        return 0

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    boxes = {arm: ahconfig.load_box(arm) for arm in arms}
    serve(
        arms,
        cfg,
        boxes,
        publish=not args.no_publish,
        port=args.port,
        bind=args.bind,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
