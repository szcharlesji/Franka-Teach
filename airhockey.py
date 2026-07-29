"""Bimanual keyboard air hockey teleop.

Prerequisites (see README):
  1. deoxys arm processes running on the NUC for both robots
  2. one franka_server.py per arm, e.g.
       python3 franka_server.py arm=right deoxys_config_path=deoxys_right_fast.yml \
           control_freq=50 num_steps=1
       python3 franka_server.py arm=left  deoxys_config_path=deoxys_left_fast.yml \
           control_freq=50 num_steps=1
  3. python3 camera_server.py   (optional, for the video feed)

Then:
  python3 airhockey.py --calibrate --arm right    # teach the right box
  python3 airhockey.py --calibrate --arm left     # teach the left box
  python3 airhockey.py --verify --arm right       # trace the perimeter
  python3 airhockey.py                            # play
"""

import argparse
import sys

import numpy as np
import pygame

from frankateach.airhockey import config as ahconfig
from frankateach.airhockey.calibrate import calibrate, trace_perimeter
from frankateach.airhockey.keyboard import (
    KEY_FREEZE,
    KEY_HOME,
    KEY_RELEASE,
    intent_from_keys,
    resolve_keys,
)
from frankateach.airhockey.operator import ArmOperator
from frankateach.airhockey.view import CameraFeed, draw


def run_calibrate(arm, cfg, do_trace):
    result = calibrate(arm, cfg)
    if result is None:
        return 1
    box, link = result
    try:
        if do_trace or input("\nTrace the perimeter now? [y/N] ").strip().lower() == "y":
            trace_perimeter(arm, box, cfg, link=link)
    finally:
        link.close()
    return 0


def run_verify(arm, cfg):
    box = ahconfig.load_box(arm)
    print(f"{arm} box: centre {box.center}, yaw {box.yaw:.4f} rad, "
          f"half extents {box.half_extents}, plane_z {box.plane_z:.4f}")
    trace_perimeter(arm, box, cfg)
    return 0


def run_play(arms, cfg, publish):
    boxes = {arm: ahconfig.load_box(arm) for arm in arms}
    keynames = {arm: cfg["arms"][arm]["keys"] for arm in arms}
    keymaps = {arm: resolve_keys(keynames[arm]) for arm in arms}

    pygame.init()
    w, h = cfg.get("window") or [1280, 720]
    screen = pygame.display.set_mode((int(w), int(h)))
    pygame.display.set_caption("Air hockey — " + " + ".join(arms))
    font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 18)
    small = pygame.font.SysFont("menlo,dejavusansmono,monospace", 14)

    camera = None
    if cfg.get("view_cam_id") is not None:
        camera = CameraFeed(int(cfg["view_cam_id"]))
        camera.start()

    operators = {arm: ArmOperator(arm, boxes[arm], cfg, publish=publish) for arm in arms}
    for op in operators.values():
        op.start()

    print("Bringing arms to their box centres...")
    for arm, op in operators.items():
        if not op.wait_ready(timeout=90):
            print(f"[{arm}] did not come up in time")
    dead = [a for a, op in operators.items() if not op.get_status().connected]
    if dead:
        print(f"WARNING: {', '.join(dead)} failed to start — see the error above.")

    # Arms come up frozen; press a movement key to take control.
    released = True
    homing = False
    movement_keys = {k for m in keymaps.values() for k in m.values()}
    clock = pygame.time.Clock()
    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == KEY_RELEASE:
                        released = True
                        homing = False
                    elif event.key == KEY_HOME:
                        # Latched: homing runs until it arrives or you steer.
                        homing = True
                        released = False
                    elif event.key in movement_keys:
                        released = False
                        homing = False

            focused = pygame.key.get_focused()
            pressed = pygame.key.get_pressed()
            frozen = released or not focused or pressed[KEY_FREEZE]
            if frozen:
                homing = False
            if homing and all(
                np.linalg.norm(op.get_status().box_pos) < 0.005
                for op in operators.values()
            ):
                homing = False

            for arm, op in operators.items():
                vx, vy = intent_from_keys(pressed, keymaps[arm])
                op.set_intent(vx, vy, frozen=frozen, home=homing)

            if not focused:
                banner = "WINDOW NOT FOCUSED — arms frozen"
            elif released:
                banner = "RELEASED — press a movement key to take control"
            elif pressed[KEY_FREEZE]:
                banner = "FREEZE"
            elif homing:
                banner = "HOMING — press a movement key to take over"
            else:
                banner = ""

            draw(screen, font, small, operators, boxes, keynames, camera, banner)
            clock.tick(60)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Parking arms...")
        for op in operators.values():
            op.stop()
        for op in operators.values():
            op.join(timeout=5)
        if camera is not None:
            camera.stop()
        pygame.quit()
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibrate", action="store_true", help="teach one arm's play box")
    p.add_argument("--verify", action="store_true", help="trace a calibrated perimeter")
    p.add_argument("--arm", choices=["right", "left"], help="arm for --calibrate/--verify")
    p.add_argument("--arms", default="right,left", help="arms to play with")
    p.add_argument("--trace", action="store_true", help="trace after calibrating, no prompt")
    p.add_argument("--no-publish", action="store_true",
                   help="skip state publishing (disables collect_data.py recording)")
    args = p.parse_args()

    cfg = ahconfig.load()

    if args.calibrate or args.verify:
        if not args.arm:
            p.error("--calibrate/--verify need --arm right|left")
        if args.calibrate:
            return run_calibrate(args.arm, cfg, args.trace)
        return run_verify(args.arm, cfg)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    return run_play(arms, cfg, publish=not args.no_publish)


if __name__ == "__main__":
    sys.exit(main())
