"""Teach one arm's play box by jogging to its four corners.

Opens a small pygame window (it needs focus for key-up events to work), jogs the
arm under WASD + Q/E, and records a corner each time you press SPACE. After four
corners it fits a conservative inscribed rectangle -- see box.box_from_corners --
writes it into configs/airhockey.yaml, and offers to trace the perimeter so you
can watch the real limits before playing.
"""

import numpy as np
import pygame

from frankateach.airhockey import config as ahconfig
from frankateach.airhockey.box import box_from_corners
from frankateach.airhockey.control import ArmLink, RateLimiter, ramp
from frankateach.airhockey.keyboard import (
    JOG_DOWN,
    JOG_UP,
    KEY_DONE,
    KEY_RECORD,
    KEY_REDO,
    resolve_keys,
)

CORNER_NAMES = [
    "corner 1 (near-right)",
    "corner 2 (far-right)",
    "corner 3 (far-left)",
    "corner 4 (near-left)",
]

HELP = [
    "W/S  jog away / toward you        A/D  jog left / right",
    "E/Q  raise / lower the mallet",
    "SPACE  record this corner         R  redo last corner",
    "ENTER  finish (needs 4 corners)   ESC  abort",
]


def _draw(screen, font, arm, corners, pos, idx, msg):
    screen.fill((18, 18, 22))
    y = 16

    def line(text, colour=(220, 220, 225), dy=24):
        nonlocal y
        screen.blit(font.render(text, True, colour), (18, y))
        y += dy

    line(f"CALIBRATING {arm.upper()} ARM", (120, 200, 255), 34)
    for h in HELP:
        line(h, (150, 150, 160), 20)
    y += 12
    line(f"EE  x {pos[0]:+.4f}   y {pos[1]:+.4f}   z {pos[2]:+.4f}", (255, 235, 140), 30)

    for i, name in enumerate(CORNER_NAMES):
        if i < len(corners):
            c = corners[i]
            line(f"  [{i + 1}] {name:<22} {c[0]:+.4f} {c[1]:+.4f} {c[2]:+.4f}", (140, 230, 150))
        elif i == idx:
            line(f"  [{i + 1}] {name:<22} <- jog here, SPACE to record", (255, 200, 90))
        else:
            line(f"  [{i + 1}] {name:<22} -", (110, 110, 120))

    if msg:
        y += 10
        for m in msg.split("\n"):
            line(m, (255, 150, 150), 20)
    pygame.display.flip()


def calibrate(arm, cfg, screen=None, font=None):
    keys = resolve_keys(cfg["arms"][arm]["keys"])
    hz = float(cfg["control_hz"])
    jog_speed = float(cfg["jog_speed"])
    jog_z_speed = float(cfg["jog_z_speed"])
    accel_time = float(cfg["accel_time"])

    owns_display = screen is None
    if owns_display:
        pygame.init()
        screen = pygame.display.set_mode((760, 420))
        pygame.display.set_caption(f"Air hockey — calibrate {arm}")
    if font is None:
        font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 15)

    link = ArmLink(arm)
    limiter = RateLimiter(hz)
    corners, msg = [], ""
    handed_off = False  # the caller owns the link only on the success path

    try:
        print(f"Resetting {arm} arm to its ready pose...")
        state = link.reset()
        target = np.asarray(state.pos, dtype=np.float64)
        vel = np.zeros(3)
        print("Jog with WASD (+ Q/E for height). SPACE records a corner.")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_ESCAPE:
                    print("Calibration aborted.")
                    return None
                if event.key == KEY_RECORD and len(corners) < 4:
                    corners.append(np.asarray(link.last_state.pos, dtype=np.float64).copy())
                    msg = ""
                    print(f"  recorded {CORNER_NAMES[len(corners) - 1]}: {corners[-1]}")
                elif event.key == KEY_REDO and corners:
                    dropped = corners.pop()
                    print(f"  dropped {dropped}")
                    msg = ""
                elif event.key == KEY_DONE:
                    if len(corners) == 4:
                        running = False
                    else:
                        msg = f"Need 4 corners, have {len(corners)}."

            if not pygame.key.get_focused():
                intent = np.zeros(3)
            else:
                pressed = pygame.key.get_pressed()
                intent = np.array(
                    [
                        (float(pressed[keys["up"]]) - float(pressed[keys["down"]])) * jog_speed,
                        (float(pressed[keys["left"]]) - float(pressed[keys["right"]])) * jog_speed,
                        (float(pressed[JOG_UP]) - float(pressed[JOG_DOWN])) * jog_z_speed,
                    ]
                )

            dt = 1.0 / hz
            vel = ramp(vel, intent, accel_time, dt, jog_speed)
            target = target + vel * dt

            # Leash to the real EE so a held key cannot build up a runaway target.
            actual = np.asarray(link.last_state.pos, dtype=np.float64)
            delta = target - actual
            dist = np.linalg.norm(delta)
            if dist > 0.05:
                target = actual + delta / dist * 0.05

            link.send_pose(target)
            _draw(screen, font, arm, corners, link.last_state.pos, len(corners), msg)
            limiter.sleep()

        quat = np.asarray(link.last_state.quat, dtype=np.float64)
        box, warnings = box_from_corners(
            np.array(corners), quat, margin=float(cfg["margin"])
        )

        print("\n" + "=" * 68)
        print(f"{arm.upper()} ARM PLAY BOX")
        print(f"  centre       : {box.center}")
        print(f"  yaw          : {np.degrees(box.yaw):+.2f} deg")
        print(f"  half extents : {box.half_extents * 1000} mm")
        print(f"  play area    : {box.half_extents[0] * 2:.3f} x {box.half_extents[1] * 2:.3f} m")
        print(f"  plane_z      : {box.plane_z:.4f} m  (corner spread {box.z_spread * 1000:.1f} mm)")
        print(f"  fixed_quat   : {box.quat}")
        for w in warnings:
            print(f"\n  !! {w}")
        print("=" * 68)

        path = ahconfig.save_box(arm, box)
        print(f"Wrote {arm} calibration to {path}")
        handed_off = True
        return box, link
    finally:
        if not handed_off:
            link.close()
        if owns_display:
            pygame.quit()


def trace_perimeter(arm, box, cfg, link=None):
    """Walk the rectangle edges slowly so you can eyeball the real limits."""
    owns_link = link is None
    if owns_link:
        link = ArmLink(arm, quat=box.quat)
        link.get_state()
    speed = float(cfg["trace_speed"])
    try:
        print(f"Tracing {arm} perimeter at {speed} m/s — watch the table edge.")
        rc = box.rect_corners()
        link.glide_to(np.array([rc[0][0], rc[0][1], box.plane_z]), speed=speed)
        for i in range(1, 5):
            c = rc[i % 4]
            link.glide_to(np.array([c[0], c[1], box.plane_z]), speed=speed)
            print(f"  corner {i % 4 + 1}: {c}")
        link.glide_to(box.home, speed=speed)
        print("Trace complete; parked at box centre.")
    finally:
        if owns_link:
            link.close()
