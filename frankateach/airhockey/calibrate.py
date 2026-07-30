"""Calibration helpers.

The interactive jogging half now lives in webapp.build_calibrate_app (the NUC
has no monitor, so the UI is a browser page). What remains here is the part with
no UI: walking the finished rectangle so you can watch its real limits.

The geometry itself -- fitting a conservative inscribed rectangle to four taught
corners -- is in box.box_from_corners.
"""

import numpy as np

from frankateach.airhockey.control import ArmLink

CORNER_NAMES = [
    "corner 1 (near-right)",
    "corner 2 (far-right)",
    "corner 3 (far-left)",
    "corner 4 (near-left)",
]


def trace_perimeter(arm, box, cfg, link=None):
    """Walk the rectangle edges slowly so you can eyeball the real limits.

    Pass `link` to reuse an existing connection (the calibration page does this
    with its jog operator's link); otherwise a temporary one is opened.
    """
    owns_link = link is None
    if owns_link:
        link = ArmLink(arm, quat=box.quat)
        link.get_state()
    speed = float(cfg["trace_speed"])
    try:
        print(f"Tracing {arm} perimeter at {speed} m/s — watch the table edge.")
        rc = box.rect_corners()
        # Corner z comes from box.z_at, so a tilted plane is traced along its own
        # slope. Straight lines between two points on a plane stay on it.
        link.glide_to(np.array([rc[0][0], rc[0][1], box.z_at(rc[0])]), speed=speed)
        for i in range(1, 5):
            c = rc[i % 4]
            link.glide_to(np.array([c[0], c[1], box.z_at(c)]), speed=speed)
            print(f"  {CORNER_NAMES[i % 4]}: {c}")
        link.glide_to(box.home, speed=speed)
        print("Trace complete; parked at box centre.")
    finally:
        if owns_link:
            link.close()
