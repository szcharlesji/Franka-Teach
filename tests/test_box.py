"""Geometry tests for the air hockey play box.

Pure numpy -- no robot, no deoxys, no pygame. Run directly:

    python3 tests/test_box.py

The important one is the inscribed-rectangle property: whatever quadrilateral
you teach, the playable rectangle and every clip result must land inside it.
That is what keeps the mallet off the table edge.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frankateach.airhockey.box import PlayBox, box_from_corners, _rot  # noqa: E402

QUAT = [1.0, 0.0, 0.0, 0.0]
fails = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        fails.append(name)


def in_poly(p, poly):
    """Point strictly inside convex polygon (ordered)."""
    n = len(poly)
    signs = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        d = p - a
        signs.append(e[0] * d[1] - e[1] * d[0])
    return all(s > -1e-9 for s in signs) or all(s < 1e-9 for s in signs)


# --- 1. axis-aligned rectangle ------------------------------------------
c = np.array([[0.30, -0.24, 0.12], [0.62, -0.24, 0.12], [0.62, 0.24, 0.12], [0.30, 0.24, 0.12]])
box, warns = box_from_corners(c, QUAT, margin=0.015)
check("axis-aligned yaw ~ 0", abs(box.yaw) < 1e-6, f"(yaw={np.degrees(box.yaw):.3f} deg)")
check("axis-aligned extents", np.allclose(box.half_extents, [0.16 - 0.015, 0.24 - 0.015]),
      f"(he={box.half_extents})")
check("axis-aligned center", np.allclose(box.center, [0.46, 0.0]))
check("axis-aligned no warnings", not warns, f"({warns})")
check("plane_z", abs(box.plane_z - 0.12) < 1e-12)

# clip behaviour
check("clip inside is identity", np.allclose(box.clip([0.46, 0.0]), [0.46, 0.0]))
far = box.clip([10.0, 10.0])
check("clip far corner lands on rect corner", np.allclose(far, [0.46 + 0.145, 0.225]), f"({far})")
check("clip result is contained", box.contains(box.clip([10.0, -10.0])))
check("contains rejects outside", not box.contains([0.9, 0.0]))

# --- 2. rotated rectangle ------------------------------------------------
th = np.radians(20.0)
local = np.array([[-0.16, -0.24], [0.16, -0.24], [0.16, 0.24], [-0.16, 0.24]])
ctr = np.array([0.46, 0.05])
rot_xy = np.array([_rot(th) @ p + ctr for p in local])
c2 = np.hstack([rot_xy, np.full((4, 1), 0.118)])
box2, warns2 = box_from_corners(c2, QUAT, margin=0.015)
check("rotated yaw recovered", abs(np.degrees(box2.yaw) - 20.0) < 1e-6,
      f"(yaw={np.degrees(box2.yaw):.4f} deg)")
check("rotated extents preserved", np.allclose(box2.half_extents, [0.145, 0.225]),
      f"(he={box2.half_extents})")
check("rotated center", np.allclose(box2.center, ctr))
check("rotated no warnings", not warns2, f"({warns2})")

# intent maps along table axes, not base axes
v = box2.rotate_intent(1.0, 0.0)
check("intent rotated into base frame", np.allclose(v, [np.cos(th), np.sin(th)]), f"({v})")

# --- 3. THE SAFETY PROPERTY: skewed quad, rect must stay inside ----------
rng = np.random.default_rng(0)
worst = 0.0
for trial in range(2000):
    base = np.array([[-0.16, -0.24], [0.16, -0.24], [0.16, 0.24], [-0.16, 0.24]])
    jitter = rng.uniform(-0.03, 0.03, size=(4, 2))   # up to 3 cm of sloppiness
    thj = rng.uniform(-np.pi / 6, np.pi / 6)
    quad = np.array([_rot(thj) @ p + ctr for p in base + jitter])
    cj = np.hstack([quad, np.full((4, 1), 0.118)])
    try:
        bj, _ = box_from_corners(cj, QUAT, margin=0.0)   # margin 0 = hardest case
    except ValueError:
        continue
    for rc in bj.rect_corners():
        if not in_poly(rc, quad):
            worst = max(worst, 1.0)
    # also sample the clip output over a grid of wild inputs
    for p in rng.uniform(-1, 1, size=(20, 2)):
        if not in_poly(bj.clip(ctr + p), quad):
            worst = max(worst, 1.0)
check("inscribed rect always inside taught quad (2000 random skewed quads)", worst == 0.0)

# --- 4. warnings fire ----------------------------------------------------
c4 = c.copy()
c4[2, 2] += 0.012   # one corner 12 mm high
_, warns4 = box_from_corners(c4, QUAT, margin=0.015)
check("z spread warning fires", any("not level" in w for w in warns4), f"({len(warns4)} warns)")

skew = np.array([[0.30, -0.24, 0.12], [0.62, -0.24, 0.12], [0.62, 0.24, 0.12], [0.30, 0.30, 0.12]])
_, warns5 = box_from_corners(skew, QUAT, margin=0.015)
check("skew warning fires", any("rectangular" in w for w in warns5), f"({warns5})")

try:
    box_from_corners(c, QUAT, margin=0.5)
    check("oversized margin raises", False)
except ValueError:
    check("oversized margin raises", True)

# --- 5. round trip -------------------------------------------------------
rt = PlayBox.from_dict(box2.to_dict())
check("to_dict/from_dict round trip",
      np.allclose(rt.center, box2.center) and abs(rt.yaw - box2.yaw) < 1e-12
      and np.allclose(rt.half_extents, box2.half_extents))

# --- 6. perimeter --------------------------------------------------------
per = box2.perimeter(10)
check("perimeter closed", np.allclose(per[0], per[-1]))
check("perimeter all contained", all(box2.contains(p[:2], tol=1e-9) for p in per))
check("perimeter at plane_z", np.allclose(per[:, 2], box2.plane_z))

print()
print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
