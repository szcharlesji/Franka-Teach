"""Oriented rectangular play area derived from four taught corners.

The four corners you jog to during calibration define a quadrilateral. We do NOT
use that quad directly -- we fit an oriented rectangle to it and then shrink to
the *inscribed* bound, so the playable area is guaranteed to sit inside every
corner you taught. That is what keeps the mallet off the table edge even if your
four corners are not perfectly rectangular.
"""

from dataclasses import dataclass, field

import numpy as np


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


@dataclass
class PlayBox:
    """An oriented rectangle on a level plane, in one arm's base frame."""

    center: np.ndarray  # (2,) xy of the rectangle centre
    yaw: float  # rotation of the box x-axis from base x, radians
    half_extents: np.ndarray  # (2,) half width along box x and box y
    plane_z: float  # fixed EE height
    quat: np.ndarray  # (4,) fixed EE orientation
    corners: np.ndarray = field(default=None)  # (4,3) as taught, for reference
    z_spread: float = 0.0  # max-min of the taught corner heights

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64).reshape(2)
        self.half_extents = np.asarray(self.half_extents, dtype=np.float64).reshape(2)
        self.quat = np.asarray(self.quat, dtype=np.float64).reshape(4)
        self.yaw = float(self.yaw)
        self.plane_z = float(self.plane_z)
        if np.any(self.half_extents <= 0):
            raise ValueError(
                f"Degenerate play box: half_extents={self.half_extents}. The margin "
                "is probably larger than the area you taught."
            )

    # -- frame conversions -------------------------------------------------
    def to_box(self, xy):
        """Base-frame xy -> box-frame xy."""
        return _rot(-self.yaw) @ (np.asarray(xy, dtype=np.float64)[:2] - self.center)

    def to_world(self, xy_box):
        """Box-frame xy -> base-frame xy."""
        return _rot(self.yaw) @ np.asarray(xy_box, dtype=np.float64)[:2] + self.center

    # -- the thing the control loop calls ----------------------------------
    def clip(self, xy):
        """Clamp a base-frame xy into the rectangle."""
        p = self.to_box(xy)
        p = np.clip(p, -self.half_extents, self.half_extents)
        return self.to_world(p)

    def contains(self, xy, tol=1e-9):
        p = np.abs(self.to_box(xy))
        return bool(np.all(p <= self.half_extents + tol))

    def rotate_intent(self, vx, vy):
        """Map key intent expressed in box axes into base-frame velocity.

        Keys should move the mallet along the *table*, not along the robot base
        axes, so W is 'up the table' whatever the yaw came out as.
        """
        return _rot(self.yaw) @ np.array([vx, vy], dtype=np.float64)

    # -- helpers -----------------------------------------------------------
    @property
    def home(self):
        """Full 3D home pose position: the box centre at the play height."""
        return np.array([self.center[0], self.center[1], self.plane_z])

    def rect_corners(self):
        """(4,2) base-frame corners of the *fitted* rectangle, in order."""
        hx, hy = self.half_extents
        local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
        return np.array([self.to_world(p) for p in local])

    def perimeter(self, points_per_edge=25):
        """(N,3) path tracing the rectangle edges, for the --verify pass."""
        rc = self.rect_corners()
        path = []
        for i in range(4):
            a, b = rc[i], rc[(i + 1) % 4]
            for t in np.linspace(0, 1, points_per_edge, endpoint=False):
                xy = a + (b - a) * t
                path.append([xy[0], xy[1], self.plane_z])
        path.append([rc[0][0], rc[0][1], self.plane_z])
        return np.array(path)

    # -- serialisation -----------------------------------------------------
    def to_dict(self):
        d = {
            "center": [float(v) for v in self.center],
            "yaw": float(self.yaw),
            "half_extents": [float(v) for v in self.half_extents],
            "plane_z": float(self.plane_z),
            "fixed_quat": [float(v) for v in self.quat],
            "z_spread": float(self.z_spread),
        }
        if self.corners is not None:
            d["corners"] = [[float(v) for v in c] for c in np.asarray(self.corners)]
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(
            center=np.array(d["center"], dtype=np.float64),
            yaw=float(d["yaw"]),
            half_extents=np.array(d["half_extents"], dtype=np.float64),
            plane_z=float(d["plane_z"]),
            quat=np.array(d["fixed_quat"], dtype=np.float64),
            corners=np.array(d["corners"], dtype=np.float64)
            if d.get("corners") is not None
            else None,
            z_spread=float(d.get("z_spread", 0.0)),
        )


def fit_yaw(corners_xy):
    """Best-fit rotation of a rectangle through 4 ordered corners.

    Each edge of a rectangle rotated by theta lies at theta + k*90 degrees, so we
    average the edge angles modulo 90 degrees via the standard 4*angle trick.
    Returns a yaw in [-pi/4, pi/4], which keeps the box x-axis within 45 degrees
    of the robot's base x-axis.
    """
    corners_xy = np.asarray(corners_xy, dtype=np.float64)[:, :2]
    angles = []
    for i in range(4):
        e = corners_xy[(i + 1) % 4] - corners_xy[i]
        if np.linalg.norm(e) < 1e-6:
            raise ValueError("Two taught corners are identical; recalibrate.")
        angles.append(np.arctan2(e[1], e[0]))
    angles = np.array(angles)
    return float(np.arctan2(np.sin(4 * angles).mean(), np.cos(4 * angles).mean()) / 4.0)


def box_from_corners(corners, quat, margin=0.015, z_spread_warn=0.005):
    """Build a conservative PlayBox from 4 taught corners.

    corners: (4,3) base-frame positions, in order around the perimeter.
    margin:  metres to shrink in from the inscribed bound on every side.

    Returns (box, warnings) where warnings is a list of human-readable strings.
    """
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 3):
        raise ValueError(f"Expected 4 corners of shape (4,3), got {corners.shape}")

    warnings = []
    center = corners[:, :2].mean(axis=0)
    yaw = fit_yaw(corners)

    # Inscribed half-extents: the nearest corner along each box axis bounds the
    # rectangle, so no edge of the result can pass outside the taught quad.
    local = np.array([_rot(-yaw) @ (c[:2] - center) for c in corners])
    hx = float(np.abs(local[:, 0]).min())
    hy = float(np.abs(local[:, 1]).min())

    fitted = np.array([np.abs(local[:, 0]).max(), np.abs(local[:, 1]).max()])
    skew = np.array([fitted[0] - hx, fitted[1] - hy])
    if np.any(skew > 0.02):
        warnings.append(
            f"Taught corners are not very rectangular (inscribed bound is "
            f"{skew[0] * 1000:.0f} x {skew[1] * 1000:.0f} mm inside the fitted "
            f"rectangle). The play area was shrunk to stay inside them."
        )

    half_extents = np.array([hx, hy]) - margin
    if np.any(half_extents <= 0):
        raise ValueError(
            f"Margin {margin * 1000:.0f} mm leaves nothing to play in "
            f"(inscribed half-extents were {hx * 1000:.0f} x {hy * 1000:.0f} mm). "
            "Teach a larger area or lower `margin`."
        )

    z_spread = float(corners[:, 2].max() - corners[:, 2].min())
    if z_spread > z_spread_warn:
        warnings.append(
            f"Corner heights span {z_spread * 1000:.1f} mm, above the "
            f"{z_spread_warn * 1000:.0f} mm threshold. The table is not level in "
            "this arm's base frame -- a single plane_z will scrape at one corner "
            "and lift at another. Re-teach more carefully, or shim the table."
        )

    box = PlayBox(
        center=center,
        yaw=yaw,
        half_extents=half_extents,
        plane_z=float(corners[:, 2].mean()),
        quat=np.asarray(quat, dtype=np.float64),
        corners=corners,
        z_spread=z_spread,
    )
    return box, warnings
