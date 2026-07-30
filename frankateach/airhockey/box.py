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
    plane_z: float  # EE height in 'constant' mode; centre height in 'tilted'
    quat: np.ndarray  # (4,) fixed EE orientation
    corners: np.ndarray = field(default=None)  # (4,3) as taught, for reference
    z_spread: float = 0.0  # max-min of the taught corner heights
    plane_mode: str = "constant"  # 'constant' | 'tilted'
    # (3,) [a, bx, by] with z = a + bx*px + by*py in BOX-frame xy. Least-squares
    # fit through the four taught corners; only consulted when tilted.
    plane_coeffs: np.ndarray = field(default=None)
    plane_residual: float = 0.0  # max |corner z - fitted plane|
    # True for a box synthesised around the arm's current pose because this arm
    # has never been calibrated. Never written to the config; the UI shows it as a
    # warning and the launcher caps speed until it is replaced by a real one.
    provisional: bool = False

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64).reshape(2)
        self.half_extents = np.asarray(self.half_extents, dtype=np.float64).reshape(2)
        self.quat = np.asarray(self.quat, dtype=np.float64).reshape(4)
        self.yaw = float(self.yaw)
        self.plane_z = float(self.plane_z)
        if self.plane_mode not in ("constant", "tilted"):
            raise ValueError(
                f"Unknown plane_mode {self.plane_mode!r}, expected 'constant' or 'tilted'"
            )
        if self.plane_coeffs is None:
            self.plane_coeffs = np.array([self.plane_z, 0.0, 0.0])
        else:
            self.plane_coeffs = np.asarray(self.plane_coeffs, dtype=np.float64).reshape(3)
        if self.plane_mode == "tilted" and self.corners is None:
            raise ValueError(
                "plane_mode='tilted' needs the taught corners the plane was fitted "
                "to; recalibrate this arm."
            )
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

    def z_at(self, xy):
        """EE height for a base-frame xy.

        'constant' returns plane_z everywhere -- correct only if the table is level
        in this arm's base frame. 'tilted' evaluates the plane fitted through the
        taught corners, which is what stops the mallet scraping at one corner and
        lifting at the opposite one.

        Accepts (2,) or (N,2); returns a float or (N,).
        """
        pts = np.asarray(xy, dtype=np.float64)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)[:, :2]
        if self.plane_mode == "constant":
            z = np.full(len(pts), self.plane_z)
        else:
            local = np.array([self.to_box(p) for p in pts])
            a, bx, by = self.plane_coeffs
            z = a + bx * local[:, 0] + by * local[:, 1]
        return float(z[0]) if single else z

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
        return np.array([self.center[0], self.center[1], self.z_at(self.center)])

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
                path.append([xy[0], xy[1], self.z_at(xy)])
        path.append([rc[0][0], rc[0][1], self.z_at(rc[0])])
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
            "plane_mode": str(self.plane_mode),
            "plane_coeffs": [float(v) for v in self.plane_coeffs],
            "plane_residual": float(self.plane_residual),
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
            plane_mode=str(d.get("plane_mode", "constant")),
            plane_coeffs=np.array(d["plane_coeffs"], dtype=np.float64)
            if d.get("plane_coeffs") is not None
            else None,
            plane_residual=float(d.get("plane_residual", 0.0)),
        )


def _quat2mat(q):
    """(x,y,z,w) -> 3x3. deoxys' transform_utils.mat2quat returns xyzw order."""
    x, y, z, w = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def level_quat(quat):
    """Snap an EE orientation to exactly tool-down, keeping its yaw.

    The orientation held during play is whatever the arm happened to be in when
    the corners were taught, so the wrist ends up a couple of degrees off vertical
    and a flat mallet face does not sit flat on the table. This keeps the rotation
    about the vertical axis and makes the tool axis exactly -z.

    Returns (x,y,z,w).
    """
    R = _quat2mat(quat)
    # Yaw of the tool's own x axis, projected onto the horizontal plane.
    ex = R[:, 0]
    yaw = float(np.arctan2(ex[1], ex[0]))
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    Rx180 = np.diag([1.0, -1.0, -1.0])  # tool z -> -z
    Rl = Rz @ Rx180
    # Rotation matrix -> (x,y,z,w), via the largest-component branch for stability.
    tr = np.trace(Rl)
    if tr > 0:
        sq = np.sqrt(1.0 + tr) * 2
        w = 0.25 * sq
        x = (Rl[2, 1] - Rl[1, 2]) / sq
        y = (Rl[0, 2] - Rl[2, 0]) / sq
        z = (Rl[1, 0] - Rl[0, 1]) / sq
    elif Rl[0, 0] > Rl[1, 1] and Rl[0, 0] > Rl[2, 2]:
        sq = np.sqrt(1.0 + Rl[0, 0] - Rl[1, 1] - Rl[2, 2]) * 2
        w = (Rl[2, 1] - Rl[1, 2]) / sq
        x = 0.25 * sq
        y = (Rl[0, 1] + Rl[1, 0]) / sq
        z = (Rl[0, 2] + Rl[2, 0]) / sq
    elif Rl[1, 1] > Rl[2, 2]:
        sq = np.sqrt(1.0 + Rl[1, 1] - Rl[0, 0] - Rl[2, 2]) * 2
        w = (Rl[0, 2] - Rl[2, 0]) / sq
        x = (Rl[0, 1] + Rl[1, 0]) / sq
        y = 0.25 * sq
        z = (Rl[1, 2] + Rl[2, 1]) / sq
    else:
        sq = np.sqrt(1.0 + Rl[2, 2] - Rl[0, 0] - Rl[1, 1]) * 2
        w = (Rl[1, 0] - Rl[0, 1]) / sq
        x = (Rl[0, 2] + Rl[2, 0]) / sq
        y = (Rl[1, 2] + Rl[2, 1]) / sq
        z = 0.25 * sq
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def tilt_from_vertical_deg(quat):
    """Angle between the tool axis and straight down, in degrees."""
    tool = _quat2mat(quat) @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(-tool[2], -1.0, 1.0))))


def provisional_box(pos, quat, half_extents=(0.06, 0.06)):
    """A tiny play box around the arm's *current* pose, for an uncalibrated arm.

    Deliberately derived from measured state rather than invented numbers: a
    fabricated centre and plane_z would drive a real arm to coordinates nobody
    checked, which is the failure mode configs/airhockey.yaml exists to prevent.
    Anchoring on the current pose means launching cannot move the arm, and the
    worst case is a small box at the height someone already placed it at.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    return PlayBox(
        center=pos[:2].copy(),
        yaw=0.0,
        half_extents=np.asarray(half_extents, dtype=np.float64),
        plane_z=float(pos[2]),
        quat=np.asarray(quat, dtype=np.float64),
        corners=None,
        plane_mode="constant",
        provisional=True,
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


def fit_plane(corners, center, yaw):
    """Least-squares plane through the taught corners, in BOX-frame xy.

    Returns ([a, bx, by], residual) for z = a + bx*px + by*py. The residual is
    the largest deviation of any corner from the fitted plane: a tilted *plane*
    cannot represent a twisted (non-coplanar) table, so a big residual means the
    surface is warped and neither plane mode will sit flat on it.
    """
    corners = np.asarray(corners, dtype=np.float64)
    local = np.array([_rot(-yaw) @ (c[:2] - center) for c in corners])
    A = np.column_stack([np.ones(len(local)), local[:, 0], local[:, 1]])
    coeffs, *_ = np.linalg.lstsq(A, corners[:, 2], rcond=None)
    residual = float(np.max(np.abs(A @ coeffs - corners[:, 2])))
    return coeffs, residual


def box_from_corners(
    corners,
    quat,
    margin=0.015,
    z_spread_warn=0.005,
    plane_mode="constant",
    level_wrist=False,
):
    """Build a conservative PlayBox from 4 taught corners.

    corners: (4,3) base-frame positions, in order around the perimeter.
    margin:  metres to shrink in from the inscribed bound on every side.
    plane_mode: 'constant' pins one height for the whole box; 'tilted' follows the
        plane fitted through the corners' own z values.

    Returns (box, warnings) where warnings is a list of human-readable strings.
    """
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 3):
        raise ValueError(f"Expected 4 corners of shape (4,3), got {corners.shape}")

    warnings = []
    center = corners[:, :2].mean(axis=0)
    yaw = fit_yaw(corners)

    tilt = tilt_from_vertical_deg(quat)
    if level_wrist:
        quat = level_quat(quat)
        warnings.append(
            f"Wrist levelled: the taught orientation was {tilt:.2f}° off vertical, "
            "snapped to exactly tool-down (yaw preserved)."
        )
    elif tilt > 1.0:
        warnings.append(
            f"Wrist is {tilt:.2f}° off vertical, so a flat mallet face will not sit "
            "flat on the table. Set level_wrist: true to snap it."
        )

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
    plane_coeffs, plane_residual = fit_plane(corners, center, yaw)

    if z_spread > z_spread_warn and plane_mode == "constant":
        warnings.append(
            f"Corner heights span {z_spread * 1000:.1f} mm, above the "
            f"{z_spread_warn * 1000:.0f} mm threshold. The table is not level in "
            "this arm's base frame -- a single plane_z will scrape at one corner "
            "and lift at another. Set plane_mode: tilted, re-teach more carefully, "
            "or shim the table."
        )
    if plane_mode == "tilted":
        tilt = float(np.linalg.norm(plane_coeffs[1:]))
        warnings.append(
            f"Tilted plane: {tilt * 1000:.1f} mm drop per metre, corners fit to "
            f"within {plane_residual * 1000:.1f} mm."
        )
        if plane_residual > z_spread_warn:
            warnings.append(
                f"Corners are {plane_residual * 1000:.1f} mm off any single plane, "
                "so the surface is twisted rather than merely tilted. A tilted "
                "plane cannot follow that -- re-teach, or shim the table."
            )

    box = PlayBox(
        center=center,
        yaw=yaw,
        half_extents=half_extents,
        plane_z=float(corners[:, 2].mean()),
        quat=np.asarray(quat, dtype=np.float64),
        corners=corners,
        z_spread=z_spread,
        plane_mode=plane_mode,
        plane_coeffs=plane_coeffs,
        plane_residual=plane_residual,
    )
    return box, warnings
