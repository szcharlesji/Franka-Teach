"""pygame rendering: realsense feed as background, per-arm HUD on top.

The camera runs on its own thread with a CONFLATE'd subscriber, so a slow or
absent camera never touches control timing -- the HUD just keeps showing the
last frame (or none at all).
"""

import threading

import numpy as np
import pygame

from frankateach.constants import CAM_PORT, HOST
from frankateach.network import ZMQCameraSubscriber

BG = (18, 18, 22)
FG = (225, 225, 232)
DIM = (120, 120, 132)
ACCENT = (120, 200, 255)
GOOD = (120, 225, 150)
WARN = (255, 200, 90)
BAD = (255, 110, 110)


class CameraFeed(threading.Thread):
    """Background thread pulling the newest RGB frame."""

    def __init__(self, cam_id):
        super().__init__(daemon=True, name=f"cam-{cam_id}")
        self.cam_id = cam_id
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.error = ""

    def run(self):
        try:
            sub = ZMQCameraSubscriber(HOST, CAM_PORT + self.cam_id, "RGB")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return
        try:
            while not self._stop.is_set():
                try:
                    image, _ = sub.recv_rgb_image()
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    break
                # BGR -> RGB, and transpose into pygame's (w, h) surface order.
                with self._lock:
                    self._frame = np.transpose(image[:, :, ::-1], (1, 0, 2)).copy()
        finally:
            try:
                sub.stop()
            except Exception:
                pass

    def latest(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._stop.set()


def _blit_camera(screen, frame):
    surf = pygame.surfarray.make_surface(frame)
    sw, sh = screen.get_size()
    fw, fh = surf.get_size()
    scale = min(sw / fw, sh / fh)
    surf = pygame.transform.smoothscale(surf, (int(fw * scale), int(fh * scale)))
    screen.blit(surf, ((sw - surf.get_width()) // 2, (sh - surf.get_height()) // 2))


def _draw_box_panel(screen, font, small, rect, arm, box, status, keyname):
    panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    panel.fill((10, 10, 14, 190))
    screen.blit(panel, rect.topleft)
    pygame.draw.rect(screen, (60, 60, 72), rect, 1)

    label = f"{arm.upper()}  [{keyname}]"
    screen.blit(font.render(label, True, ACCENT), (rect.x + 12, rect.y + 8))

    if not status.connected:
        screen.blit(small.render(status.error or "not connected", True, BAD),
                    (rect.x + 12, rect.y + 34))
        return

    # Top-down rectangle. Box +x (up the table) is drawn upward, +y (left) leftward.
    pad = 14
    top = rect.y + 32
    area = pygame.Rect(rect.x + pad, top, rect.width - 2 * pad, rect.height - top + rect.y - 58)
    hx, hy = box.half_extents
    aspect = (2 * hy) / (2 * hx)  # width (y) over height (x)
    if area.width / area.height > aspect:
        bw = area.height * aspect
        bh = area.height
    else:
        bw = area.width
        bh = area.width / aspect
    bx = area.x + (area.width - bw) / 2
    by = area.y + (area.height - bh) / 2
    border = pygame.Rect(int(bx), int(by), int(bw), int(bh))
    pygame.draw.rect(screen, DIM, border, 1)

    # Box frame -> screen: +x up, +y left.
    px = bx + bw * (0.5 - status.box_pos[1] / (2 * hy))
    py = by + bh * (0.5 - status.box_pos[0] / (2 * hx))
    colour = WARN if status.stale else GOOD
    pygame.draw.circle(screen, colour, (int(px), int(py)), 7)
    pygame.draw.circle(screen, (0, 0, 0), (int(px), int(py)), 7, 1)

    foot = rect.bottom - 44
    screen.blit(small.render(f"x {status.pos[0]:+.3f}  y {status.pos[1]:+.3f}", True, FG),
                (rect.x + 12, foot))
    rate_col = GOOD if status.rate >= 45 else (WARN if status.rate >= 30 else BAD)
    if status.homing:
        tag, tag_col = "HOMING", ACCENT
    elif status.stale:
        tag, tag_col = "FROZEN", WARN
    else:
        tag, tag_col = "LIVE", GOOD
    screen.blit(small.render(f"{status.speed:.2f} m/s", True, FG), (rect.x + 12, foot + 18))
    screen.blit(small.render(f"{status.rate:5.1f} Hz", True, rate_col), (rect.right - 150, foot + 18))
    screen.blit(small.render(tag, True, tag_col), (rect.right - 70, foot))


def draw(screen, font, small, operators, boxes, keynames, camera, banner=""):
    screen.fill(BG)

    if camera is not None:
        frame = camera.latest()
        if frame is not None:
            _blit_camera(screen, frame)
        else:
            note = camera.error or f"waiting for camera {camera.cam_id}..."
            screen.blit(small.render(note, True, DIM), (16, 12))

    sw, sh = screen.get_size()
    pw, ph = 300, 240
    slots = {"left": pygame.Rect(20, sh - ph - 46, pw, ph),
             "right": pygame.Rect(sw - pw - 20, sh - ph - 46, pw, ph)}
    for arm, op in operators.items():
        _draw_box_panel(screen, font, small, slots[arm], arm, boxes[arm],
                        op.get_status(), keynames[arm])

    hint = "H home    SPACE freeze    ESC release    Q quit"
    screen.blit(small.render(hint, True, DIM), (20, sh - 30))
    if banner:
        screen.blit(font.render(banner, True, WARN), (20, 12))
    pygame.display.flip()
