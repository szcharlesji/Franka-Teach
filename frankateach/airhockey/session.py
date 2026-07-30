"""Per-arm session: which operator currently owns the arm.

Play and calibrate are the same arm driven by different operators over the same
REQ socket, so exactly one may exist at a time -- `FrankaServer` is a synchronous
REQ/REP loop and two clients would interleave requests on it. `ArmSession` is the
thing that guarantees that: it stops and joins the outgoing operator before
constructing the incoming one.

Both operators expose the same `set_intent()` seam, so the web layer does not
care which mode it is talking to.
"""

import threading

import numpy as np

from frankateach.airhockey import config as ahconfig
from frankateach.airhockey.box import provisional_box
from frankateach.airhockey.control import ArmLink
from frankateach.airhockey.operator import ArmOperator, JogOperator

PLAY, CALIBRATE = "play", "calibrate"


def load_or_provisional(arm, cfg):
    """(box, provisional) for one arm.

    Falls back to a box around the arm's current pose when the arm has never been
    calibrated, so the launcher comes up instead of refusing. Reading the pose
    needs a short-lived link, which must be closed before an operator opens its
    own -- see the one-client rule above.
    """
    try:
        return ahconfig.load_box(arm), False
    except RuntimeError:
        link = ArmLink(arm)
        try:
            state = link.get_state()
            box = provisional_box(
                np.asarray(state.pos, dtype=np.float64),
                np.asarray(state.quat, dtype=np.float64),
                half_extents=cfg.get("provisional_half_extents", (0.06, 0.06)),
            )
        finally:
            link.close()
        print(
            f"[{arm}] NOT CALIBRATED -- using a provisional "
            f"{box.half_extents[0] * 200:.0f}x{box.half_extents[1] * 200:.0f} cm box "
            f"around the current pose at z={box.plane_z:.4f}. Calibrate before playing."
        )
        return box, True


class ArmSession:
    """Owns whichever operator currently drives one arm."""

    def __init__(self, arm, cfg, publish=True):
        self.arm = arm
        self.cfg = cfg
        self.publish = publish
        self.mode = None
        self.operator = None
        self.box = None
        self.provisional = False
        self.error = ""
        self._lock = threading.Lock()

    # -- mode -------------------------------------------------------------
    def start(self, mode=PLAY):
        return self.set_mode(mode)

    def set_mode(self, mode, reload_box=False):
        """Switch this arm between play and calibrate. Idempotent."""
        if mode not in (PLAY, CALIBRATE):
            raise ValueError(f"Unknown mode {mode!r}")
        with self._lock:
            if mode == self.mode and not reload_box:
                return True
            self._teardown()

            if mode == PLAY:
                if reload_box or self.box is None:
                    self.box, self.provisional = load_or_provisional(self.arm, self.cfg)
                speed = float(self.cfg["speed"])
                if self.provisional:
                    # An unverified box gets a hard speed cap regardless of config.
                    speed = min(speed, 0.1)
                op = ArmOperator(
                    self.arm,
                    self.box,
                    self.cfg,
                    publish=self.publish,
                    speed=speed,
                    # A provisional box is built around the pose the arm is in, so
                    # resetting and gliding would move an arm into a play area
                    # nobody has verified. Hold still and wait to be calibrated.
                    reset=not self.provisional,
                )
            else:
                # Never joint-reset into calibration: the arm is already at a
                # height someone chose, and the ready pose can sit in the table.
                op = JogOperator(self.arm, self.cfg, reset=False)

            op.start()
            if not op.wait_ready(timeout=90):
                self.error = f"{self.arm} {mode} operator did not come up"
                op.stop()
                op.join(timeout=5)
                return False
            status = op.get_status()
            if not status.connected:
                self.error = status.error or f"{self.arm} arm not connected"
                op.stop()
                op.join(timeout=5)
                return False

            self.operator = op
            self.mode = mode
            self.error = ""
            return True

    def _teardown(self):
        if self.operator is not None:
            self.operator.stop()
            # Must join: the outgoing operator still holds the arm's REQ socket,
            # and a second client on it would interleave requests.
            self.operator.join(timeout=10)
            self.operator = None

    def stop(self):
        with self._lock:
            self._teardown()
            self.mode = None

    # -- the seam the web layer uses ---------------------------------------
    def set_intent(self, *args, **kwargs):
        op = self.operator
        if op is not None:
            op.set_intent(*args, **kwargs)

    def get_status(self):
        op = self.operator
        return None if op is None else op.get_status()

    def reload_box(self):
        """Re-read this arm's calibration and restart play with it."""
        return self.set_mode(PLAY, reload_box=True)
