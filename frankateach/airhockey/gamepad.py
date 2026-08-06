"""Linux joystick reader for a gamepad physically attached to the robot NUC."""

import glob
import os
import select
import struct
import threading
import time

import numpy as np

JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
EVENT = struct.Struct("IhBB")


class LocalGamepad(threading.Thread):
    """Read Linux's stable /dev/input/js* interface without extra packages."""

    def __init__(self, deadzone=0.12):
        super().__init__(daemon=True, name="usb-gamepad")
        self.deadzone = float(deadzone)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._axes = np.zeros(8, dtype=np.float64)
        self.connected = False
        self.path = ""
        self.error = ""

    def stop(self):
        self._stop_event.set()

    def _find(self):
        # by-id excludes keyboard/mouse js compatibility devices and stays stable
        # if the kernel assigns a different js number after reconnecting.
        paths = sorted(
            path
            for path in glob.glob("/dev/input/by-id/*-joystick")
            if not path.endswith("-event-joystick")
        )
        if paths:
            return os.path.realpath(paths[0])
        paths = sorted(glob.glob("/dev/input/js*"))
        return paths[-1] if paths else None

    def _stick(self, x_axis, y_axis):
        with self._lock:
            x, y = self._axes[x_axis], self._axes[y_axis]
        magnitude = min(1.0, float(np.hypot(x, y)))
        if magnitude <= self.deadzone:
            return np.zeros(2)
        scaled = (magnitude - self.deadzone) / (1.0 - self.deadzone)
        # Linux axes are [screen-right, screen-down]. Operator intent is
        # [table-forward, table-left], so both signs are inverted.
        return np.array([-y, -x]) / magnitude * scaled

    def intents(self):
        if not self.connected:
            return {}
        # Xbox 360 Linux joystick mapping: LX=0, LY=1, RX=3, RY=4.
        return {"left": self._stick(0, 1), "right": self._stick(3, 4)}

    def status(self):
        intents = self.intents()
        return {
            "connected": bool(self.connected),
            "active": any(np.linalg.norm(stick) > 1e-6 for stick in intents.values()),
            "path": self.path,
            "error": self.error,
        }

    def run(self):
        while not self._stop_event.is_set():
            path = self._find()
            if path is None:
                self.error = "no joystick found"
                time.sleep(1.0)
                continue
            fd = None
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                self.path = path
                self.connected = True
                self.error = ""
                print(f"[gamepad] connected on {path}")
                while not self._stop_event.is_set():
                    readable, _, _ = select.select([fd], [], [], 0.2)
                    if not readable:
                        continue
                    data = os.read(fd, EVENT.size * 32)
                    if not data:
                        raise OSError("joystick disconnected")
                    for offset in range(0, len(data) - EVENT.size + 1, EVENT.size):
                        _, value, event_type, number = EVENT.unpack_from(data, offset)
                        if (event_type & ~JS_EVENT_INIT) == JS_EVENT_AXIS:
                            if number < len(self._axes):
                                with self._lock:
                                    self._axes[number] = np.clip(value / 32767.0, -1, 1)
            except (OSError, ValueError) as exc:
                self.error = str(exc)
            finally:
                self.connected = False
                self.path = ""
                with self._lock:
                    self._axes.fill(0)
                if fd is not None:
                    os.close(fd)
            if not self._stop_event.is_set():
                time.sleep(0.5)
