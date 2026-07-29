"""A stand-in FrankaServer for tests that must not touch hardware.

Speaks the same REQ/REP protocol as frankateach.franka_server.FrankaServer and
models the arm as a first-order system: the EE chases the commanded pose with a
speed limit. Enough to exercise the whole teleop stack except deoxys itself.
"""

import pickle
import threading
import time

import numpy as np
import zmq

from frankateach.constants import arm_ports
from frankateach.messages import FrankaAction, FrankaState


class FakeServer(threading.Thread):
    def __init__(self, arm, max_speed=0.6, hz=50.0, work=0.0):
        super().__init__(daemon=True, name=f"fake-{arm}")
        self.port = arm_ports(arm)[0]
        self.pos = np.array([0.45, 0.0, 0.25])
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.max_step = max_speed / hz
        self.work = work  # seconds of fake osc_move cost per command
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.commands = []
        self.resets = 0

    def run(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://localhost:{self.port}")
        sock.setsockopt(zmq.RCVTIMEO, 200)
        self.ready.set()
        try:
            while not self.stop.is_set():
                try:
                    msg = sock.recv()
                except zmq.Again:
                    continue
                if msg != b"get_state":
                    action: FrankaAction = pickle.loads(msg)
                    if action.reset:
                        self.resets += 1
                        self.pos = np.array([0.45, 0.0, 0.25])
                    else:
                        if self.work:
                            time.sleep(self.work)
                        target = np.array(action.pos, dtype=np.float64)
                        self.commands.append(target)
                        delta = target - self.pos
                        dist = np.linalg.norm(delta)
                        if dist > self.max_step:
                            delta = delta / dist * self.max_step
                        self.pos = self.pos + delta
                sock.send(
                    pickle.dumps(
                        FrankaState(
                            pos=self.pos.astype(np.float32),
                            quat=self.quat.astype(np.float32),
                            gripper=1,
                            timestamp=time.time(),
                        ),
                        protocol=-1,
                    )
                )
        finally:
            sock.close(0)
            ctx.term()
