"""Lifecycle management for the Discovery-to-NUC SSH forward."""

import os
import signal
import socket
import subprocess
import time


class SSHTunnel:
    def __init__(self, host="franka", local_port=18765, remote_port=8765):
        self.host = host
        self.local_port = int(local_port)
        self.remote_port = int(remote_port)
        self.process = None

    @property
    def alive(self):
        return self.process is not None and self.process.poll() is None

    def start(self, timeout=10.0):
        if self.alive:
            return self
        command = [
            "ssh",
            "-NT",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=3",
            "-o",
            "ServerAliveCountMax=2",
            "-L",
            f"127.0.0.1:{self.local_port}:127.0.0.1:{self.remote_port}",
            self.host,
        ]
        self.process = subprocess.Popen(command, start_new_session=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"SSH tunnel exited with code {self.process.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", self.local_port), timeout=0.2):
                    return self
            except OSError:
                time.sleep(0.1)
        self.stop()
        raise RuntimeError("SSH tunnel did not expose the robot bridge in time")

    def stop(self, timeout=3.0):
        if not self.alive:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=timeout)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if self.process.poll() is None:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=1)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()
