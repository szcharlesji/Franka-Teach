"""Cross-user advisory ownership for the shared robot NUC."""

import fcntl
import os
from pathlib import Path


class ArmOwnership:
    def __init__(self, arms, lock_root="/tmp"):
        self.arms = tuple(arms)
        self.lock_root = Path(lock_root)
        self._files = []

    def acquire(self):
        try:
            for arm in self.arms:
                path = self.lock_root / f"frankateach-airhockey-{arm}.lock"
                try:
                    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
                    writable = True
                except PermissionError:
                    fd = os.open(path, os.O_RDONLY)
                    writable = False
                if writable:
                    try:
                        os.chmod(path, 0o666)
                    except PermissionError:
                        pass
                stream = os.fdopen(fd, "r+" if writable else "r")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    stream.close()
                    raise RuntimeError(
                        f"{arm} arm is already owned by another Franka-Teach session"
                    ) from exc
                if stream.writable():
                    stream.seek(0)
                    stream.truncate()
                    stream.write(f"pid={os.getpid()}\n")
                    stream.flush()
                self._files.append(stream)
        except Exception:
            self.release()
            raise
        return self

    def release(self):
        for stream in self._files:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()
        self._files.clear()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()
