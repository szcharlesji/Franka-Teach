"""Crash-conscious raw session and episode storage on Discovery."""

import hashlib
import json
import os
import re
import shutil
import socket
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from frankateach.recording.protocol import SCHEMA_VERSION, jsonable

SESSION_RE = re.compile(r"[^A-Za-z0-9._-]+")
NUC_HOSTNAME = "robotlab-NUC8i7BEH"


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_label(value):
    value = SESSION_RE.sub("_", str(value).strip()).strip("._-")
    if not value:
        raise ValueError("session label is empty after sanitization")
    return value[:80]


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class NDJSONWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def write(self, value):
        encoded = json.dumps(jsonable(value), separators=(",", ":"), allow_nan=False)
        with self._lock:
            self._stream.write(encoded + "\n")

    def flush(self, durable=False):
        with self._lock:
            self._stream.flush()
            if durable:
                os.fsync(self._stream.fileno())

    def close(self):
        with self._lock:
            if not self._stream.closed:
                self._stream.flush()
                os.fsync(self._stream.fileno())
                self._stream.close()


class EpisodeWriter:
    def __init__(self, store, episode_id):
        self.store = store
        self.episode_id = safe_label(episode_id)
        self.path = store.partial_dir / self.episode_id
        self.path.mkdir(parents=False, exist_ok=False)
        self.robot = NDJSONWriter(self.path / "robot.ndjson")
        self.keys = NDJSONWriter(self.path / "keys.ndjson")
        self.closed = False

    def write_robot(self, value):
        self.robot.write(value)

    def write_keys(self, value):
        self.keys.write(value)

    def write_json(self, name, value):
        write_json_atomic(self.path / name, value)

    def close_streams(self):
        if not self.closed:
            self.robot.close()
            self.keys.close()
            self.closed = True

    def finalize(self, accepted):
        self.close_streams()
        destination_root = self.store.raw_dir if accepted else self.store.rejected_dir
        destination = destination_root / self.episode_id
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(self.path, destination)
        return destination

    def discard(self):
        self.close_streams()
        if self.path.exists():
            shutil.rmtree(self.path)


class SessionStore:
    def __init__(
        self,
        storage_root,
        session_label,
        *,
        enforce_discovery=False,
        hostname=None,
        started_at=None,
    ):
        home_data = (Path.home() / "data").resolve()
        storage_root = Path(storage_root).expanduser().resolve()
        hostname = hostname or socket.gethostname()
        if enforce_discovery and hostname == NUC_HOSTNAME:
            raise RuntimeError("Discovery recorder refuses to run on the robot NUC")
        if (
            enforce_discovery
            and storage_root != home_data
            and home_data not in storage_root.parents
        ):
            raise RuntimeError(f"storage root must be inside {home_data}, got {storage_root}")

        started = started_at or datetime.now(timezone.utc)
        stamp = started.strftime("%Y%m%dT%H%M%SZ")
        self.label = safe_label(session_label)
        self.storage_root = storage_root
        self.air_hockey_root = storage_root / "air_hockey"
        self.path = self.air_hockey_root / f"{self.label}_{stamp}"
        self.raw_dir = self.path / "raw"
        self.rejected_dir = self.path / "rejected"
        self.partial_dir = self.path / ".partial"
        for directory in (self.raw_dir, self.rejected_dir, self.partial_dir):
            directory.mkdir(parents=True, exist_ok=False)
        self.events = NDJSONWriter(self.path / "session_events.ndjson")
        write_json_atomic(
            self.path / "session.json",
            {
                "schema_version": SCHEMA_VERSION,
                "session": self.label,
                "created_at": utc_now(),
                "hostname": hostname,
                "storage_root": str(storage_root),
            },
        )

    def new_episode(self, sequence):
        return EpisodeWriter(self, f"episode_{int(sequence):06d}")

    def audit(self, event, **fields):
        self.events.write({"timestamp": utc_now(), "event": event, **fields})

    def close(self):
        self.events.close()

    def scan_orphan_partials(self):
        found = []
        pattern = f"{self.label}_*/.partial/*"
        for path in sorted(self.air_hockey_root.glob(pattern)):
            if not path.is_dir():
                continue
            start_path = path / "camera_start.json"
            camera_start = None
            if start_path.is_file():
                try:
                    camera_start = json.loads(start_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
            found.append(
                {
                    "path": str(path),
                    "recording_id": (camera_start or {}).get("id"),
                    "camera_start": camera_start,
                }
            )
        return found


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(directory, names):
    directory = Path(directory)
    lines = []
    for name in names:
        path = directory / name
        if path.exists() and path.is_file():
            lines.append(f"{sha256_file(path)}  {name}")
    target = directory / "checksums.sha256"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
