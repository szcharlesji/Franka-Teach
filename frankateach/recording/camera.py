"""Adapter around the standalone Camera-API Python client on Discovery."""

import importlib
import inspect
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

from frankateach.recording.clock import camera_sample


class CameraAPIAdapter:
    """Normalize Camera-API client revisions to the HTTP contract in its docs."""

    def __init__(self, camera):
        self.camera = camera
        self._event_condition = threading.Condition()
        self._event_records = deque(maxlen=10000)
        self._event_sequence = 0
        self._event_error = None
        self._event_thread = None
        self._event_stop = threading.Event()
        self._closed = False

    @classmethod
    def from_repo(cls, repo_root, usbmux=True):
        client_dir = Path(repo_root).expanduser().resolve() / "client"
        module_path = client_dir / "camera_api.py"
        if not module_path.is_file():
            raise RuntimeError(f"CameraAPI client not found at {module_path}")
        if str(client_dir) not in sys.path:
            sys.path.insert(0, str(client_dir))
        module = importlib.import_module("camera_api")
        return cls(module.CameraAPI(usbmux=usbmux))

    @staticmethod
    def _unwrap(value):
        if isinstance(value, (bytes, bytearray)):
            return json.loads(value.decode("utf-8"))
        if isinstance(value, str):
            return json.loads(value)
        if hasattr(value, "json") and callable(value.json):
            return value.json()
        if isinstance(value, tuple):
            for part in reversed(value):
                if isinstance(part, (dict, list)):
                    return part
                if isinstance(part, (str, bytes, bytearray)):
                    try:
                        return CameraAPIAdapter._unwrap(part)
                    except Exception:
                        pass
        return value

    def _public(self, names, *args, **kwargs):
        for name in names:
            method = getattr(self.camera, name, None)
            if callable(method):
                return self._unwrap(method(*args, **kwargs))
        raise AttributeError(f"CameraAPI client lacks {', '.join(names)}")

    def _request(self, method, path, body=None, timeout=None):
        """Use the client's transport so usbmux remains an implementation detail."""
        request = getattr(self.camera, "request", None) or getattr(
            self.camera, "_request", None
        )
        if not callable(request):
            raise RuntimeError(
                f"CameraAPI client cannot call {method} {path}: "
                "no public method or request transport"
            )
        signature = inspect.signature(request)
        names = set(signature.parameters)
        kwargs = {}
        if "body" in names:
            kwargs["body"] = body
        elif "json_body" in names:
            kwargs["json_body"] = body
        elif "json" in names:
            kwargs["json"] = body
        elif "data" in names:
            kwargs["data"] = body
        if timeout is not None and "timeout" in names:
            kwargs["timeout"] = timeout
        parameters = list(signature.parameters)
        path_first = bool(parameters) and parameters[0] in {
            "path",
            "endpoint",
            "url",
        }
        try:
            if path_first:
                return self._unwrap(request(path, method=method, **kwargs))
            return self._unwrap(request(method, path, **kwargs))
        except TypeError:
            if body is None:
                return self._unwrap(request(path, method) if path_first else request(method, path))
            return self._unwrap(
                request(path, method, body) if path_first else request(method, path, body)
            )

    def wait_until_ready(self):
        method = getattr(self.camera, "wait_until_ready", None)
        return method() if callable(method) else self.status()

    def status(self):
        try:
            return self._public(("status", "get_status"))
        except AttributeError:
            return self._request("GET", "/status")

    def configure(self, config):
        method = getattr(self.camera, "configure", None)
        if callable(method):
            parameters = set(inspect.signature(method).parameters)
            keyframe_name = (
                "key_frame_interval"
                if "key_frame_interval" in parameters
                else "keyframe_interval"
            )
            snake = {
                "keyFrameInterval": keyframe_name,
                "rotationDegrees": "rotation_degrees",
                "formatIndex": "format_index",
                "allowFrameReordering": "allow_frame_reordering",
            }
            kwargs = {snake.get(key, key): value for key, value in config.items()}
            try:
                return self._unwrap(method(**kwargs))
            except TypeError:
                return self._unwrap(method(**config))
        return self._request("POST", "/configure", dict(config))

    def control(self, controls):
        method = getattr(self.camera, "control", None)
        if callable(method):
            return self._unwrap(method(**controls))
        return self._request("POST", "/control", dict(controls))

    def clock(self):
        try:
            return self._public(("clock", "get_clock"))
        except AttributeError:
            return self._request("GET", "/clock")

    def clock_samples(self, count=20):
        samples = []
        documents = []
        for _ in range(int(count)):
            before = time.perf_counter_ns()
            document = self.clock()
            after = time.perf_counter_ns()
            if not document.get("captureClockAvailable", False):
                raise RuntimeError("CameraAPI capture synchronization clock is unavailable")
            remote = document.get("captureClockNanos")
            if remote is None:
                remote = int(round(float(document["captureClockSeconds"]) * 1e9))
            sample = camera_sample(before, int(remote), after)
            samples.append(sample)
            documents.append({"clock": document, "sample": sample.to_dict()})
        return samples, documents

    def start_recording(self, name, duration):
        body = {"name": name, "container": "mov", "maxDurationSeconds": float(duration)}
        method = getattr(self.camera, "start_recording", None)
        if callable(method):
            try:
                return self._unwrap(
                    method(name=name, container="mov", max_duration_seconds=float(duration))
                )
            except TypeError:
                return self._unwrap(method(**body))
        return self._request("POST", "/record/start", body)

    def stop_recording(self):
        method = getattr(self.camera, "stop_recording", None)
        if callable(method):
            return self._unwrap(method())
        return self._request("POST", "/record/stop", timeout=35)

    def active_recording(self):
        try:
            return self._public(("recording", "get_recording"))
        except AttributeError:
            return self._request("GET", "/record")

    def file_info(self, recording_id):
        method = getattr(self.camera, "file_info", None) or getattr(self.camera, "get_file", None)
        if callable(method):
            return self._unwrap(method(recording_id))
        return self._request("GET", f"/files/{recording_id}")

    def files(self):
        try:
            return self._public(("files", "list_files"))
        except AttributeError:
            return self._request("GET", "/files")

    @staticmethod
    def _json_value(value):
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8")
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    @classmethod
    def _normalize_event(cls, raw):
        """Normalize public-client SSE shapes without discarding server fields."""
        if isinstance(raw, (bytes, bytearray, str)):
            raw = cls._json_value(raw)
        if isinstance(raw, tuple) and len(raw) >= 2:
            event_type, data = raw[0], cls._json_value(raw[1])
            if isinstance(data, dict):
                document = dict(data)
                document.setdefault("type", str(event_type))
                if "payload" not in document:
                    document = {"type": str(event_type), "payload": document}
                return document
            return {"type": str(event_type), "payload": data}
        if not isinstance(raw, dict):
            event_type = getattr(raw, "event", None) or getattr(raw, "type", None)
            data = cls._json_value(getattr(raw, "data", None))
            if event_type is not None:
                return cls._normalize_event((event_type, data))
            raise RuntimeError(f"unsupported CameraAPI event value: {type(raw).__name__}")

        document = dict(raw)
        if "data" in document and "payload" not in document:
            data = cls._json_value(document.pop("data"))
            if isinstance(data, dict) and data.get("type"):
                nested = dict(data)
                nested.setdefault("type", document.get("event"))
                return nested
            document["payload"] = data
        event_type = document.get("type") or document.get("event")
        if not event_type:
            raise RuntimeError("CameraAPI event is missing its type")
        document["type"] = str(event_type)
        document.pop("event", None)
        document.setdefault("payload", {})
        return document

    def _event_method(self):
        for name in ("events", "iter_events", "event_stream", "watch_events"):
            method = getattr(self.camera, name, None)
            if callable(method):
                return method
        return None

    def _event_worker(self):
        while not self._event_stop.is_set():
            try:
                source = self._event_method()()
                if hasattr(source, "__enter__"):
                    with source as entered:
                        self._consume_events(entered)
                else:
                    self._consume_events(source)
                if not self._event_stop.wait(0.1):
                    raise RuntimeError("CameraAPI event stream ended")
            except Exception as exc:
                with self._event_condition:
                    self._event_error = f"{type(exc).__name__}: {exc}"
                    self._event_condition.notify_all()
                self._event_stop.wait(0.25)

    def _consume_events(self, source):
        for raw in source:
            if self._event_stop.is_set():
                break
            document = self._normalize_event(raw)
            received = time.perf_counter_ns()
            with self._event_condition:
                self._event_sequence += 1
                self._event_records.append(
                    {
                        "sequence": self._event_sequence,
                        "discovery_received_mono_ns": received,
                        "event": document,
                    }
                )
                self._event_error = None
                self._event_condition.notify_all()

    def start_event_monitor(self, timeout=5.0):
        """Start the persistent SSE reader and require its initial hello event."""
        if self._event_method() is None:
            raise RuntimeError(
                "CameraAPI Python client has no SSE event iterator; expected one of "
                "events(), iter_events(), event_stream(), or watch_events()"
            )
        with self._event_condition:
            if self._event_thread is None or not self._event_thread.is_alive():
                self._event_stop.clear()
                self._event_thread = threading.Thread(
                    target=self._event_worker,
                    name="cameraapi-events",
                    daemon=True,
                )
                self._event_thread.start()
            deadline = time.monotonic() + float(timeout)
            while not self._event_records:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = self._event_error or "no hello event received"
                    raise RuntimeError(f"CameraAPI SSE stream is unavailable: {detail}")
                self._event_condition.wait(min(remaining, 0.1))
        return self.event_cursor()

    def event_cursor(self):
        with self._event_condition:
            return self._event_sequence

    def events_since(self, cursor):
        with self._event_condition:
            return [
                dict(record)
                for record in self._event_records
                if int(record["sequence"]) > int(cursor)
            ]

    @property
    def event_error(self):
        with self._event_condition:
            return self._event_error

    def wait_event(self, event_types, cursor, timeout, recording_id=None):
        wanted = {event_types} if isinstance(event_types, str) else set(event_types)
        deadline = time.monotonic() + float(timeout)
        with self._event_condition:
            while True:
                for record in self._event_records:
                    if int(record["sequence"]) <= int(cursor):
                        continue
                    event = record["event"]
                    if event.get("type") not in wanted:
                        continue
                    payload = event.get("payload") or {}
                    if recording_id is not None and payload.get("id") != recording_id:
                        continue
                    return dict(record)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = f"; last SSE error: {self._event_error}" if self._event_error else ""
                    names = ", ".join(sorted(wanted))
                    raise RuntimeError(f"CameraAPI event timeout waiting for {names}{detail}")
                self._event_condition.wait(min(remaining, 0.1))

    def wait_first_frame_event(self, recording_id, cursor, timeout=8.0):
        record = self.wait_event(
            "recording.firstFrame",
            cursor,
            timeout,
            recording_id=recording_id,
        )
        payload = record["event"].get("payload") or {}
        if payload.get("firstVideoPTSSeconds") is None:
            raise RuntimeError(
                "recording.firstFrame did not contain firstVideoPTSSeconds"
            )
        return record

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._event_stop.set()
        with self._event_condition:
            self._event_condition.notify_all()
        close = getattr(self.camera, "close", None)
        if callable(close):
            close()
        if (
            self._event_thread is not None
            and self._event_thread is not threading.current_thread()
        ):
            self._event_thread.join(timeout=1.0)

    def wait_finished(self, recording_id, timeout):
        deadline = time.monotonic() + timeout
        last_error = None
        while time.monotonic() < deadline:
            try:
                info = self.file_info(recording_id)
                if info and info.get("filename"):
                    return info
            except Exception as exc:
                last_error = exc
            time.sleep(0.1)
        raise RuntimeError(f"CameraAPI recording did not finalize: {last_error or 'timeout'}")

    def download(self, recording_id, destination):
        destination = str(destination)
        method = getattr(self.camera, "download", None)
        if not callable(method):
            raise RuntimeError("CameraAPI client has no resumable download() method")
        try:
            return method(recording_id, destination, resume=True)
        except TypeError:
            return method(recording_id, destination)

    def delete(self, recording_id):
        try:
            return self._public(("delete", "delete_file"), recording_id)
        except AttributeError:
            return self._request("DELETE", f"/files/{recording_id}")

    def snapshot(self, max_width=640, quality=0.65):
        return self._public(("snapshot",), max_width=max_width, quality=quality)
