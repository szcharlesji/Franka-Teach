"""Adapter around the standalone Camera-API Python client on Discovery."""

import importlib
import inspect
import json
import sys
import time
from pathlib import Path

from frankateach.recording.clock import camera_sample


class CameraAPIAdapter:
    """Normalize Camera-API client revisions to the HTTP contract in its docs."""

    def __init__(self, camera):
        self.camera = camera

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

    def wait_first_frame(self, recording_id, timeout=8.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            active = self.active_recording()
            if active.get("id") == recording_id and (
                active.get("firstVideoPTSSeconds") is not None
                or int(active.get("framesWritten", 0)) > 0
            ):
                return active
            time.sleep(0.02)
        raise RuntimeError("CameraAPI did not produce a first frame in time")

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
