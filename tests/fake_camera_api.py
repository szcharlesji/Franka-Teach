"""In-memory CameraAPI adapter for Discovery recorder tests."""

import time
from pathlib import Path

from frankateach.recording.clock import camera_sample


class FakeCameraAPI:
    def __init__(self, payload=b"fake-mov-payload", drops=None, interrupted=False):
        self.payload = payload
        self.recording_id = "FAKE-RECORDING"
        self.deleted = []
        self.config = {}
        self.controls = {}
        self.active = False
        self.drops = drops or {}
        self.interrupted = interrupted
        self.event_records = []
        self.event_sequence = 0
        self.finished_emitted = False

    def wait_until_ready(self):
        return True

    def _emit(self, event_type, payload):
        self.event_sequence += 1
        record = {
            "sequence": self.event_sequence,
            "discovery_received_mono_ns": time.perf_counter_ns(),
            "event": {
                "type": event_type,
                "timestamp": "2000-01-01T00:00:00Z",
                "payload": payload,
            },
        }
        self.event_records.append(record)
        return record

    def start_event_monitor(self, timeout=5):
        if not self.event_records:
            self._emit("hello", self.status())
        return self.event_cursor()

    def event_cursor(self):
        return self.event_sequence

    def events_since(self, cursor):
        return [record for record in self.event_records if record["sequence"] > cursor]

    @property
    def event_error(self):
        return None

    def wait_event(self, event_types, cursor, timeout, recording_id=None):
        wanted = {event_types} if isinstance(event_types, str) else set(event_types)
        for record in self.events_since(cursor):
            event = record["event"]
            payload = event.get("payload") or {}
            if event["type"] in wanted and (
                recording_id is None or payload.get("id") == recording_id
            ):
                return record
        raise RuntimeError(f"event not found: {sorted(wanted)}")

    def configure(self, config):
        self.config = dict(config)
        return dict(config)

    def control(self, controls):
        self.controls = dict(controls)
        return {
            "focusMode": controls["focus"]["mode"],
            "lensPosition": controls["focus"]["lensPosition"],
            "exposureMode": controls["exposure"]["mode"],
            "exposureDurationSeconds": controls["exposure"]["durationSeconds"],
            "iso": controls["exposure"]["iso"],
            "whiteBalanceMode": controls["whiteBalance"]["mode"],
            "temperature": controls["whiteBalance"]["temperature"],
            "tint": controls["whiteBalance"]["tint"],
        }

    def status(self):
        return {
            "session": {
                "running": True,
                "interrupted": False,
                "interruptionReason": None,
                "config": {
                    **dict(self.config),
                    "audioEnabled": bool(self.config.get("audio", False)),
                },
            },
            "device": {"thermalState": "nominal"},
        }

    def files(self):
        return {"recordings": [], "freeDiskBytes": 64 * 1024**3}

    def clock_samples(self, count=20):
        samples, docs = [], []
        for _ in range(count):
            before = time.perf_counter_ns()
            remote = before + 50_000_000
            after = before + 100_000
            sample = camera_sample(before, remote, after)
            samples.append(sample)
            docs.append({"sample": sample.to_dict()})
        return samples, docs

    def start_recording(self, name, duration):
        self.active = True
        self.finished_emitted = False
        started = {
            "id": self.recording_id,
            "name": name,
            "maxDurationSeconds": duration,
        }
        self._emit("recording.started", started)
        self._emit(
            "recording.firstFrame",
            {
                **started,
                "framesWritten": 1,
                "firstVideoPTSSeconds": 1000.0,
            },
        )
        return started

    def wait_first_frame_event(self, recording_id, cursor, timeout=8):
        return self.wait_event(
            "recording.firstFrame", cursor, timeout, recording_id=recording_id
        )

    def _finished(self):
        return {
            "id": self.recording_id,
            "filename": "fake.mov",
            "sizeBytes": len(self.payload),
            "width": 1920,
            "height": 1080,
            "fps": 60,
            "codec": "h264",
            "framesWritten": 60,
            "timing": {
                "firstVideoPTSSeconds": 1000.0,
                "lastVideoPTSSeconds": 1000.983333333,
                "captureDrops": self.drops.get("capture", 0),
                "writerBackpressureDrops": self.drops.get("writer", 0),
                "appendFailures": self.drops.get("append", 0),
                "keyFrameInterval": 12,
                "interruptions": [{"reason": "test"}] if self.interrupted else [],
            },
        }

    def file_info(self, recording_id):
        assert recording_id == self.recording_id
        self.active = False
        finished = self._finished()
        if not self.finished_emitted:
            self._emit("recording.autostopped", {"message": "max_duration_reached"})
            self._emit("recording.stopped", finished)
            self.finished_emitted = True
        return finished

    def stop_recording(self):
        return self.file_info(self.recording_id)

    def download(self, recording_id, destination):
        assert recording_id == self.recording_id
        Path(destination).write_bytes(self.payload)

    def delete(self, recording_id):
        self.deleted.append(recording_id)
        return {"deleted": [recording_id]}

    def snapshot(self, max_width=640, quality=0.65):
        return b"jpeg"

    def close(self):
        return None
