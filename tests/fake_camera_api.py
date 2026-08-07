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

    def wait_until_ready(self):
        return True

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
        return {
            "id": self.recording_id,
            "name": name,
            "maxDurationSeconds": duration,
        }

    def wait_first_frame(self, recording_id, timeout=8):
        assert recording_id == self.recording_id
        return {
            "id": recording_id,
            "framesWritten": 1,
            "firstVideoPTSSeconds": 1000.0,
        }

    def file_info(self, recording_id):
        assert recording_id == self.recording_id
        self.active = False
        return {
            "id": recording_id,
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
