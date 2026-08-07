"""Hardware-free tests for synchronized raw recording infrastructure."""

import asyncio
import json
import queue
import tempfile
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from frankateach.recording import recorder as recorder_module
from frankateach.recording import validation as validation_module
from frankateach.recording.bridge import RobotBridge, TelemetryHub
from frankateach.recording.camera import CameraAPIAdapter
from frankateach.recording.clock import ClockSample, fit_clock
from frankateach.recording.ownership import ArmOwnership
from frankateach.recording.protocol import PROTOCOL_VERSION, validate_keys
from frankateach.recording.profile import apply_recording_profile, validate_recording_profile
from frankateach.recording.recorder import EpisodeRecorder, recorded_dimensions
from frankateach.recording.storage import SessionStore
from frankateach.recording.validation import ValidationReport, validate_episode
from tests.fake_camera_api import FakeCameraAPI

fails = []


def check(name, condition, extra=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name} {extra}")
    if not condition:
        fails.append(name)


def test_clock_and_protocol():
    samples = []
    offset = 42_000_000
    for index in range(40):
        local = 1_000_000_000 + index * 100_000_000
        delay = 100_000 + (index % 5) * 20_000
        remote = local - offset
        samples.append(ClockSample(local, remote + delay // 2, remote + delay // 2, local + delay))
    fit = fit_clock(samples)
    mapped = fit.map_ns(2_000_000_000 - offset)
    check("clock fit recovers offset", abs(mapped - 2_000_000_000) < 100_000, mapped)
    check("clock uncertainty is bounded", fit.uncertainty_ns < 2_000_000)
    event = validate_keys(
        {
            "t": "keys",
            "sequence": 7,
            "discovery_mono_ns": 123,
            "held": ["KeyW", "KeyW"],
            "frozen": False,
        }
    )
    check("key protocol deduplicates held state", event["held"] == ["KeyW"])
    profile = apply_recording_profile(
        {"control_hz": 50, "arms": {"left": {}, "right": {}}}
    )
    check("recording profile selects 60 Hz", profile["control_hz"] == 60)
    check("recording profile configs agree", validate_recording_profile(profile))
    check(
        "quarter-turn camera rotation swaps recorded dimensions",
        recorded_dimensions(
            {"width": 1920, "height": 1080, "rotationDegrees": 90}
        )
        == (1080, 1920),
    )
    with tempfile.TemporaryDirectory() as temporary:
        first_owner = ArmOwnership(["left"], lock_root=temporary).acquire()
        try:
            try:
                ArmOwnership(["left"], lock_root=temporary).acquire()
            except RuntimeError:
                lock_refused = True
            else:
                lock_refused = False
            check("cross-user arm lock refuses a second owner", lock_refused)
        finally:
            first_owner.release()
        try:
            SessionStore(
                Path(temporary),
                "forbidden",
                enforce_discovery=True,
                hostname="robotlab-NUC8i7BEH",
            )
        except RuntimeError:
            refused = True
        else:
            refused = False
        check("Discovery recorder refuses the robot NUC", refused)


def test_strict_validation():
    samples = [
        ClockSample(
            index * 1_000_000,
            index * 1_000_000,
            index * 1_000_000,
            index * 1_000_000 + 100_000,
        )
        for index in range(20)
    ]
    fit = fit_clock(samples)
    camera = {
        "width": 1920,
        "height": 1080,
        "fps": 60,
        "codec": "h264",
        "framesWritten": 60,
        "timing": {
            "firstVideoPTSSeconds": 1000.0,
            "captureDrops": 0,
            "writerBackpressureDrops": 0,
            "appendFailures": 0,
            "keyFrameInterval": 12,
            "interruptions": [],
        },
    }
    telemetry = {"by_arm": {}, "bridge_rtt_ns": [1_000_000] * 100}
    for arm in ("left", "right"):
        telemetry["by_arm"][arm] = [
            {
                "tick_sequence": index + 1,
                "command_mono_ns": index * 16_666_667,
                "connected": True,
                "provisional": False,
                "telemetry_drops": 0,
                "error": "",
            }
            for index in range(61)
        ]
    frames = [
        {"pts_seconds": index / 60.0, "key_frame": index % 12 == 0}
        for index in range(60)
    ]
    original_probe = validation_module.probe_video
    validation_module.probe_video = lambda _: (
        {
            "width": 1920,
            "height": 1080,
            "codec_name": "h264",
            "avg_frame_rate": "60/1",
        },
        frames,
    )
    try:
        report, index = validate_episode(
            "unused.mov", camera, telemetry, fit, fit, expected_duration=1.0
        )
        check("strict validator accepts a complete episode", report.accepted)
        check("strict validator maps every frame", len(index) == 60)
        frames[0]["decode_order"] = 1
        frames[1]["decode_order"] = 0
        report, _ = validate_episode(
            "unused.mov", camera, telemetry, fit, fit, expected_duration=1.0
        )
        check(
            "strict validator rejects frame reordering",
            "video_frame_reordering" in report.failures,
        )
        frames[0].pop("decode_order")
        frames[1].pop("decode_order")
        camera["timing"]["captureDrops"] = 1
        camera["timing"]["interruptions"] = [{"reason": "test"}]
        report, _ = validate_episode(
            "unused.mov", camera, telemetry, fit, fit, expected_duration=1.0
        )
        check(
            "strict validator identifies capture-side drops",
            "camera_captureDrops" in report.failures,
        )
        check(
            "strict validator identifies interruptions",
            "camera_interruption" in report.failures,
        )
    finally:
        validation_module.probe_video = original_probe


def test_packet_probe():
    calls = []
    original_run = validation_module.subprocess.run

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "ffmpeg":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout="""{
  "streams": [{"codec_name":"h264","width":1920,"height":1080,"avg_frame_rate":"60/1"}],
  "packets": [
    {"pts_time":"0.033333","flags":"___"},
    {"pts_time":"0.000000","flags":"K__"},
    {"pts_time":"0.016667","flags":"___"}
  ]
}""",
        )

    validation_module.subprocess.run = fake_run
    try:
        _, packets = validation_module.probe_video("unused.mov")
    finally:
        validation_module.subprocess.run = original_run
    check("video validation performs a real decode", calls[0][0] == "ffmpeg")
    check("frame indexing reads packets", "-show_packets" in calls[1])
    check(
        "packet PTS are sorted into presentation order",
        packets[0]["pts_seconds"] == 0,
    )
    check(
        "packet probe retains original decode order",
        [packet["decode_order"] for packet in packets] == [1, 2, 0],
    )
    check("packet flags preserve keyframes", packets[0]["key_frame"])


def test_camera_adapter_contract():
    class DocumentedClient:
        def __init__(self):
            self.event_queue = queue.Queue()
            self.configure_args = None

        def configure(
            self,
            *,
            format_index=None,
            key_frame_interval=None,
            rotation_degrees=None,
            allow_frame_reordering=None,
            **kwargs,
        ):
            self.configure_args = {
                "format_index": format_index,
                "key_frame_interval": key_frame_interval,
                "rotation_degrees": rotation_degrees,
                "allow_frame_reordering": allow_frame_reordering,
                **kwargs,
            }
            return self.configure_args

        def events(self):
            yield {
                "type": "hello",
                "timestamp": "2000-01-01T00:00:00Z",
                "payload": {},
            }
            while True:
                event = self.event_queue.get()
                if event is None:
                    return
                yield event

        def snapshot(self, max_width=640, quality=0.65):
            return b"\xff\xd8cameraapi-jpeg\xff\xd9"

        def close(self):
            self.event_queue.put(None)

    client = DocumentedClient()
    adapter = CameraAPIAdapter(client)
    adapter.start_event_monitor()
    cursor = adapter.event_cursor()
    configured = adapter.configure(
        {
            "formatIndex": 12,
            "keyFrameInterval": 12,
            "rotationDegrees": 90,
            "allowFrameReordering": False,
            "fps": 60,
        }
    )
    check("adapter maps exact format index", configured["format_index"] == 12)
    check(
        "adapter maps frame-reordering control",
        configured["allow_frame_reordering"] is False,
    )
    client.event_queue.put(
        (
            "recording.firstFrame",
            {
                "type": "recording.firstFrame",
                "timestamp": "2000-01-01T00:00:01Z",
                "payload": {
                    "id": "recording-1",
                    "firstVideoPTSSeconds": 123.5,
                },
            },
        )
    )
    event = adapter.wait_first_frame_event("recording-1", cursor, timeout=1)
    check(
        "adapter waits for real first-frame SSE",
        event["event"]["payload"]["firstVideoPTSSeconds"] == 123.5,
    )
    check(
        "adapter timestamps SSE receipt on Discovery",
        event["discovery_received_mono_ns"] > 0,
    )
    check(
        "adapter preserves binary JPEG snapshots",
        adapter.snapshot().startswith(b"\xff\xd8"),
    )
    adapter.close()


class FakeSession:
    def __init__(self, arm):
        self.arm = arm
        self.mode = "play"
        self.provisional = False
        self.error = ""
        self.calls = []
        self.box = SimpleNamespace(half_extents=np.array([0.1, 0.2]), yaw=0.0)
        self.operator = SimpleNamespace(link=None)
        self._status = SimpleNamespace(
            connected=True,
            stale=True,
            error="",
            rate=60.0,
            pos=np.array([0.3, 0.0, 0.2]),
            box_pos=np.zeros(2),
            speed=0.0,
            homing=False,
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )

    def set_intent(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def get_status(self):
        return self._status

    def set_speed_limit(self, speed):
        return speed


class FakeSupervisor:
    def __init__(self):
        root = Path(__file__).resolve().parent.parent
        self.stacks = {
            arm: SimpleNamespace(
                deoxys_config=f"deoxys_{arm}_60.yml",
                nuc_config=root / "deoxys_configs" / f"franka_{arm}_60.yml",
                control_freq=60,
                num_steps=1,
            )
            for arm in ("left", "right")
        }

    def status(self):
        return {"left": {"server": "up"}, "right": {"server": "up"}}


async def test_robot_bridge():
    sessions = {arm: FakeSession(arm) for arm in ("left", "right")}
    cfg = {
        "control_hz": 60,
        "speed": 0.1,
        "margin": 0.01,
        "arms": {"left": {"keys": "wasd"}, "right": {"keys": "ijkl"}},
    }
    hub = TelemetryHub(maxsize=4)
    bridge = RobotBridge(sessions, cfg, FakeSupervisor(), hub)
    client = TestClient(TestServer(bridge.app()))
    await client.start_server()
    ws = await client.ws_connect("/ws")
    await ws.send_json(
        {"t": "hello", "protocol_version": PROTOCOL_VERSION, "session": "test"}
    )
    hello = await ws.receive_json(timeout=2)
    check("bridge version handshake", hello["t"] == "hello")
    health = await client.get("/health")
    health_document = await health.json()
    check(
        "bridge reports hashed 60 Hz configuration",
        health_document["host"]["configuration"]["control_hz"] == 60,
    )
    await ws.send_json(
        {
            "t": "keys",
            "sequence": 9,
            "discovery_mono_ns": 111,
            "held": ["KeyW", "KeyJ"],
            "frozen": False,
        }
    )
    await asyncio.sleep(0.03)
    check("bridge routes WASD to left", sessions["left"].calls[-1][0][:2] == (1.0, 0.0))
    check("bridge routes IJKL to right", sessions["right"].calls[-1][0][:2] == (0.0, 1.0))
    check(
        "bridge preserves intent sequence",
        sessions["left"].calls[-1][1]["sequence"] == 9,
    )
    hub.publish(
        {
            "t": "telemetry",
            "arm": "left",
            "tick_sequence": 1,
            "command_mono_ns": time.perf_counter_ns(),
        }
    )
    telemetry = None
    for _ in range(20):
        message = await ws.receive_json(timeout=2)
        if message.get("t") == "telemetry_batch":
            telemetry = message
            break
    check("bridge publishes telemetry batch", telemetry is not None)
    await ws.send_json(
        {"t": "clock_probe", "probe_id": "clock", "local_send_ns": 100}
    )
    reply = None
    for _ in range(20):
        message = await ws.receive_json(timeout=2)
        if message.get("t") == "clock_reply":
            reply = message
            break
    check(
        "bridge provides two-timestamp clock reply",
        reply["remote_send_ns"] >= reply["remote_recv_ns"],
    )
    await ws.close()
    await client.close()
    await asyncio.sleep(0.05)
    check("bridge disconnect freezes arms", sessions["left"].calls[-1][1].get("frozen"))


class FakeDiscoveryBridge:
    def __init__(self):
        self.connected = True
        self.error = ""
        self.rtt_ns = deque([1_000_000] * 200, maxlen=4000)
        self.status = {
            "control_hz": 60,
            "arms": {
                arm: {
                    "connected": True,
                    "provisional": False,
                    "mode": "play",
                }
                for arm in ("left", "right")
            },
        }

    async def clock_samples(self, count=20):
        samples, docs = [], []
        for index in range(count):
            local = time.perf_counter_ns() + index * 1000
            remote = local - 25_000_000
            sample = ClockSample(
                local, remote + 50_000, remote + 50_000, local + 100_000
            )
            samples.append(sample)
            docs.append({"sample": sample.to_dict()})
        return samples, docs


def warm_rate_gate(recorder, first_sequence=1):
    now = time.perf_counter_ns()
    for arm in ("left", "right"):
        for index in range(301):
            event = {
                "t": "telemetry",
                "arm": arm,
                "tick_sequence": first_sequence + index,
                "intent_sequence": index,
                "command_mono_ns": now - int(5e9) + index * 16_666_667,
                "state_mono_ns": now - int(5e9) + index * 16_666_667 + 1000,
                "discovery_recv_mono_ns": now - int(5e9) + index * 16_666_667,
                "connected": True,
                "provisional": False,
                "error": "",
                "telemetry_drops": 0,
                "commanded_box_xy": [0.0, 0.0],
            }
            recorder.on_telemetry(event)


class SlowFakeCamera(FakeCameraAPI):
    def file_info(self, recording_id):
        if self.active:
            raise RuntimeError("still recording")
        return super().file_info(recording_id)

    def stop_recording(self):
        self.active = False
        return super().file_info(self.recording_id)


async def test_episode_lifecycle():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        profile = root / "camera.yaml"
        profile.write_text(
            """capture:
  formatIndex: 12
  width: 1920
  height: 1080
  fps: 60
  codec: h264
  keyFrameInterval: 12
  allowFrameReordering: false
  audio: false
  stabilization: "off"
  rotationDegrees: 0
controls:
  focus: {mode: manual, lensPosition: 0.5}
  exposure: {mode: manual, durationSeconds: 0.002, iso: 200}
  whiteBalance: {mode: manual, temperature: 5000, tint: 0}
""",
            encoding="utf-8",
        )
        camera_repo = root / "Camera-API"
        camera_repo.mkdir()
        store = SessionStore(root / "data", "test", started_at=None)
        orphan = store.air_hockey_root / "test_20000101T000000Z" / ".partial" / "episode_999999"
        orphan.mkdir(parents=True)
        (orphan / "camera_start.json").write_text(
            '{"id":"PHONE-ORPHAN"}\n', encoding="utf-8"
        )
        recovered = store.scan_orphan_partials()
        check(
            "recovery scan finds partial and phone recording ID",
            any(item["recording_id"] == "PHONE-ORPHAN" for item in recovered),
        )
        camera = FakeCameraAPI()
        bridge = FakeDiscoveryBridge()
        recorder = EpisodeRecorder(
            store,
            camera,
            bridge,
            profile,
            Path(__file__).resolve().parent.parent,
            camera_repo,
            default_duration=1,
            post_roll_seconds=0,
        )
        await recorder.prepare_camera()
        now = time.perf_counter_ns()
        warm_rate_gate(recorder)

        original_validate = recorder_module.validate_episode

        def accepted_validation(*args, **kwargs):
            return ValidationReport(), [
                {
                    "frame": 0,
                    "file_pts_seconds": 0.0,
                    "camera_pts_seconds": 1000.0,
                    "discovery_mono_ns": now,
                }
            ]

        recorder_module.validate_episode = accepted_validation
        try:
            task = await recorder.start(1)
            result = await task
        finally:
            recorder_module.validate_episode = original_validate
        check("accepted episode finalized under raw", result["accepted"])
        path = Path(result["path"])
        check("raw episode has manifest", (path / "manifest.json").is_file())
        check("raw episode has checksums", (path / "checksums.sha256").is_file())
        camera_document = json.loads(
            (path / "camera.json").read_text(encoding="utf-8")
        )
        event_types = [row["event"]["type"] for row in camera_document["events"]]
        check(
            "camera events are retained verbatim",
            event_types
            == [
                "recording.started",
                "recording.firstFrame",
                "recording.autostopped",
                "recording.stopped",
            ],
        )
        check("camera health is sampled during capture", bool(camera_document["health"]))
        check("phone file deleted after verification", camera.deleted == [camera.recording_id])
        check("partial directory is empty", not any(store.partial_dir.iterdir()))

        rejected_camera = FakeCameraAPI(drops={"capture": 1})
        recorder.camera = rejected_camera
        await recorder.prepare_camera()
        warm_rate_gate(recorder, first_sequence=302)

        def rejected_validation(*args, **kwargs):
            report = ValidationReport()
            report.fail("camera_captureDrops")
            return report, []

        recorder_module.validate_episode = rejected_validation
        try:
            task = await recorder.start(1)
            rejected = await task
        finally:
            recorder_module.validate_episode = original_validate
        check("invalid completed episode is quarantined", not rejected["accepted"])
        check(
            "quarantined episode lives under rejected",
            Path(rejected["path"]).parent == store.rejected_dir,
        )
        check(
            "verified rejected phone file is deleted",
            rejected_camera.deleted == [rejected_camera.recording_id],
        )

        abort_camera = SlowFakeCamera()
        recorder.camera = abort_camera
        await recorder.prepare_camera()
        warm_rate_gate(recorder, first_sequence=603)
        task = await recorder.start(1)
        for _ in range(100):
            if recorder.state == "recording":
                break
            await asyncio.sleep(0.01)
        aborted = await recorder.abort()
        check(
            "manual abort leaves audit-only outcome",
            aborted["outcome"] == "manual_abort",
        )
        check(
            "manual abort removes phone file",
            abort_camera.deleted == [abort_camera.recording_id],
        )
        check("manual abort removes partial directory", not any(store.partial_dir.iterdir()))
        check(
            "manual abort is present in session audit",
            '"event":"manual_abort"' in (store.path / "session_events.ndjson").read_text(),
        )
        store.close()


async def main():
    test_clock_and_protocol()
    test_strict_validation()
    test_packet_probe()
    test_camera_adapter_contract()
    await test_robot_bridge()
    await test_episode_lifecycle()
    print("\nFAILED:", ", ".join(fails) if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
