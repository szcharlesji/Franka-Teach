"""Discovery-side synchronized episode state machine."""

import asyncio
import shutil
import socket
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path

import yaml

from frankateach.recording.clock import fit_clock, percentile_ns
from frankateach.recording.protocol import SCHEMA_VERSION
from frankateach.recording.storage import sha256_file, utc_now, write_checksums
from frankateach.recording.validation import validate_episode, write_frame_index


class PreflightError(RuntimeError):
    pass


def recorded_dimensions(capture):
    width = int(capture["width"])
    height = int(capture["height"])
    if int(capture.get("rotationDegrees", 0)) in {90, 270}:
        return height, width
    return width, height


def load_camera_profile(path):
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        profile = yaml.safe_load(stream) or {}
    capture = profile.get("capture") or {}
    controls = profile.get("controls") or {}
    missing = []
    required_capture = {
        "width": 1920,
        "height": 1080,
        "fps": 60,
        "codec": "h264",
        "keyFrameInterval": 12,
        "allowFrameReordering": False,
        "audio": False,
        "stabilization": "off",
    }
    for field, expected in required_capture.items():
        if capture.get(field) != expected:
            missing.append(f"capture.{field} must be {expected!r}")
    format_index = capture.get("formatIndex")
    if (
        isinstance(format_index, bool)
        or not isinstance(format_index, int)
        or format_index < 0
    ):
        missing.append("capture.formatIndex must be a non-negative integer from /formats")
    if capture.get("rotationDegrees", 0) not in {0, 90, 180, 270}:
        missing.append("capture.rotationDegrees must be 0, 90, 180, or 270")
    for group, fields in {
        "focus": ("lensPosition",),
        "exposure": ("durationSeconds", "iso"),
        "whiteBalance": ("temperature", "tint"),
    }.items():
        value = controls.get(group) or {}
        if value.get("mode") != "manual":
            missing.append(f"controls.{group}.mode must be manual")
        for field in fields:
            if value.get(field) is None:
                missing.append(f"controls.{group}.{field} must be calibrated")
    if missing:
        raise PreflightError("camera profile is incomplete:\n  - " + "\n  - ".join(missing))
    return {"capture": capture, "controls": controls}


def git_state(repo):
    repo = str(repo)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except Exception as exc:
        return {"revision": None, "dirty": None, "error": str(exc)}


class EpisodeRecorder:
    def __init__(
        self,
        store,
        camera,
        bridge,
        camera_profile_path,
        repo_root,
        camera_repo_root,
        default_duration=20.0,
        pre_roll_seconds=1.0,
        post_roll_seconds=1.0,
    ):
        self.store = store
        self.camera = camera
        self.bridge = bridge
        self.camera_profile_path = Path(camera_profile_path).expanduser().resolve()
        self.repo_root = Path(repo_root).resolve()
        self.camera_repo_root = Path(camera_repo_root).expanduser().resolve()
        self.default_duration = float(default_duration)
        self.pre_roll_ns = int(pre_roll_seconds * 1e9)
        self.post_roll_seconds = float(post_roll_seconds)

        self.telemetry_ring = deque(maxlen=5000)
        self.key_ring = deque(maxlen=1000)
        self.sequence = 0
        self.key_sequence = 0
        self.writer = None
        self.task = None
        self.abort_event = asyncio.Event()
        self.episode_telemetry = []
        self.episode_keys = []
        self.episode_rtts = []
        self.state = "idle"
        self.message = ""
        self.current_episode = None
        self.camera_status = {}
        self.camera_health_error = None
        self.applied_capture = {}
        self.applied_controls = {}
        self.last_result = None
        self.orphan_partials = store.scan_orphan_partials()
        if self.orphan_partials:
            store.audit("orphan_partials_detected", partials=self.orphan_partials)

    @property
    def busy(self):
        return self.task is not None and not self.task.done()

    def on_telemetry(self, event):
        self.telemetry_ring.append(event)
        if self.writer is not None:
            self.writer.write_robot(event)
            self.episode_telemetry.append(event)

    def on_keys(self, event):
        self.key_ring.append(event)
        if self.writer is not None:
            self.writer.write_keys(event)
            self.episode_keys.append(event)

    async def prepare_camera(self):
        profile = load_camera_profile(self.camera_profile_path)
        await asyncio.to_thread(self.camera.wait_until_ready)
        await asyncio.to_thread(self.camera.start_event_monitor)
        self.applied_capture = await asyncio.to_thread(
            self.camera.configure, profile["capture"]
        )
        self.applied_controls = await asyncio.to_thread(
            self.camera.control, profile["controls"]
        )
        await self.refresh_camera_status()
        return self.camera_status

    async def refresh_camera_status(self):
        try:
            status = await asyncio.to_thread(self.camera.status)
        except Exception as exc:
            self.camera_health_error = f"{type(exc).__name__}: {exc}"
            raise
        self.camera_status = status
        self.camera_health_error = None
        return status

    async def camera_health_loop(self, interval=1.0):
        while True:
            try:
                await self.refresh_camera_status()
            except Exception:
                pass
            await asyncio.sleep(interval)

    def _recent_by_arm(self, seconds=5.0):
        cutoff = time.perf_counter_ns() - int(seconds * 1e9)
        rows = defaultdict(list)
        for event in self.telemetry_ring:
            if int(event.get("discovery_recv_mono_ns", 0)) >= cutoff:
                rows[event.get("arm")].append(event)
        return rows

    def _rates(self, seconds=5.0):
        rates = {}
        for arm, rows in self._recent_by_arm(seconds).items():
            if len(rows) < 2:
                rates[arm] = 0.0
                continue
            start = int(rows[0]["command_mono_ns"])
            end = int(rows[-1]["command_mono_ns"])
            rates[arm] = (len(rows) - 1) * 1e9 / max(1, end - start)
        return rates

    def status(self):
        rtt_p99 = percentile_ns(self.bridge.rtt_ns, 99)
        return {
            "state": self.state,
            "message": self.message,
            "busy": self.busy,
            "episode": self.current_episode,
            "session_path": str(self.store.path),
            "bridge_connected": self.bridge.connected,
            "bridge_error": self.bridge.error,
            "bridge_p99_ms": None if rtt_p99 is None else rtt_p99 / 1e6,
            "rates_hz": self._rates(),
            "robot": self.bridge.status,
            "camera": self.camera_status,
            "camera_error": self.camera_health_error,
            "camera_event_error": getattr(self.camera, "event_error", None),
            "last_result": self.last_result,
            "orphan_partials": self.orphan_partials,
        }

    async def preflight(self):
        failures = []
        if not self.bridge.connected:
            failures.append("robot bridge disconnected")
        status = self.bridge.status
        if float(status.get("control_hz", 0)) != 60.0:
            failures.append("NUC is not running the recording_60 profile")
        for arm in ("left", "right"):
            arm_status = (status.get("arms") or {}).get(arm) or {}
            if not arm_status.get("connected"):
                failures.append(f"{arm} arm disconnected")
            if arm_status.get("provisional"):
                failures.append(f"{arm} arm has only a provisional calibration")
            if arm_status.get("mode") != "play":
                failures.append(f"{arm} arm is not in play mode")

        rates = self._rates(5.0)
        recent = self._recent_by_arm(5.0)
        for arm in ("left", "right"):
            rows = recent.get(arm) or []
            span = (
                int(rows[-1].get("discovery_recv_mono_ns", 0))
                - int(rows[0].get("discovery_recv_mono_ns", 0))
                if len(rows) > 1
                else 0
            )
            if span < int(4e9):
                failures.append(f"{arm} rate gate is still warming up")
            elif rates.get(arm, 0.0) < 57.0:
                failures.append(f"{arm} loop rate {rates.get(arm, 0.0):.1f} Hz is below 57 Hz")

        rtt_p99 = percentile_ns(self.bridge.rtt_ns, 99)
        if len(self.bridge.rtt_ns) < 20:
            failures.append("bridge RTT gate is still warming up")
        elif rtt_p99 is None or rtt_p99 > int(20e6):
            failures.append(
                f"bridge p99 RTT {(rtt_p99 or 0) / 1e6:.2f} ms exceeds 20 ms"
            )

        profile = load_camera_profile(self.camera_profile_path)
        event_error = getattr(self.camera, "event_error", None)
        if event_error:
            failures.append(f"CameraAPI SSE stream failed: {event_error}")
        try:
            self.camera_status = await self.refresh_camera_status()
        except Exception as exc:
            failures.append(f"CameraAPI health check failed: {type(exc).__name__}: {exc}")
            self.camera_status = {}
        camera_files = await asyncio.to_thread(self.camera.files)
        phone_free = int(camera_files.get("freeDiskBytes", 0) or 0)
        if phone_free < 2 * 1024**3:
            failures.append(
                f"phone free storage {phone_free / 1024**3:.1f} GiB is below 2 GiB"
            )
        discovery_free = shutil.disk_usage(self.store.path).free
        if discovery_free < 20 * 1024**3:
            failures.append(
                f"Discovery free storage {discovery_free / 1024**3:.1f} GiB is below 20 GiB"
            )
        session_status = self.camera_status.get("session") or {}
        capture = (
            session_status.get("config")
            or self.camera_status.get("configuration")
            or self.camera_status.get("capture")
            or self.camera_status
        )
        if session_status and not session_status.get("running", False):
            failures.append("CameraAPI capture session is not running")
        if session_status.get("interrupted", False):
            failures.append(
                "CameraAPI session is interrupted: "
                f"{session_status.get('interruptionReason')}"
            )
        expected_capture = {
            "width": 1920,
            "height": 1080,
            "fps": 60,
            "codec": "h264",
            "formatIndex": profile["capture"]["formatIndex"],
            "audioEnabled": False,
            "stabilization": "off",
            "rotationDegrees": profile["capture"].get("rotationDegrees", 0),
        }
        for field, expected in expected_capture.items():
            actual = capture.get(field)
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                try:
                    matches = abs(float(actual) - float(expected)) <= 1e-6
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = str(actual).lower() == str(expected).lower()
            if not matches:
                failures.append(f"camera {field}={actual!r}, expected {expected!r}")
        applied_controls = self.applied_controls.get("controls") or self.applied_controls
        expected_controls = {
            "focusMode": profile["controls"]["focus"]["mode"],
            "lensPosition": profile["controls"]["focus"]["lensPosition"],
            "exposureMode": profile["controls"]["exposure"]["mode"],
            "exposureDurationSeconds": profile["controls"]["exposure"][
                "durationSeconds"
            ],
            "iso": profile["controls"]["exposure"]["iso"],
            "whiteBalanceMode": profile["controls"]["whiteBalance"]["mode"],
            "temperature": profile["controls"]["whiteBalance"]["temperature"],
            "tint": profile["controls"]["whiteBalance"]["tint"],
        }
        for field, expected in expected_controls.items():
            actual = applied_controls.get(field)
            try:
                tolerance = max(1e-6, abs(float(expected)) * 0.01)
                matches = abs(float(actual) - float(expected)) <= tolerance
            except (TypeError, ValueError):
                matches = actual == expected
            if not matches:
                failures.append(
                    f"camera control {field}={actual!r}, expected {expected!r}"
                )
        device = self.camera_status.get("device") or {}
        if str(device.get("thermalState", "nominal")).lower() in {
            "serious",
            "critical",
        }:
            failures.append(f"phone thermal state is {device.get('thermalState')}")
        if failures:
            raise PreflightError("; ".join(failures))

        camera_samples, camera_docs = await asyncio.to_thread(self.camera.clock_samples, 20)
        nuc_samples, nuc_docs = await self.bridge.clock_samples(20)
        camera_fit = fit_clock(camera_samples)
        nuc_fit = fit_clock(nuc_samples)
        if camera_fit.uncertainty_ns > int(2e6):
            raise PreflightError(
                f"camera clock uncertainty {camera_fit.uncertainty_ns / 1e6:.3f} ms exceeds 2 ms"
            )
        if nuc_fit.uncertainty_ns > int(2e6):
            raise PreflightError(
                f"NUC clock uncertainty {nuc_fit.uncertainty_ns / 1e6:.3f} ms exceeds 2 ms"
            )
        return {
            "profile": profile,
            "camera_samples": camera_samples,
            "camera_documents": camera_docs,
            "camera_fit": camera_fit,
            "nuc_samples": nuc_samples,
            "nuc_documents": nuc_docs,
            "nuc_fit": nuc_fit,
            "rates": rates,
            "bridge_p99_ns": rtt_p99,
        }

    async def start(self, duration=None):
        if self.busy:
            raise RuntimeError("an episode is already active or finalizing")
        duration = self.default_duration if duration is None else float(duration)
        if not 1.0 <= duration <= 600.0:
            raise ValueError("duration must be between 1 and 600 seconds")
        self.abort_event = asyncio.Event()
        self.task = asyncio.create_task(self._run(duration), name="air-hockey-episode")
        return self.task

    async def abort(self):
        if not self.busy:
            raise RuntimeError("no episode is active")
        if self.state not in {"preflight", "recording"}:
            raise RuntimeError(f"episode can no longer be aborted while {self.state}")
        self.abort_event.set()
        return await self.task

    async def _wait_camera(self, recording_id, duration, automatic_failures):
        deadline = time.monotonic() + duration + 40.0
        next_health = 0.0
        health_failures = 0
        health_samples = []
        while time.monotonic() < deadline:
            if self.abort_event.is_set():
                finished = await asyncio.to_thread(self.camera.stop_recording)
                return "abort", finished, health_samples
            if not self.bridge.connected and "bridge_disconnected" not in automatic_failures:
                automatic_failures.append("bridge_disconnected")
                try:
                    finished = await asyncio.to_thread(self.camera.stop_recording)
                    return "failed", finished, health_samples
                except Exception:
                    pass

            now = time.monotonic()
            if now >= next_health:
                sample = {"discovery_mono_ns": time.perf_counter_ns()}
                try:
                    status = await self.refresh_camera_status()
                    sample["status"] = status
                    health_failures = 0
                    session = status.get("session") or {}
                    device = status.get("device") or {}
                    stop_reason = None
                    if not session.get("running", False):
                        stop_reason = "camera_session_not_running"
                    elif session.get("interrupted", False):
                        stop_reason = "camera_interrupted"
                    thermal = str(device.get("thermalState", "nominal")).lower()
                    if thermal in {"serious", "critical"}:
                        reason = f"camera_thermal_{thermal}"
                        if reason not in automatic_failures:
                            automatic_failures.append(reason)
                    event_error = getattr(self.camera, "event_error", None)
                    if event_error and "camera_event_stream" not in automatic_failures:
                        automatic_failures.append("camera_event_stream")
                        sample["event_error"] = event_error
                    if stop_reason is not None:
                        if stop_reason not in automatic_failures:
                            automatic_failures.append(stop_reason)
                        health_samples.append(sample)
                        try:
                            finished = await asyncio.to_thread(self.camera.stop_recording)
                            return "failed", finished, health_samples
                        except Exception as exc:
                            sample["stop_error"] = f"{type(exc).__name__}: {exc}"
                except Exception as exc:
                    health_failures += 1
                    sample["error"] = f"{type(exc).__name__}: {exc}"
                    if health_failures >= 3 and "camera_disconnected" not in automatic_failures:
                        automatic_failures.append("camera_disconnected")
                        try:
                            finished = await asyncio.to_thread(self.camera.stop_recording)
                            health_samples.append(sample)
                            return "failed", finished, health_samples
                        except Exception as stop_exc:
                            sample["stop_error"] = (
                                f"{type(stop_exc).__name__}: {stop_exc}"
                            )
                health_samples.append(sample)
                next_health = now + 0.25
            try:
                info = await asyncio.to_thread(self.camera.file_info, recording_id)
                if info and info.get("filename"):
                    return "complete", info, health_samples
            except Exception:
                pass
            await asyncio.sleep(0.05)
        automatic_failures.append("camera_finalize_timeout")
        finished = await asyncio.to_thread(self.camera.stop_recording)
        return "failed", finished, health_samples

    async def _run(self, duration):
        self.state = "preflight"
        self.message = "running strict camera, clock, network, and 60 Hz gates"
        self.current_episode = None
        self.last_result = None
        try:
            preflight = await self.preflight()
        except Exception as exc:
            self.state = "blocked"
            self.message = str(exc)
            self.store.audit("preflight_blocked", reason=str(exc))
            raise

        if self.abort_event.is_set():
            self.store.audit("manual_abort", episode=None, phase="preflight")
            self.state = "idle"
            self.message = "preflight aborted"
            self.last_result = {"episode": None, "outcome": "manual_abort"}
            return self.last_result

        self.sequence += 1
        writer = self.store.new_episode(self.sequence)
        self.writer = writer
        self.current_episode = writer.episode_id
        self.episode_telemetry = []
        self.episode_keys = []
        self.episode_rtts = []
        start_discovery_ns = time.perf_counter_ns()
        cutoff = start_discovery_ns - self.pre_roll_ns
        for event in self.telemetry_ring:
            if int(event.get("discovery_recv_mono_ns", 0)) >= cutoff:
                writer.write_robot(event)
                self.episode_telemetry.append(event)
        for event in self.key_ring:
            if int(event.get("discovery_recv_mono_ns", 0)) >= cutoff:
                writer.write_keys(event)
                self.episode_keys.append(event)
        rtt_start = len(self.bridge.rtt_ns)

        recording_id = None
        automatic_failures = []
        self.store.audit("episode_started", episode=writer.episode_id, duration=duration)
        try:
            self.state = "recording"
            self.message = "starting CameraAPI recording"
            camera_event_cursor = self.camera.event_cursor()
            camera_start = await asyncio.to_thread(
                self.camera.start_recording, writer.episode_id, duration
            )
            camera_start_received_ns = time.perf_counter_ns()
            writer.write_json("camera_start.json", camera_start)
            recording_id = camera_start["id"]
            first_frame_event = await asyncio.to_thread(
                self.camera.wait_first_frame_event,
                recording_id,
                camera_event_cursor,
                8.0,
            )
            first_frame = first_frame_event["event"]["payload"]
            self.message = "recording fixed-duration clip"
            outcome, finished, camera_health = await self._wait_camera(
                recording_id, duration, automatic_failures
            )
            finished_received_ns = time.perf_counter_ns()

            if outcome == "abort":
                self.state = "aborting"
                phone_delete_error = None
                try:
                    await asyncio.to_thread(self.camera.delete, recording_id)
                except Exception as exc:
                    phone_delete_error = f"{type(exc).__name__}: {exc}"
                finally:
                    self.writer = None
                    writer.discard()
                self.store.audit(
                    "manual_abort",
                    episode=writer.episode_id,
                    recording_id=recording_id,
                    started_discovery_mono_ns=start_discovery_ns,
                    aborted_discovery_mono_ns=time.perf_counter_ns(),
                    phone_delete_error=phone_delete_error,
                )
                self.state = "idle"
                self.message = "episode aborted and discarded"
                if phone_delete_error:
                    self.message += f"; phone cleanup failed: {phone_delete_error}"
                self.last_result = {
                    "episode": writer.episode_id,
                    "outcome": "manual_abort",
                    "phone_deleted": phone_delete_error is None,
                    "phone_delete_error": phone_delete_error,
                }
                return self.last_result

            try:
                await asyncio.to_thread(
                    self.camera.wait_event,
                    "recording.stopped",
                    camera_event_cursor,
                    2.0,
                    recording_id,
                )
            except Exception:
                automatic_failures.append("camera_stopped_event_missing")
            camera_events = self.camera.events_since(camera_event_cursor)
            event_types = [record["event"].get("type") for record in camera_events]
            if "recording.started" not in event_types:
                automatic_failures.append("camera_started_event_missing")
            if "recording.firstFrame" not in event_types:
                automatic_failures.append("camera_first_frame_event_missing")

            self.state = "finalizing"
            self.message = "capturing post-roll and clock samples"
            await asyncio.sleep(self.post_roll_seconds)
            self.writer = None
            writer.close_streams()
            self.episode_rtts = list(self.bridge.rtt_ns)[rtt_start:]

            post_camera_samples = []
            post_camera_docs = []
            post_nuc_samples = []
            post_nuc_docs = []
            try:
                post_camera_samples, post_camera_docs = await asyncio.to_thread(
                    self.camera.clock_samples, 20
                )
            except Exception as exc:
                automatic_failures.append(f"post_camera_clock:{exc}")
            try:
                post_nuc_samples, post_nuc_docs = await self.bridge.clock_samples(20)
            except Exception as exc:
                automatic_failures.append(f"post_nuc_clock:{exc}")

            camera_fit = fit_clock(preflight["camera_samples"] + post_camera_samples)
            nuc_fit = fit_clock(preflight["nuc_samples"] + post_nuc_samples)
            clocks = {
                "camera": {
                    "pre": preflight["camera_documents"],
                    "post": post_camera_docs,
                    "fit": camera_fit.to_dict(),
                },
                "nuc": {
                    "pre": preflight["nuc_documents"],
                    "post": post_nuc_docs,
                    "fit": nuc_fit.to_dict(),
                },
            }
            writer.write_json("clocks.json", clocks)
            camera_document = {
                "status": self.camera_status,
                "applied_capture": self.applied_capture,
                "applied_controls": self.applied_controls,
                "start": camera_start,
                "first_frame": first_frame,
                "finished": finished,
                "http_receipts": {
                    "record_start_discovery_mono_ns": camera_start_received_ns,
                    "record_finished_discovery_mono_ns": finished_received_ns,
                },
                "events": camera_events,
                "health": camera_health,
            }
            writer.write_json("camera.json", camera_document)
            (writer.path / "camera_start.json").unlink(missing_ok=True)

            self.message = "downloading and validating video"
            video_path = writer.path / "video.mov"
            await asyncio.to_thread(self.camera.download, recording_id, video_path)
            size_matches = video_path.stat().st_size == int(finished.get("sizeBytes", -1))
            capture_profile = preflight["profile"]["capture"]
            output_width, output_height = recorded_dimensions(capture_profile)
            report, frame_index = await asyncio.to_thread(
                validate_episode,
                video_path,
                finished,
                {
                    "by_arm": {
                        arm: [row for row in self.episode_telemetry if row.get("arm") == arm]
                        for arm in ("left", "right")
                    },
                    "bridge_rtt_ns": self.episode_rtts,
                },
                camera_fit,
                nuc_fit,
                expected_duration=duration,
                expected_width=output_width,
                expected_height=output_height,
                expected_fps=float(capture_profile["fps"]),
                expected_gop=int(capture_profile["keyFrameInterval"]),
            )
            if not size_matches:
                report.fail("download_size")
            for reason in automatic_failures:
                report.fail(reason)
            write_frame_index(writer.path / "frames.csv", frame_index)

            manifest = {
                "schema_version": SCHEMA_VERSION,
                "session": self.store.label,
                "episode": writer.episode_id,
                "created_at": utc_now(),
                "duration_seconds": duration,
                "hostname": socket.gethostname(),
                "recording_id": recording_id,
                "canonical_action": {
                    "description": "post-ramp absolute commanded EE box position",
                    "order": ["left_x", "left_y", "right_x", "right_y"],
                    "source": "robot.ndjson.commanded_box_xy",
                },
                "git": {
                    "franka_teach": git_state(self.repo_root),
                    "camera_api": git_state(self.camera_repo_root),
                },
                "nuc": self.bridge.status.get("host"),
                "camera_profile": str(self.camera_profile_path),
                "camera_profile_sha256": sha256_file(self.camera_profile_path),
                "start_discovery_mono_ns": start_discovery_ns,
                "validation": report.to_dict(),
            }
            writer.write_json("manifest.json", manifest)
            write_checksums(
                writer.path,
                [
                    "video.mov",
                    "camera.json",
                    "robot.ndjson",
                    "keys.ndjson",
                    "clocks.json",
                    "frames.csv",
                    "manifest.json",
                ],
            )
            destination = writer.finalize(report.accepted)

            # A capture-quality rejection is safe to remove from the phone once
            # the local file is complete and decodable. A corrupt local copy stays
            # on the phone for a future resume/retry.
            locally_verified = size_matches and "video_decode" not in report.failures
            phone_deleted = False
            phone_delete_error = None
            if locally_verified:
                try:
                    await asyncio.to_thread(self.camera.delete, recording_id)
                    phone_deleted = True
                except Exception as exc:
                    phone_delete_error = f"{type(exc).__name__}: {exc}"
            self.store.audit(
                "episode_finalized",
                episode=writer.episode_id,
                accepted=report.accepted,
                path=str(destination),
                failures=report.failures,
                phone_deleted=phone_deleted,
                phone_delete_error=phone_delete_error,
            )
            self.last_result = {
                "episode": writer.episode_id,
                "accepted": report.accepted,
                "path": str(destination),
                "failures": report.failures,
                "phone_deleted": phone_deleted,
                "phone_delete_error": phone_delete_error,
            }
            self.state = "idle"
            self.message = (
                "episode accepted" if report.accepted else "episode quarantined"
            )
            return self.last_result
        except Exception as exc:
            self.writer = None
            writer.close_streams()
            # Leave .partial and the phone copy intact for diagnosis/resume unless
            # this was an explicit manual abort handled above.
            self.store.audit(
                "episode_error",
                episode=writer.episode_id,
                recording_id=recording_id,
                reason=f"{type(exc).__name__}: {exc}",
                partial_path=str(writer.path),
            )
            self.state = "error"
            self.message = f"{type(exc).__name__}: {exc}"
            self.last_result = {
                "episode": writer.episode_id,
                "outcome": "error",
                "partial_path": str(writer.path),
                "error": self.message,
            }
            raise
