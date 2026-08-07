"""Strict validation and frame-index generation for completed raw episodes."""

import csv
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from frankateach.recording.clock import ClockFit, percentile_ns


@dataclass
class ValidationReport:
    failures: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def accepted(self):
        return not self.failures

    def fail(self, reason):
        if reason not in self.failures:
            self.failures.append(reason)

    def to_dict(self):
        return {"accepted": self.accepted, "failures": self.failures, "details": self.details}


def probe_video(path):
    decode = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if decode.returncode:
        raise RuntimeError(decode.stderr.strip() or "ffmpeg decode failed")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_packets",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,duration,nb_frames:"
        "packet=pts_time,flags",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    document = json.loads(result.stdout)
    streams = document.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe found no video stream")
    packets = []
    for decode_order, raw in enumerate(document.get("packets") or []):
        pts = raw.get("pts_time")
        if pts is not None:
            packets.append(
                {
                    "pts_seconds": float(pts),
                    "key_frame": "K" in str(raw.get("flags", "")),
                    "decode_order": decode_order,
                }
            )
    packets.sort(key=lambda packet: packet["pts_seconds"])
    return streams[0], packets


def _metadata_timing(camera):
    return camera.get("timing") or {}


def validate_episode(
    video_path,
    camera,
    telemetry,
    camera_fit: ClockFit,
    nuc_fit: ClockFit,
    *,
    expected_duration,
    expected_width=1920,
    expected_height=1080,
    expected_fps=60.0,
    expected_gop=12,
    expected_frame_reordering=False,
    minimum_robot_hz=57.0,
    maximum_gap_ms=50.0,
    maximum_sync_ms=2.0,
    maximum_rtt_ms=20.0,
):
    report = ValidationReport()
    timing = _metadata_timing(camera)
    for field in ("captureDrops", "writerBackpressureDrops", "appendFailures"):
        value = int(timing.get(field, camera.get(field, 0)) or 0)
        report.details[field] = value
        if value:
            report.fail(f"camera_{field}")
    interruptions = timing.get("interruptions") or []
    report.details["interruptions"] = interruptions
    if interruptions:
        report.fail("camera_interruption")

    if int(camera.get("width", 0)) != expected_width or int(
        camera.get("height", 0)
    ) != expected_height:
        report.fail("camera_resolution")
    if abs(float(camera.get("fps", 0)) - expected_fps) > 0.01:
        report.fail("camera_fps")
    if str(camera.get("codec", "")).lower() not in {"h264", "avc1"}:
        report.fail("camera_codec")
    if int(timing.get("keyFrameInterval", 0)) != expected_gop:
        report.fail("camera_gop")

    try:
        stream, frames = probe_video(video_path)
    except Exception as exc:
        report.details["ffprobe_error"] = str(exc)
        report.fail("video_decode")
        return report, []

    report.details["video_frame_count"] = len(frames)
    decode_order = [
        int(frame.get("decode_order", index)) for index, frame in enumerate(frames)
    ]
    reordered = decode_order != sorted(decode_order)
    report.details["video_frame_reordering"] = reordered
    if reordered and not expected_frame_reordering:
        report.fail("video_frame_reordering")
    if int(stream.get("width", 0)) != expected_width or int(
        stream.get("height", 0)
    ) != expected_height:
        report.fail("video_resolution")
    if stream.get("codec_name") != "h264":
        report.fail("video_codec")
    try:
        numerator, denominator = str(stream.get("avg_frame_rate", "0/1")).split(
            "/", 1
        )
        video_fps = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        video_fps = 0.0
    report.details["video_fps"] = video_fps
    if abs(video_fps - expected_fps) > 0.01:
        report.fail("video_fps")
    pts = np.asarray([frame["pts_seconds"] for frame in frames], dtype=np.float64)
    if len(pts) < 2 or np.any(np.diff(pts) <= 0):
        report.fail("video_pts")
    elif np.any(np.diff(pts) > 1.5 / expected_fps):
        report.fail("video_frame_gap")
    if frames:
        duration = frames[-1]["pts_seconds"] - frames[0]["pts_seconds"] + 1.0 / expected_fps
        report.details["video_duration_seconds"] = duration
        if abs(duration - expected_duration) > 2.0 / expected_fps:
            report.fail("video_duration")
    written = camera.get("framesWritten")
    if written is not None and int(written) != len(frames):
        report.fail("video_frame_count")
    key_indices = [i for i, frame in enumerate(frames) if frame["key_frame"]]
    if not key_indices or key_indices[0] != 0:
        report.fail("video_first_keyframe")
    elif any(b - a > expected_gop for a, b in zip(key_indices, key_indices[1:])):
        report.fail("video_keyframe_spacing")

    max_sync_ns = int(maximum_sync_ms * 1e6)
    report.details["camera_sync_uncertainty_ns"] = camera_fit.uncertainty_ns
    report.details["nuc_sync_uncertainty_ns"] = nuc_fit.uncertainty_ns
    if camera_fit.uncertainty_ns > max_sync_ns:
        report.fail("camera_clock_uncertainty")
    if nuc_fit.uncertainty_ns > max_sync_ns:
        report.fail("nuc_clock_uncertainty")

    rtts = telemetry.get("bridge_rtt_ns") or []
    p99 = percentile_ns(rtts, 99)
    report.details["bridge_p99_rtt_ns"] = p99
    if p99 is None or p99 > int(maximum_rtt_ms * 1e6):
        report.fail("bridge_rtt")

    by_arm = telemetry.get("by_arm") or {}
    for arm in ("left", "right"):
        rows = list(by_arm.get(arm) or [])
        if len(rows) < 2:
            report.fail(f"{arm}_telemetry_missing")
            continue
        seq = [int(row["tick_sequence"]) for row in rows]
        if any(b != a + 1 for a, b in zip(seq, seq[1:])):
            report.fail(f"{arm}_telemetry_sequence")
        command_ns = np.asarray([int(row["command_mono_ns"]) for row in rows], dtype=np.int64)
        span = max(1, int(command_ns[-1] - command_ns[0]))
        rate = (len(command_ns) - 1) * 1e9 / span
        gaps = np.diff(command_ns)
        report.details[f"{arm}_rate_hz"] = rate
        report.details[f"{arm}_max_gap_ns"] = int(np.max(gaps)) if len(gaps) else None
        if rate < minimum_robot_hz:
            report.fail(f"{arm}_control_rate")
        if len(gaps) and int(np.max(gaps)) > int(maximum_gap_ms * 1e6):
            report.fail(f"{arm}_command_gap")
        drop_counts = [int(row.get("telemetry_drops", 0)) for row in rows]
        if drop_counts and max(drop_counts) > drop_counts[0]:
            report.fail(f"{arm}_telemetry_drop")
        if any(not row.get("connected", False) for row in rows):
            report.fail(f"{arm}_disconnected")
        if any(row.get("provisional", False) for row in rows):
            report.fail(f"{arm}_provisional")
        if any(row.get("error") for row in rows):
            report.fail(f"{arm}_error")

    first_pts = timing.get("firstVideoPTSSeconds")
    frame_index = []
    if first_pts is None:
        report.fail("camera_first_video_pts")
    else:
        for index, frame in enumerate(frames):
            camera_pts_ns = int(round((float(first_pts) + frame["pts_seconds"]) * 1e9))
            frame_index.append(
                {
                    "frame": index,
                    "file_pts_seconds": frame["pts_seconds"],
                    "camera_pts_seconds": camera_pts_ns / 1e9,
                    "discovery_mono_ns": camera_fit.map_ns(camera_pts_ns),
                }
            )
    return report, frame_index


def write_frame_index(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["frame", "file_pts_seconds", "camera_pts_seconds", "discovery_mono_ns"],
        )
        writer.writeheader()
        writer.writerows(rows)
