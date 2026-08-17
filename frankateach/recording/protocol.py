"""Wire and on-disk schema shared by Discovery and the robot NUC."""

import math
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

PROTOCOL_VERSION = 1
# 2: episodes carry actions.csv, the robot stream resampled onto video frames.
# 3: manifest git key "camera_api" is now "anycamera" (the app was renamed).
SCHEMA_VERSION = 3
ARMS = ("left", "right")


class ProtocolError(ValueError):
    pass


def jsonable(value):
    """Recursively convert numpy/dataclass/path values for JSON serialization."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def require_message(message, expected=None):
    """Validate the common envelope and return its type."""
    if not isinstance(message, dict):
        raise ProtocolError("message must be a JSON object")
    kind = message.get("t")
    if not isinstance(kind, str) or not kind:
        raise ProtocolError("message has no string 't' field")
    if expected is not None and kind not in expected:
        raise ProtocolError(f"unexpected message type {kind!r}")
    return kind


def validate_hello(message):
    require_message(message, {"hello"})
    if int(message.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol mismatch: peer={message.get('protocol_version')} "
            f"local={PROTOCOL_VERSION}"
        )
    session = str(message.get("session") or "").strip()
    if not session:
        raise ProtocolError("hello requires a session")
    return session


def validate_keys(message):
    require_message(message, {"keys"})
    try:
        sequence = int(message["sequence"])
        discovery_mono_ns = int(message["discovery_mono_ns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("keys requires integer sequence and discovery_mono_ns") from exc
    held = message.get("held")
    if not isinstance(held, list) or not all(isinstance(k, str) for k in held):
        raise ProtocolError("keys.held must be a list of KeyboardEvent.code strings")
    return {
        "sequence": sequence,
        "discovery_mono_ns": discovery_mono_ns,
        "browser_mono_ms": message.get("browser_mono_ms"),
        "held": sorted(set(held)),
        "frozen": bool(message.get("frozen", True)),
        "home": bool(message.get("home", False)),
    }
