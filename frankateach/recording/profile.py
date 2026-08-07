"""The isolated, internally consistent 60 Hz recording control profile."""

import copy
from pathlib import Path

import yaml

from frankateach.airhockey.supervisor import NUC_CONFIG_DIR, REPO_ROOT

PROFILE_NAME = "recording_60"
CONTROL_HZ = 60
NUM_STEPS = 1


def apply_recording_profile(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["control_hz"] = CONTROL_HZ
    for arm in ("left", "right"):
        arm_cfg = cfg.setdefault("arms", {}).setdefault(arm, {})
        arm_cfg["deoxys_config"] = f"deoxys_{arm}_60.yml"
        arm_cfg["nuc_config"] = f"franka_{arm}_60.yml"
    return cfg


def _policy_rate(path):
    with Path(path).open(encoding="utf-8") as stream:
        return int(yaml.safe_load(stream)["CONTROL"]["POLICY_RATE"])


def validate_recording_profile(cfg):
    failures = []
    if float(cfg.get("control_hz", 0)) != CONTROL_HZ:
        failures.append(f"control_hz={cfg.get('control_hz')} (expected {CONTROL_HZ})")
    for arm in ("left", "right"):
        arm_cfg = cfg.get("arms", {}).get(arm, {})
        client = REPO_ROOT / "frankateach" / "configs" / arm_cfg.get(
            "deoxys_config", ""
        )
        nuc = NUC_CONFIG_DIR / arm_cfg.get("nuc_config", "")
        for side, path in (("client", client), ("NUC", nuc)):
            if not path.is_file():
                failures.append(f"{arm} {side} config missing: {path}")
                continue
            try:
                rate = _policy_rate(path)
            except Exception as exc:
                failures.append(f"{arm} {side} config unreadable: {exc}")
                continue
            if rate != CONTROL_HZ:
                failures.append(f"{arm} {side} POLICY_RATE={rate} (expected {CONTROL_HZ})")
    if failures:
        raise RuntimeError("invalid recording_60 profile:\n  - " + "\n  - ".join(failures))
    return True
