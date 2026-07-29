"""Read/write configs/airhockey.yaml without disturbing the parts we don't own.

Calibration writes one arm's block at a time, so this uses a plain yaml
round-trip rather than hydra/OmegaConf -- we need to read the file, replace a
subtree, and write it back while keeping the other arm intact.
"""

from pathlib import Path

import yaml

from frankateach.airhockey.box import PlayBox

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "airhockey.yaml"


def load(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def load_box(arm, path=CONFIG_PATH):
    """Build a PlayBox for one arm, or raise if it has not been calibrated."""
    cfg = load(path)
    arm_cfg = (cfg.get("arms") or {}).get(arm)
    if not arm_cfg or arm_cfg.get("center") is None:
        raise RuntimeError(
            f"The {arm!r} arm has no calibration in {path}. Run:\n"
            f"    python3 airhockey.py --calibrate --arm {arm}"
        )
    return PlayBox.from_dict(arm_cfg)


def save_box(arm, box: PlayBox, path=CONFIG_PATH):
    """Merge one arm's calibration into the config, leaving everything else alone."""
    cfg = load(path)
    cfg.setdefault("arms", {}).setdefault(arm, {}).update(box.to_dict())
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return path
