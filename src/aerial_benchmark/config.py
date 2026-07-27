from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    parent_name = payload.pop("inherits", None)
    if not parent_name:
        return payload
    parent = load_config(config_path.parent / parent_name)
    return deep_merge(parent, payload)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_config(config: dict[str, Any]) -> None:
    required = ("dataset", "protocol", "evaluation", "model")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")
    if config["model"].get("checkpoint") in (None, ""):
        raise ValueError("Model checkpoint must be explicit or set to the literal 'TBD'")
