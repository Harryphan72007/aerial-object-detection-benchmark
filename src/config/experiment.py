"""Strict v1 experiment configuration contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.utils.serialization import read_yaml

SCHEMA_VERSION = 1
MODEL_IDS = frozenset(
    {
        "faster_rcnn_resnet50",
        "faster_rcnn_swin_t",
        "faster_rcnn_vmamba_t",
        "rtdetrv2_l",
        "yolox_s",
    }
)
DATASET_TRACKS = frozenset({"2class", "10class"})
EXECUTION_MODES = frozenset({"legacy", "smoke"})

FIELD_RULES: dict[str, tuple[type | tuple[type, ...], float | None]] = {
    "schema_version": (int, None),
    "model_id": (str, None),
    "dataset_track": (str, None),
    "execution_mode": (str, None),
    "image_size": (int, 32),
    "batch_size": (int, 1),
    "gradient_accumulation_steps": (int, 1),
    "epochs": (int, 1),
    "seed": (int, 0),
    "use_amp": (bool, None),
    "max_detections_per_image": (int, 1),
    "learning_rate": ((int, float), 0),
    "weight_decay": ((int, float), 0),
}


class ConfigValidationError(ValueError):
    """Raised when an experiment config violates the versioned contract."""


def validate_experiment_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy a v1 experiment configuration.

    The runtime validator deliberately has no optional JSON Schema dependency;
    ``schemas/experiment_config/v1.json`` is the portable representation of the
    same contract for editors and external tooling.
    """

    config = dict(value)
    missing = sorted(set(FIELD_RULES) - set(config))
    unknown = sorted(set(config) - set(FIELD_RULES))
    if missing:
        raise ConfigValidationError(f"missing required fields: {missing}")
    if unknown:
        raise ConfigValidationError(f"unknown fields: {unknown}")

    for field, (expected_type, minimum) in FIELD_RULES.items():
        field_value = config[field]
        if isinstance(field_value, bool) and expected_type is not bool:
            raise ConfigValidationError(f"{field} must not be boolean")
        if not isinstance(field_value, expected_type):
            raise ConfigValidationError(
                f"{field} has invalid type {type(field_value).__name__}"
            )
        if minimum is not None and field_value < minimum:
            relation = "greater than" if field == "learning_rate" else "at least"
            raise ConfigValidationError(f"{field} must be {relation} {minimum}")
        if field == "learning_rate" and field_value == 0:
            raise ConfigValidationError("learning_rate must be greater than 0")

    if config["schema_version"] != SCHEMA_VERSION:
        raise ConfigValidationError(
            f"unsupported schema_version {config['schema_version']!r}"
        )
    if config["model_id"] not in MODEL_IDS:
        raise ConfigValidationError(f"unknown model_id {config['model_id']!r}")
    if config["dataset_track"] not in DATASET_TRACKS:
        raise ConfigValidationError(
            f"unknown dataset_track {config['dataset_track']!r}"
        )
    if config["execution_mode"] not in EXECUTION_MODES:
        raise ConfigValidationError(
            f"unknown execution_mode {config['execution_mode']!r}"
        )
    return config


def deterministic_config_hash(value: Mapping[str, Any]) -> str:
    """Hash the semantic config independent of YAML layout or key order."""

    config = validate_experiment_config(value)
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def config_path(
    repo_root: str | Path,
    execution_mode: str,
    model_id: str,
    dataset_track: str = "2class",
) -> Path:
    """Resolve a checked-in legacy or smoke configuration path."""

    if execution_mode not in EXECUTION_MODES:
        raise ConfigValidationError(f"unknown execution_mode {execution_mode!r}")
    if model_id not in MODEL_IDS:
        raise ConfigValidationError(f"unknown model_id {model_id!r}")
    if dataset_track not in DATASET_TRACKS:
        raise ConfigValidationError(f"unknown dataset_track {dataset_track!r}")
    return (
        Path(repo_root)
        / "configs"
        / execution_mode
        / f"{model_id}_{dataset_track}.yaml"
    )


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load and validate an experiment YAML file."""

    return validate_experiment_config(read_yaml(path))
