from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import (
    ConfigValidationError,
    config_path,
    deterministic_config_hash,
    load_experiment_config,
    validate_experiment_config,
)
from src.config.experiment import FIELD_RULES, MODEL_IDS
from src.utils.serialization import read_yaml

ROOT = Path(__file__).resolve().parents[1]
LEGACY_FIELDS = set(FIELD_RULES) - {"schema_version", "execution_mode"}


def test_every_checked_in_experiment_config_validates() -> None:
    paths = sorted((ROOT / "configs" / "legacy").glob("*.yaml")) + sorted(
        (ROOT / "configs" / "smoke").glob("*.yaml")
    )
    assert len(paths) == len(MODEL_IDS) * 2
    for path in paths:
        config = load_experiment_config(path)
        assert path == config_path(
            ROOT,
            config["execution_mode"],
            config["model_id"],
            config["dataset_track"],
        )


@pytest.mark.parametrize("model_id", sorted(MODEL_IDS))
def test_legacy_config_preserves_existing_model_track_constants(model_id: str) -> None:
    existing = read_yaml(ROOT / "configs" / model_id / "2class.yaml")
    migrated = load_experiment_config(config_path(ROOT, "legacy", model_id))
    assert {key: migrated[key] for key in LEGACY_FIELDS} == existing


def test_hash_is_semantic_and_deterministic() -> None:
    config = load_experiment_config(
        config_path(ROOT, "legacy", "faster_rcnn_resnet50")
    )
    reversed_config = dict(reversed(list(config.items())))
    assert deterministic_config_hash(config) == deterministic_config_hash(
        reversed_config
    )
    changed = dict(config, epochs=config["epochs"] + 1)
    assert deterministic_config_hash(config) != deterministic_config_hash(changed)
    assert len(deterministic_config_hash(config)) == 64


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"surprise": True}, "unknown fields"),
        ({"epochs": 0}, "epochs must be at least 1"),
        ({"learning_rate": 0}, "learning_rate must be greater than 0"),
        ({"schema_version": 2}, "unsupported schema_version"),
        ({"image_size": True}, "image_size must not be boolean"),
    ],
)
def test_invalid_or_unknown_fields_are_rejected(
    change: dict[str, object], message: str
) -> None:
    config = load_experiment_config(
        config_path(ROOT, "legacy", "faster_rcnn_resnet50")
    )
    with pytest.raises(ConfigValidationError, match=message):
        validate_experiment_config(dict(config, **change))


def test_runtime_contract_matches_portable_json_schema() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "experiment_config" / "v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(FIELD_RULES)
    assert set(schema["properties"]) == set(FIELD_RULES)
    assert set(schema["properties"]["model_id"]["enum"]) == MODEL_IDS
