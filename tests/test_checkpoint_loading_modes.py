from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.compatibility.checkpoints import (
    CheckpointLoadingMode,
    classify_checkpoint,
    require_loading_mode,
)

ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict[str, object]:
    return {
        "schema_version": 2,
        "model_id": "rtdetrv2_l",
        "model_signature": "1" * 64,
        "config_hash": "2" * 64,
        "dataset_hash": "3" * 64,
        "optimizer_signature": "adamw-v1",
        "scheduler_signature": "warmup-cosine-v1",
        "accumulation_steps": 4,
        "seed": 42,
        "state_keys": ["model", "optimizer", "scheduler", "training_state", "scaler"],
    }


def test_exact_v2_contract_is_full_resume() -> None:
    result = classify_checkpoint(_contract(), _contract())
    assert result.mode is CheckpointLoadingMode.FULL_RESUME
    require_loading_mode(result, CheckpointLoadingMode.FULL_RESUME)


def test_run_signature_drift_is_weights_only() -> None:
    changed = dict(_contract(), config_hash="4" * 64, accumulation_steps=8)
    result = classify_checkpoint(changed, _contract())
    assert result.mode is CheckpointLoadingMode.WEIGHTS_ONLY
    assert result.reasons == ("config_hash mismatch", "accumulation_steps mismatch")


def test_legacy_checkpoint_is_evaluation_only() -> None:
    result = classify_checkpoint(None, _contract())
    assert result.mode is CheckpointLoadingMode.EVALUATION_ONLY
    with pytest.raises(ValueError, match="evaluation_only"):
        require_loading_mode(result, CheckpointLoadingMode.FULL_RESUME)


def test_model_signature_drift_is_incompatible() -> None:
    changed = dict(_contract(), model_signature="9" * 64)
    result = classify_checkpoint(changed, _contract())
    assert result.mode is CheckpointLoadingMode.INCOMPATIBLE


def test_schema_fields_match_runtime_contract() -> None:
    schema = json.loads((ROOT / "schemas" / "checkpoint" / "v2.json").read_text())
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(_contract())
