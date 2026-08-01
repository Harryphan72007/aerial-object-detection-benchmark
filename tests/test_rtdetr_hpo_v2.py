from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.hpo.rtdetr_v2 import (
    RTDETR_HPO_PROTOCOL_ID,
    RTDetrOptunaV2,
    classify_trial_failure,
    load_search_config,
    snapshot_sqlite_database,
    validate_study_metadata,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v2_search_config_has_all_five_parameters() -> None:
    config = load_search_config(ROOT)
    assert set(config["search_space"]) == {
        "detector_learning_rate",
        "backbone_lr_multiplier",
        "weight_decay",
        "warmup_epochs",
        "gradient_clip_norm",
    }


def test_v2_storage_is_persistent_and_legacy_database_is_untouched(tmp_path: Path) -> None:
    optuna = pytest.importorskip("optuna")
    legacy = (
        tmp_path
        / "hpo"
        / "two_stage_random_hpo_v1"
        / "rtdetrv2_l"
        / "2class"
        / "study.db"
    )
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy-sentinel")
    workflow = RTDetrOptunaV2(ROOT, tmp_path, "2class")
    metadata = workflow._metadata(
        {"hashes": {"train": "a", "validation": "b"}},
        workflow._broad_search_space(),
    )
    first = workflow._study(metadata)
    first.optimize(lambda trial: (trial.suggest_float("x", 0, 1), 0.5), n_trials=1)
    reconnected = workflow._study(metadata)
    reconnected.optimize(lambda trial: (trial.suggest_float("x", 0, 1), 0.6), n_trials=1)
    assert [trial.number for trial in reconnected.trials] == [0, 1]
    assert workflow.study_path.name == "study_v2.db"
    assert legacy.read_bytes() == b"legacy-sentinel"


def test_snapshot_is_a_valid_independent_sqlite_copy(tmp_path: Path) -> None:
    source = tmp_path / "study.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE trials (value INTEGER)")
        connection.execute("INSERT INTO trials VALUES (1)")
    snapshot = snapshot_sqlite_database(source, tmp_path / "snapshots" / "latest.db")
    with sqlite3.connect(snapshot) as connection:
        assert connection.execute("SELECT value FROM trials").fetchall() == [(1,)]


def test_failure_policy_prunes_numerical_and_propagates_unexpected() -> None:
    assert classify_trial_failure(ValueError("loss is NaN")) == "numerical_divergence"
    assert classify_trial_failure(RuntimeError("CUDA out of memory")) == "out_of_memory"
    assert classify_trial_failure(RuntimeError("broken implementation")) == "unexpected"


def test_metadata_schema_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="fields"):
        validate_study_metadata(
            {"schema_version": 2, "protocol_id": RTDETR_HPO_PROTOCOL_ID}
        )
