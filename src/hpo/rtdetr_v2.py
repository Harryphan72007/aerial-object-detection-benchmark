"""Persistent, isolated RT-DETR Optuna search protocol v2."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from src.optional_outputs import run_optional_output
from src.hpo.workflow import TwoStageRandomHPO, _failure_kind
from src.models.rtdetrv2.optimizer import checked_in_recipe
from src.utils.serialization import read_yaml

RTDETR_HPO_PROTOCOL_ID = "rtdetr_optuna_v2"
STUDY_METADATA_SCHEMA_VERSION = 2


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_search_config(repo_root: str | Path) -> dict[str, Any]:
    config = read_yaml(
        Path(repo_root) / "configs" / "rtdetrv2_l" / "hpo_recipe_v2.yaml"
    )
    if config.get("schema_version") != STUDY_METADATA_SCHEMA_VERSION:
        raise ValueError("unsupported RT-DETR HPO config schema")
    if config.get("protocol_id") != RTDETR_HPO_PROTOCOL_ID:
        raise ValueError("RT-DETR HPO protocol id does not match recipe v2")
    space = config.get("search_space")
    expected = {
        "detector_learning_rate",
        "backbone_lr_multiplier",
        "weight_decay",
        "warmup_steps",
        "gradient_clip_norm",
    }
    if not isinstance(space, dict) or set(space) != expected:
        raise ValueError("RT-DETR HPO v2 search fields do not match the contract")
    return config


def validate_study_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(value)
    required = {
        "schema_version",
        "model_id",
        "dataset_track",
        "protocol_id",
        "search_seed",
        "dataset_hashes",
        "source_commit",
        "environment_fingerprint",
        "search_space_hash",
        "recipe_hash",
        "objective",
        "storage_policy",
    }
    if set(metadata) != required:
        raise ValueError("RT-DETR study metadata fields do not match schema v2")
    if metadata["schema_version"] != STUDY_METADATA_SCHEMA_VERSION:
        raise ValueError("unsupported RT-DETR study metadata schema")
    if metadata["protocol_id"] != RTDETR_HPO_PROTOCOL_ID:
        raise ValueError("incorrect RT-DETR study protocol")
    return metadata


def classify_trial_failure(error: BaseException) -> str:
    return _failure_kind(error) or "unexpected"


def snapshot_sqlite_database(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    source_connection = sqlite3.connect(source_path)
    target_connection = sqlite3.connect(temporary)
    try:
        source_connection.backup(target_connection)
        target_connection.commit()
    finally:
        target_connection.close()
        source_connection.close()
    os.replace(temporary, target)
    return target


class RTDetrOptunaV2(TwoStageRandomHPO):
    """Five-parameter recipe-v2 search in a new persistent study namespace."""

    def __init__(self, repo_root: str | Path, drive_root: str | Path, dataset_track: str, **kwargs: Any) -> None:
        super().__init__(
            repo_root,
            drive_root,
            "rtdetrv2_l",
            dataset_track,
            protocol_id=RTDETR_HPO_PROTOCOL_ID,
            **kwargs,
        )
        self.search_config = load_search_config(self.repo_root)

    @property
    def study_path(self) -> Path:
        filename = "study_v2_smoke.db" if self.smoke_test else "study_v2.db"
        return self.root / filename

    def _broad_search_space(self) -> dict[str, dict[str, Any]]:
        return dict(self.search_config["search_space"])

    def _phase_b_search_space(
        self,
        broad: dict[str, dict[str, Any]],
        strongest: list[Any],
    ) -> dict[str, dict[str, Any]]:
        return broad

    def _metadata(
        self,
        split_summary: dict[str, Any],
        search_space: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        base = super()._metadata(split_summary, search_space)
        metadata = {
            **base,
            "schema_version": STUDY_METADATA_SCHEMA_VERSION,
            "recipe_hash": _stable_hash(checked_in_recipe(self.repo_root)),
            "storage_policy": {
                "database": "study_v2.db",
                "snapshot": "snapshots/study_v2_latest.db",
                "trial_weights": "local_scratch_deleted",
                "old_study_mutated": False,
            },
        }
        return validate_study_metadata(metadata)

    def _after_trial(self, study: Any) -> None:
        if self.study_path.is_file():
            run_optional_output(
                "snapshot_optuna_database",
                self.root,
                lambda: snapshot_sqlite_database(
                    self.study_path,
                    self.root / "snapshots" / "study_v2_latest.db",
                ),
            )

    def _classify_failure(self, error: BaseException) -> str | None:
        kind = classify_trial_failure(error)
        return None if kind == "unexpected" else kind
