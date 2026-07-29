"""Run manifests and atomic experiment registry."""
from __future__ import annotations

import csv
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock

from src.paths import ProjectPaths
from src.utils.serialization import read_json, write_json

MANIFEST_REQUIRED = {
    "run_id",
    "model_id",
    "architecture_family",
    "dataset_track",
    "class_names",
    "seed",
    "input_resolution",
    "checkpoint_best_map",
    "checkpoint_best_aptiny",
    "checkpoint_last",
    "config_path",
    "created_at",
    "framework",
    "framework_version",
    "pytorch_version",
    "cuda_version",
    "gpu_name",
    "total_parameters",
    "trainable_parameters",
    "frozen_parameters",
    "best_validation_map",
    "best_validation_aptiny",
    "best_epoch",
    "total_training_seconds",
    "status",
}


def make_run_id(
    model_id: str,
    dataset_track: str,
    resolution: int,
    seed: int,
    timestamp: datetime | None = None,
) -> str:
    stamp = (timestamp or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    return f"{model_id}__{dataset_track}__{resolution}__{stamp}__seed{seed}"


def initialize_run(
    paths: ProjectPaths,
    model_id: str,
    dataset_track: str,
    resolution: int,
    seed: int,
) -> tuple[str, Path]:
    run_id = make_run_id(model_id, dataset_track, resolution, seed)
    run_dir = paths.run_dir(model_id, run_id)
    for subdirectory in ("tensorboard", "logs"):
        (run_dir / subdirectory).mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def validate_manifest_dict(
    manifest: dict[str, Any], check_files: bool = False
) -> list[str]:
    errors = [
        f"missing field: {key}" for key in sorted(MANIFEST_REQUIRED - set(manifest))
    ]
    if manifest.get("dataset_track") not in {"2class", "10class"}:
        errors.append("invalid dataset_track")
    if manifest.get("status") not in {
        "created",
        "running",
        "completed",
        "failed",
        "interrupted",
    }:
        errors.append("invalid status")
    if check_files:
        for field in (
            "checkpoint_best_map",
            "checkpoint_best_aptiny",
            "checkpoint_last",
        ):
            value = manifest.get(field)
            if value and not Path(value).exists():
                errors.append(f"{field} does not exist: {value}")
    return errors


@dataclass
class RunRegistry:
    paths: ProjectPaths

    def _load(self) -> dict[str, Any]:
        if not self.paths.checkpoint_registry.exists():
            return {"schema_version": 1, "runs": {}}
        return read_json(self.paths.checkpoint_registry)

    def register_run(self, manifest_path: str | Path) -> dict[str, Any]:
        manifest = read_json(manifest_path)
        errors = validate_manifest_dict(manifest)
        if errors:
            raise ValueError("invalid run manifest:\n" + "\n".join(errors))
        lock = FileLock(str(self.paths.registry_dir / "registry.lock"), timeout=60)
        with lock:
            registry = self._load()
            registry.setdefault("schema_version", 1)
            registry.setdefault("runs", {})
            registry["runs"][manifest["run_id"]] = manifest
            if self.paths.checkpoint_registry.exists():
                shutil.copy2(
                    self.paths.checkpoint_registry,
                    self.paths.checkpoint_registry.with_suffix(".json.bak"),
                )
            write_json(self.paths.checkpoint_registry, registry, atomic=True)
            self._rewrite_csv(registry)
        return manifest

    def _rewrite_csv(self, registry: dict[str, Any]) -> None:
        rows = list(registry.get("runs", {}).values())
        fields = (
            sorted({key for row in rows for key in row})
            if rows
            else ["run_id", "model_id", "status"]
        )
        temporary = self.paths.runs_csv.with_suffix(".csv.tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.paths.runs_csv)

    def list_available_runs(
        self,
        model_id: str | None = None,
        dataset_track: str | None = None,
        status: str | None = "completed",
    ) -> list[dict[str, Any]]:
        runs = list(self._load().get("runs", {}).values())
        if model_id:
            runs = [run for run in runs if run.get("model_id") == model_id]
        if dataset_track:
            runs = [
                run for run in runs if run.get("dataset_track") == dataset_track
            ]
        if status:
            runs = [run for run in runs if run.get("status") == status]
        return sorted(
            runs, key=lambda run: run.get("created_at", ""), reverse=True
        )

    def get_best_run(
        self,
        model_id: str,
        metric: str = "best_validation_map",
        dataset_track: str | None = None,
    ) -> dict[str, Any]:
        runs = self.list_available_runs(model_id, dataset_track)
        if not runs:
            raise KeyError(
                f"no completed runs for {model_id} track={dataset_track}"
            )
        return max(runs, key=lambda run: float(run.get(metric, float("-inf"))))

    def load_checkpoint_from_registry(
        self, run_id: str, preference: str = "best_map"
    ) -> Path:
        manifest = self._load().get("runs", {}).get(run_id)
        if not manifest:
            raise KeyError(f"run not found: {run_id}")
        field = {
            "best_map": "checkpoint_best_map",
            "best_aptiny": "checkpoint_best_aptiny",
            "last": "checkpoint_last",
        }[preference]
        path = Path(manifest[field])
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def validate_checkpoint_manifest(self, run_id: str) -> list[str]:
        manifest = self._load().get("runs", {}).get(run_id)
        if manifest is None:
            return [f"run not found: {run_id}"]
        return validate_manifest_dict(manifest, check_files=True)


def materialize_checkpoint_alias(
    source: str | Path, destination: str | Path
) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    shutil.copy2(source_path, temporary)
    os.replace(temporary, destination_path)


def atomic_torch_save(value: Any, destination: str | Path) -> None:
    """Save a torch checkpoint atomically, retaining the previous valid file."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to save checkpoints") from exc
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        torch.save(value, name)
        with open(name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)
