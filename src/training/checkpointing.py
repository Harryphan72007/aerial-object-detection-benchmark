"""Run manifests and atomic experiment registry."""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from filelock import FileLock

from src.paths import ProjectPaths
from src.utils.serialization import read_json, write_json

LEGACY_MANIFEST_REQUIRED = {
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
# Public compatibility name for the frozen v1 contract.
MANIFEST_REQUIRED = LEGACY_MANIFEST_REQUIRED

MANIFEST_REQUIRED_V2 = (
    LEGACY_MANIFEST_REQUIRED
    - {"checkpoint_best_map", "checkpoint_best_aptiny", "checkpoint_last"}
) | {
    "schema_version",
    "checkpoint_best",
    "checkpoint_sha256",
    "checkpoint_selection_metric",
    "weight_variant",
}

MODEL_CHECKPOINT_SUFFIXES = frozenset({".pth", ".pt", ".ckpt"})


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
    required = (
        MANIFEST_REQUIRED_V2
        if int(manifest.get("schema_version", 1)) >= 2
        else LEGACY_MANIFEST_REQUIRED
    )
    errors = [
        f"missing field: {key}" for key in sorted(required - set(manifest))
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
    if check_files and int(manifest.get("schema_version", 1)) >= 2:
        value = manifest.get("checkpoint_best")
        if manifest.get("status") == "completed" and (
            not value or not Path(str(value)).is_file()
        ):
            errors.append(f"checkpoint_best does not exist: {value}")
    elif check_files:
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
            registry["schema_version"] = max(
                int(registry["schema_version"]),
                int(manifest.get("schema_version", 1)),
            )
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
        self, run_id: str, preference: str = "best"
    ) -> Path:
        manifest = self._load().get("runs", {}).get(run_id)
        if not manifest:
            raise KeyError(f"run not found: {run_id}")
        if preference not in {"best", "best_map", "best_aptiny", "last"}:
            raise ValueError(f"unsupported checkpoint preference: {preference}")
        return resolve_manifest_checkpoint(
            manifest,
            allow_resume=preference == "last",
            allow_legacy_aliases=True,
        )

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
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
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


def model_checkpoint_files(run_dir: str | Path) -> list[Path]:
    """Return only direct child model files; never traverse outside a run."""
    root = Path(run_dir).resolve()
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in MODEL_CHECKPOINT_SUFFIXES
    )


def resolve_manifest_checkpoint(
    manifest: Mapping[str, Any],
    run_dir: str | Path | None = None,
    *,
    allow_resume: bool = False,
    allow_legacy_aliases: bool = False,
) -> Path:
    """Resolve a canonical or legacy checkpoint without filename guessing."""
    root_value = run_dir or manifest.get("run_dir")
    root = Path(str(root_value)).resolve() if root_value else None
    candidates: list[Path] = []

    def add(value: Any) -> None:
        if value:
            candidate = Path(str(value))
            if not candidate.is_absolute() and root is not None:
                candidate = root / candidate
            if candidate not in candidates:
                candidates.append(candidate)

    if root is not None:
        add(root / "best.pth")
    add(manifest.get("checkpoint_best"))
    if root is not None:
        add(root / "best_map.pth")
    add(manifest.get("checkpoint_best_map"))
    if root is not None:
        add(root / "best_raw.pth")
    add(manifest.get("checkpoint_best_raw"))
    if allow_legacy_aliases:
        if root is not None:
            add(root / "best.pt")
            add(root / "best_aptiny.pth")
        add(manifest.get("checkpoint_best_aptiny"))
    if allow_resume:
        add(manifest.get("checkpoint_resume"))
        add(manifest.get("checkpoint_last"))
        if root is not None:
            add(root / "last.pth")
            if allow_legacy_aliases:
                add(root / "latest.pt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = ", ".join(str(path) for path in candidates) or "<none>"
    raise FileNotFoundError(f"no compatible checkpoint found; checked: {rendered}")


def _torch_checkpoint_loader(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to validate checkpoints") from exc
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    return value


def validate_checkpoint_identity(
    checkpoint: str | Path,
    expected: Mapping[str, Any] | None = None,
    *,
    loader: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and validate the v2 identity when one is expected."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = dict((loader or _torch_checkpoint_loader)(path))
    identity = payload.get("checkpoint_identity")
    if expected is not None:
        if not isinstance(identity, Mapping):
            raise ValueError("checkpoint is missing checkpoint_identity")
        mismatches = {
            key: (identity.get(key), value)
            for key, value in expected.items()
            if identity.get(key) != value
        }
        if mismatches:
            raise ValueError(f"checkpoint identity mismatch: {mismatches}")
    return payload


def materialize_canonical_best(
    resume_checkpoint: str | Path,
    destination: str | Path,
    identity: Mapping[str, Any],
    *,
    loader: Callable[[Path], Mapping[str, Any]] | None = None,
    saver: Callable[[Mapping[str, Any], Path], None] | None = None,
) -> tuple[Path, str]:
    """Extract the selected weights from one rolling resume checkpoint."""
    source = Path(resume_checkpoint)
    state = dict((loader or _torch_checkpoint_loader)(source))
    selected = state.get("best_model_state_dict")
    if selected is None:
        selected = state.get("model_state_dict", state.get("model", state.get("state_dict")))
    if selected is None:
        raise ValueError("resume checkpoint contains no selected model state")
    output: dict[str, Any] = {
        "checkpoint_schema_version": 2,
        "checkpoint_identity": dict(identity),
        "weight_variant": identity.get("weight_variant", "raw"),
        "selection_metric": identity.get("selection_metric"),
        "selection_metric_value": identity.get("selection_metric_value"),
    }
    if "state_dict" in state and "model" not in state:
        output["state_dict"] = selected
        if isinstance(state.get("meta"), Mapping):
            output["meta"] = dict(state["meta"])
    else:
        output["model"] = selected
        output["model_state_dict"] = selected
        for key in ("id2label", "model_name", "config"):
            if key in state:
                output[key] = state[key]
    target = Path(destination)
    if saver is None:
        atomic_torch_save(output, target)
    else:
        saver(output, target)
    validate_checkpoint_identity(target, identity, loader=loader)
    checksum = hashlib.sha256(target.read_bytes()).hexdigest()
    return target, checksum


def enforce_completed_checkpoint_policy(run_dir: str | Path) -> list[str]:
    """Remove duplicate model files only after a validated best.pth exists."""
    root = Path(run_dir).resolve()
    canonical = root / "best.pth"
    if not canonical.is_file():
        raise FileNotFoundError(canonical)
    removed: list[str] = []
    temporary_files = [
        path
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name == "last_checkpoint"
            or path.name.endswith((".pth.tmp", ".pt.tmp", ".ckpt.tmp"))
        )
    ]
    for path in temporary_files:
        if path.resolve().parent != root:
            raise ValueError(f"checkpoint cleanup escaped run directory: {path}")
        path.unlink()
        removed.append(path.name)
    files = [path for path in model_checkpoint_files(root) if path != canonical]
    # The resume file is deliberately removed last. Any earlier failure therefore
    # leaves the run resumable instead of half-cleaned without recovery state.
    files.sort(key=lambda path: path.name == "last.pth")
    for path in files:
        resolved = path.resolve()
        if resolved.parent != root:
            raise ValueError(f"checkpoint cleanup escaped run directory: {path}")
        path.unlink()
        removed.append(path.name)
    remaining = model_checkpoint_files(root)
    if remaining != [canonical]:
        raise RuntimeError(
            f"completed run checkpoint policy failed: {[p.name for p in remaining]}"
        )
    return removed
