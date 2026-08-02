"""Validated, portable export of one lightweight benchmark result bundle."""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.benchmark_status import find_selected_config
from src.models.registry import MODEL_CONFIGS
from src.paths import ProjectPaths
from src.training.checkpointing import RunRegistry, resolve_manifest_checkpoint
from src.utils.serialization import (
    read_json,
    read_yaml,
    sha256_file,
    write_json,
    write_yaml,
)

REQUIRED_BUNDLE_FILES = {
    "bundle_manifest.json",
    "README.md",
    "configs/selected_lr.yaml",
    "configs/final_resolved_config.yaml",
    "search/candidates.csv",
    "search/promotion_history.csv",
    "search/search_summary.json",
    "metrics/final_metrics.json",
    "metrics/per_class_metrics.csv",
    "metrics/profiling_summary.json",
    "reports/model_report.md",
    "provenance/environment_summary.json",
    "provenance/dataset_hashes.json",
    "provenance/git_commit.txt",
}
REQUIRED_BUNDLE_DIRECTORIES = {
    "configs", "search", "metrics", "reports", "reports/figures", "provenance"
}
HPO_REQUIRED_BUNDLE_FILES = {
    "bundle_manifest.json",
    "README.md",
    "configs/best_config.yaml",
    "search/search_summary.json",
    "metrics/comparison.json",
    "reports/model_report.md",
    "provenance/environment_summary.json",
    "provenance/dataset_hashes.json",
    "provenance/git_commit.txt",
}
REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "result_bundle_id",
    "created_at",
    "model_id",
    "architecture_family",
    "dataset_track",
    "class_names",
    "run_id",
    "seed",
    "seed_status",
    "selected_learning_rate",
    "checkpoint_sha256",
    "annotation_sha256",
    "official_full_train_verified",
    "evaluation_git_commit",
    "training_git_commit",
    "generated_files",
    "intentionally_excluded_files",
    "export_status",
}
APPROVED_EXTENSIONS = {
    ".csv", ".json", ".yaml", ".yml", ".md", ".html", ".png", ".jpg",
    ".jpeg", ".txt",
}
EXCLUDED_EXTENSIONS = {
    ".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".engine", ".weights",
    ".db", ".zip", ".tar", ".gz", ".npz", ".npy",
}
EXCLUDED_PARTS = {
    "datasets", "predictions", "raw_predictions", "tensorboard", "credentials",
    "checkpoints", "optuna",
}
SECRET_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,]{8,}"),
)
ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:[\\/]|/content/drive/|/mnt/|/home/)[^\s\"']+"
)
METRIC_RANGES = {
    "ap": (0.0, 1.0),
    "map": (0.0, 1.0),
    "map50": (0.0, 1.0),
    "map75": (0.0, 1.0),
    "aptiny": (0.0, 1.0),
    "precision": (0.0, 1.0),
    "recall": (0.0, 1.0),
    "f1": (0.0, 1.0),
}
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,40}$", re.I)
PLACEHOLDER_PATTERN = re.compile(
    r"(?i)\b(todo|tbd|replace[_ -]?me|dummy|fabricated|example metric)\b"
)


def sanitize_text(value: str) -> str:
    """Remove private machine paths while preserving portable references."""
    return ABSOLUTE_PATH_PATTERN.sub("<portable-path>", value.replace("\\", "/"))


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def find_secret_like_content(text: str) -> list[str]:
    return [pattern.pattern for pattern in SECRET_PATTERNS if pattern.search(text)]


def validate_metric_value(name: str, value: Any) -> list[str]:
    if value is None or value == "":
        return []
    try:
        number = float(value)
    except (TypeError, ValueError):
        return []
    if not math.isfinite(number):
        return [f"{name} is NaN or infinite"]
    key = name.lower().replace("_", "")
    for token, (lower, upper) in METRIC_RANGES.items():
        if token in key and not lower <= number <= upper:
            return [f"{name}={number} is outside [{lower}, {upper}]"]
    if any(
        token in key
        for token in ("time", "latency", "memory", "vram", "bytes", "seconds")
    ) and number < 0:
        return [f"{name}={number} must be nonnegative"]
    return []


def _iter_mapping_rows(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for nested in value.values():
            yield from _iter_mapping_rows(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_mapping_rows(nested)


def _nested_value(payload: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = payload
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            return current
    return None


def _selected_lr(payload: Mapping[str, Any]) -> float:
    value = _nested_value(
        payload,
        ("search", "selected_learning_rate"),
        ("final_training", "learning_rate"),
    )
    if value is None:
        raise ValueError("selected LR config has no selected learning rate")
    return float(value)


def _metric_rows(metrics: Any) -> list[dict[str, Any]]:
    if isinstance(metrics, dict) and isinstance(metrics.get("evaluations"), list):
        return [row for row in metrics["evaluations"] if isinstance(row, dict)]
    if isinstance(metrics, list):
        return [row for row in metrics if isinstance(row, dict)]
    return [metrics] if isinstance(metrics, dict) else []


def _validate_identity(
    payload: Any,
    relative: str,
    model_id: str,
    track: str,
    run_id: str,
) -> list[str]:
    errors: list[str] = []
    for row in _iter_mapping_rows(payload):
        if row.get("model_id") and row["model_id"] != model_id:
            errors.append(f"mixed model IDs in {relative}")
        if row.get("dataset_track") and row["dataset_track"] != track:
            errors.append(f"mixed dataset tracks in {relative}")
        if row.get("run_id") and row["run_id"] != run_id:
            errors.append(f"mixed run IDs in {relative}")
    return errors


def validate_bundle(bundle_dir: str | Path, max_file_size_mb: float = 20) -> list[str]:
    """Validate one lightweight bundle and return all actionable errors."""
    root = Path(bundle_dir).expanduser().resolve()
    if not root.is_dir():
        return [f"missing result bundle directory: {root}"]
    errors: list[str] = []
    manifest_path = root / "bundle_manifest.json"
    if not manifest_path.is_file():
        return [f"missing {manifest_path}"]
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid bundle manifest: {exc}"]
    if int(manifest.get("schema_version", 0)) >= 3:
        return _validate_hpo_bundle(root, manifest, max_file_size_mb)
    errors.extend(
        f"missing required bundle file: {relative}"
        for relative in sorted(REQUIRED_BUNDLE_FILES)
        if not (root / relative).is_file()
    )
    errors.extend(
        f"missing required bundle directory: {relative}"
        for relative in sorted(REQUIRED_BUNDLE_DIRECTORIES)
        if not (root / relative).is_dir()
    )
    errors.extend(
        f"manifest missing field: {field}"
        for field in sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    )
    model_id = str(manifest.get("model_id", ""))
    track = str(manifest.get("dataset_track", ""))
    run_id = str(manifest.get("run_id", ""))
    if model_id not in MODEL_CONFIGS:
        errors.append(f"manifest has unrecognized model_id: {model_id}")
    if track not in {"2class", "10class"}:
        errors.append("manifest has invalid dataset_track")
    bundle_id = str(manifest.get("result_bundle_id", ""))
    if bundle_id != root.name:
        errors.append("bundle directory name does not match result_bundle_id")
    if model_id and model_id not in bundle_id:
        errors.append("bundle ID does not include model_id")
    if track and track not in bundle_id:
        errors.append("bundle ID does not include dataset_track")
    if not run_id:
        errors.append("manifest has no run_id")
    if not manifest.get("class_names"):
        errors.append("manifest has no class_names")
    elif track == "2class" and tuple(manifest["class_names"]) != ("person", "vehicle"):
        errors.append("2class bundle must use the person/vehicle class mapping")
    if manifest.get("seed_status") != "single-seed":
        errors.append("per-model final bundle must state single-seed")
    if manifest.get("seed") != 42:
        errors.append("controlled benchmark final seed must be 42")
    if manifest.get("official_full_train_verified") is not True:
        errors.append("full official train identity is not verified")
    for commit_field in ("evaluation_git_commit", "training_git_commit"):
        if not COMMIT_PATTERN.fullmatch(str(manifest.get(commit_field, ""))):
            errors.append(f"{commit_field} is missing or invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("checkpoint_sha256", "")), re.I):
        errors.append("checkpoint_sha256 is missing or invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("annotation_sha256", "")), re.I):
        errors.append("annotation_sha256 is missing or invalid")

    parsed: dict[str, Any] = {}
    for file in root.rglob("*"):
        if not file.is_file():
            continue
        relative = file.relative_to(root).as_posix()
        lower_parts = {part.lower() for part in file.relative_to(root).parts}
        if file.suffix.lower() not in APPROVED_EXTENSIONS:
            errors.append(f"unapproved artifact type: {relative}")
        if file.suffix.lower() in EXCLUDED_EXTENSIONS or lower_parts & EXCLUDED_PARTS:
            errors.append(f"excluded artifact present: {relative}")
        if file.stat().st_size > max_file_size_mb * 1024 * 1024:
            errors.append(f"oversized file: {relative}")
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        errors.extend(
            f"secret-like content in {relative}: {pattern}"
            for pattern in find_secret_like_content(text)
        )
        if "/content/drive/" in text or re.search(r"[A-Za-z]:[\\/]Users[\\/]", text):
            errors.append(f"private absolute path in {relative}")
        if PLACEHOLDER_PATTERN.search(text):
            errors.append(f"placeholder content in {relative}")
        try:
            if file.suffix.lower() == ".json":
                parsed[relative] = json.loads(text)
            elif file.suffix.lower() in {".yaml", ".yml"}:
                parsed[relative] = read_yaml(file)
            elif file.suffix.lower() == ".csv":
                reader = csv.DictReader(text.splitlines())
                if reader.fieldnames is None:
                    errors.append(f"CSV has no header: {relative}")
                parsed[relative] = list(reader)
        except (json.JSONDecodeError, csv.Error, ValueError) as exc:
            errors.append(f"invalid {file.suffix.lower()[1:].upper()} {relative}: {exc}")
            continue
        if relative in parsed:
            errors.extend(
                _validate_identity(parsed[relative], relative, model_id, track, run_id)
            )
            for row in _iter_mapping_rows(parsed[relative]):
                for name, value in row.items():
                    errors.extend(validate_metric_value(str(name), value))

    selected = parsed.get("configs/selected_lr.yaml", {})
    final_config = parsed.get("configs/final_resolved_config.yaml", {})
    if selected:
        experiment = selected.get("experiment", {})
        if experiment.get("model_id") != model_id:
            errors.append("selected LR config model identity mismatch")
        if experiment.get("dataset_track") != track:
            errors.append("selected LR config dataset track mismatch")
    try:
        selected_lr = _selected_lr(selected)
        if not math.isclose(
            selected_lr,
            float(manifest.get("selected_learning_rate")),
            rel_tol=1e-12,
        ):
            errors.append("selected LR does not match bundle manifest")
        final_lr = float(final_config.get("overrides", {}).get("learning_rate"))
        if not math.isclose(selected_lr, final_lr, rel_tol=1e-12):
            errors.append("selected LR was not applied to final training")
    except (TypeError, ValueError):
        errors.append("selected/final learning-rate provenance is invalid")
    if final_config:
        fixed = {
            "dataset_track": "2class",
            "image_size": 640,
            "seed": 42,
            "epochs": 25,
            "scheduler_horizon": 25,
            "run_kind": "final_complete_official_train",
        }
        for key, expected in fixed.items():
            if final_config.get(key) != expected:
                errors.append(f"final config has incompatible {key}")
    metrics = _metric_rows(parsed.get("metrics/final_metrics.json", {}))
    if not metrics:
        errors.append("final_metrics.json contains no evaluation rows")
    elif not any(
        any(name in row for name in ("mAP", "map", "AP", "AP50", "map50"))
        for row in metrics
    ):
        errors.append("final_metrics.json has no AP/mAP metric")
    per_class = parsed.get("metrics/per_class_metrics.csv", [])
    if not any(
        row.get("class_name")
        and any(name in row for name in ("AP", "ap", "mAP", "map"))
        for row in per_class
    ):
        errors.append("per_class_metrics.csv has no measured per-class AP")
    profile = parsed.get("metrics/profiling_summary.json", {})
    measurements = profile.get("measurements", []) if isinstance(profile, Mapping) else []
    profile_keys = {
        key for row in measurements if isinstance(row, Mapping) for key in row
    }
    if not {"total_parameters", "total_training_seconds"} <= profile_keys:
        errors.append("profiling summary is missing parameter or training-runtime measurements")
    if not profile_keys.intersection({"mean_latency_ms", "fps"}):
        errors.append("profiling summary is missing latency/FPS measurements")
    candidates = parsed.get("search/candidates.csv", [])
    promotions = parsed.get("search/promotion_history.csv", [])
    if not candidates or not any(row.get("learning_rate") for row in candidates):
        errors.append("search candidates are missing")
    if not promotions or not any(row.get("epoch") for row in promotions):
        errors.append("search promotion history is missing")
    dataset_hashes = parsed.get("provenance/dataset_hashes.json", {})
    if dataset_hashes:
        if dataset_hashes.get("official_full_train_verified") is not True:
            errors.append("dataset provenance does not prove full official train")
        if dataset_hashes.get("official_validation_verified") is not True:
            errors.append("dataset provenance does not prove official validation")
        for key in ("official_train_sha256", "official_validation_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(dataset_hashes.get(key, "")), re.I):
                errors.append(f"dataset provenance missing {key}")
        if (
            dataset_hashes.get("evaluation_annotation_sha256")
            != manifest.get("annotation_sha256")
        ):
            errors.append("evaluation annotation hash provenance mismatch")
    generated = set(manifest.get("generated_files", []))
    missing_generated = REQUIRED_BUNDLE_FILES - generated
    errors.extend(
        f"manifest generated_files missing: {relative}"
        for relative in sorted(missing_generated)
    )
    return sorted(set(errors))


def _validate_hpo_bundle(
    root: Path,
    manifest: dict[str, Any],
    max_file_size_mb: float,
) -> list[str]:
    """Validate a measured multi-seed HPO bundle without LR-only assumptions."""
    errors = [
        f"missing required HPO bundle file: {relative}"
        for relative in sorted(HPO_REQUIRED_BUNDLE_FILES)
        if not (root / relative).is_file()
    ]
    required_fields = {
        "schema_version",
        "result_bundle_id",
        "created_at",
        "model_id",
        "architecture_family",
        "dataset_track",
        "class_names",
        "protocol_id",
        "run_ids",
        "seeds",
        "seed_status",
        "checkpoint_sha256",
        "annotation_sha256",
        "official_full_train_verified",
        "evaluation_git_commit",
        "training_git_commits",
        "generated_files",
        "intentionally_excluded_files",
        "export_status",
    }
    errors.extend(
        f"manifest missing field: {field}"
        for field in sorted(required_fields - set(manifest))
    )
    model_id = str(manifest.get("model_id", ""))
    track = str(manifest.get("dataset_track", ""))
    if model_id not in MODEL_CONFIGS:
        errors.append(f"manifest has unrecognized model_id: {model_id}")
    if track not in {"2class", "10class"}:
        errors.append("manifest has invalid dataset_track")
    if manifest.get("result_bundle_id") != root.name:
        errors.append("bundle directory name does not match result_bundle_id")
    if model_id and model_id not in root.name:
        errors.append("bundle ID does not include model_id")
    if track and track not in root.name:
        errors.append("bundle ID does not include dataset_track")
    if manifest.get("protocol_id") != "two_stage_random_hpo_v1":
        errors.append("HPO bundle has incompatible protocol_id")
    if manifest.get("seed_status") != "multi-seed":
        errors.append("HPO bundle must state multi-seed")
    if sorted(manifest.get("seeds", [])) != [17, 42, 3407]:
        errors.append("HPO bundle must contain seeds 17, 42, and 3407")
    if len(manifest.get("run_ids", [])) != 6:
        errors.append("HPO bundle must contain six baseline/tuned run IDs")
    if manifest.get("official_full_train_verified") is not True:
        errors.append("full official train identity is not verified")
    if not COMMIT_PATTERN.fullmatch(
        str(manifest.get("evaluation_git_commit", ""))
    ):
        errors.append("evaluation_git_commit is missing or invalid")
    if not manifest.get("training_git_commits") or any(
        not COMMIT_PATTERN.fullmatch(str(value))
        for value in manifest.get("training_git_commits", [])
    ):
        errors.append("training_git_commits are missing or invalid")
    hashes = manifest.get("checkpoint_sha256", [])
    if len(hashes) != 6 or any(
        not re.fullmatch(r"[0-9a-f]{64}", str(value), re.I)
        for value in hashes
    ):
        errors.append("checkpoint_sha256 must contain six valid hashes")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(manifest.get("annotation_sha256", "")), re.I
    ):
        errors.append("annotation_sha256 is missing or invalid")
    generated = set(manifest.get("generated_files", []))
    for relative in HPO_REQUIRED_BUNDLE_FILES - generated:
        errors.append(f"manifest generated_files missing: {relative}")
    for file in root.rglob("*"):
        if not file.is_file():
            continue
        relative = file.relative_to(root).as_posix()
        parts = {part.lower() for part in file.relative_to(root).parts}
        if (
            file.suffix.lower() not in APPROVED_EXTENSIONS
            or file.suffix.lower() in EXCLUDED_EXTENSIONS
            or parts & EXCLUDED_PARTS
        ):
            errors.append(f"excluded or unapproved artifact: {relative}")
        if file.stat().st_size > max_file_size_mb * 1024 * 1024:
            errors.append(f"oversized file: {relative}")
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "/content/drive/" in text or re.search(
            r"[A-Za-z]:[\\/]Users[\\/]", text
        ):
            errors.append(f"private absolute path in {relative}")
        errors.extend(
            f"secret-like content in {relative}: {pattern}"
            for pattern in find_secret_like_content(text)
        )
        try:
            if file.suffix.lower() == ".json":
                json.loads(text)
            elif file.suffix.lower() in {".yaml", ".yml"}:
                read_yaml(file)
        except (json.JSONDecodeError, ValueError) as error:
            errors.append(f"invalid structured file {relative}: {error}")
    comparison = root / "metrics" / "comparison.json"
    if comparison.is_file():
        groups = read_json(comparison).get("groups", [])
        selected = [
            group
            for group in groups
            if group.get("model_id") == model_id
            and group.get("dataset_track") == track
        ]
        if len(selected) != 2 or any(
            group.get("status") != "COMPLETE" for group in selected
        ):
            errors.append("comparison lacks complete baseline and tuned groups")
    return sorted(set(errors))


def _copy_approved(
    source: Path,
    destination: Path,
    copied: list[str],
    excluded: list[str],
) -> None:
    relative_parts = {part.lower() for part in source.parts}
    if (
        source.suffix.lower() in EXCLUDED_EXTENSIONS
        or relative_parts & EXCLUDED_PARTS
        or source.suffix.lower() not in APPROVED_EXTENSIONS
    ):
        excluded.append(str(source))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".json", ".yaml", ".yml", ".csv", ".md", ".html", ".txt"}:
        text = sanitize_text(source.read_text(encoding="utf-8"))
        if source.suffix.lower() == ".json":
            text = json.dumps(sanitize_value(json.loads(text)), indent=2, sort_keys=True) + "\n"
        destination.write_text(text, encoding="utf-8")
    else:
        shutil.copy2(source, destination)
    copied.append(str(destination))


def export_bundle(
    drive_root: str | Path,
    bundle_id: str,
    repo_root: str | Path,
    max_file_size_mb: float = 20,
    clean_target: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Copy one validated bundle into ``results/bundles`` without Git actions."""
    drive = Path(drive_root).expanduser().resolve()
    bundle = drive / "result_bundles" / bundle_id
    errors = validate_bundle(bundle, max_file_size_mb)
    errors.extend(_verify_bundle_registry(drive, bundle))
    if errors:
        raise ValueError("result bundle validation failed:\n" + "\n".join(sorted(set(errors))))
    results_root = Path(repo_root).expanduser().resolve() / "results"
    target_bundle = results_root / "bundles" / bundle_id
    files = [path for path in sorted(bundle.rglob("*")) if path.is_file()]
    preview: list[dict[str, Any]] = []
    copied: list[str] = []
    excluded: list[str] = []
    for source in files:
        relative = source.relative_to(bundle)
        destination = target_bundle / relative
        allowed = (
            source.suffix.lower() in APPROVED_EXTENSIONS
            and source.suffix.lower() not in EXCLUDED_EXTENSIONS
            and not ({part.lower() for part in relative.parts} & EXCLUDED_PARTS)
        )
        preview.append(
            {
                "source": relative.as_posix(),
                "destination": destination.relative_to(results_root).as_posix(),
                "size_bytes": source.stat().st_size,
                "action": "copy" if allowed else "exclude",
            }
        )
    total_size = sum(item["size_bytes"] for item in preview if item["action"] == "copy")
    latest_manifest = results_root / "manifests" / "latest_result_manifest.json"
    if not dry_run:
        if target_bundle.exists() and clean_target:
            archive = results_root / "archive" / f"{bundle_id}__replaced"
            if archive.exists():
                raise FileExistsError(f"refusing to overwrite archive: {archive}")
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target_bundle), str(archive))
        for source in files:
            _copy_approved(
                source,
                target_bundle / source.relative_to(bundle),
                copied,
                excluded,
            )
        bundle_manifest = read_json(bundle / "bundle_manifest.json")
        write_json(
            latest_manifest,
            sanitize_value(
                {
                    "result_bundle_id": bundle_id,
                    "bundle_path": f"bundles/{bundle_id}",
                    "model_id": bundle_manifest["model_id"],
                    "dataset_track": bundle_manifest["dataset_track"],
                    "run_id": bundle_manifest.get("run_id"),
                    "run_ids": bundle_manifest.get("run_ids"),
                    "protocol_id": bundle_manifest.get("protocol_id"),
                    "seed_status": bundle_manifest["seed_status"],
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "file_count": len(copied),
                    "total_size_bytes": total_size,
                    "checkpoint_policy": "checkpoints remain in Google Drive",
                }
            ),
        )
    else:
        copied = [
            str(results_root / item["destination"])
            for item in preview
            if item["action"] == "copy"
        ]
        excluded = [
            item["source"] for item in preview if item["action"] == "exclude"
        ]
    return {
        "bundle_id": bundle_id,
        "target_branch": "experiment-results",
        "target_bundle": str(target_bundle),
        "latest_manifest": str(latest_manifest),
        "validation": "passed",
        "file_count": len(copied),
        "total_size_bytes": total_size,
        "copied": copied,
        "excluded": excluded,
        "preview": preview,
        "projected_git_diff": [
            f"A results/bundles/{bundle_id}/...",
            "M results/manifests/latest_result_manifest.json",
        ],
        "dry_run": dry_run,
    }


def _verify_bundle_registry(drive_root: Path, bundle: Path) -> list[str]:
    manifest = read_json(bundle / "bundle_manifest.json")
    registry_path = drive_root / "experiment_registry" / "checkpoint_registry.json"
    if not registry_path.is_file():
        return [f"missing registry: {registry_path}"]
    registered = read_json(registry_path).get("runs", {})
    if manifest.get("run_ids"):
        errors: list[str] = []
        expected_hashes = manifest.get("checkpoint_sha256", [])
        for index, run_id in enumerate(manifest["run_ids"]):
            run = registered.get(run_id)
            if not run:
                errors.append(f"selected run is absent from registry: {run_id}")
                continue
            if (
                run.get("status") != "completed"
                or run.get("model_id") != manifest.get("model_id")
                or run.get("dataset_track") != manifest.get("dataset_track")
                or run.get("protocol_id") != manifest.get("protocol_id")
            ):
                errors.append(f"selected run is incompatible: {run_id}")
                continue
            try:
                checkpoint = resolve_manifest_checkpoint(
                    run, allow_legacy_aliases=True
                )
            except FileNotFoundError as error:
                errors.append(str(error))
                continue
            if (
                index >= len(expected_hashes)
                or sha256_file(checkpoint) != expected_hashes[index]
            ):
                errors.append(f"checkpoint hash mismatch: {run_id}")
        return errors
    run = registered.get(manifest.get("run_id"))
    if not run:
        return [f"selected run is absent from registry: {manifest.get('run_id')}"]
    errors: list[str] = []
    if run.get("status") != "completed":
        errors.append("selected run is not completed")
    for field in ("model_id", "dataset_track"):
        if run.get(field) != manifest.get(field):
            errors.append(f"selected run has incompatible {field}")
    if tuple(run.get("class_names", [])) != tuple(manifest.get("class_names", [])):
        errors.append("selected run has incompatible classes")
    try:
        checkpoint = resolve_manifest_checkpoint(run, allow_legacy_aliases=True)
    except FileNotFoundError as error:
        errors.append(str(error))
    else:
        if sha256_file(checkpoint) != manifest.get("checkpoint_sha256"):
            errors.append("checkpoint hash mismatch")
    annotation = (
        drive_root
        / "datasets"
        / "VisDrone2019-DET"
        / "processed"
        / f"coco_{manifest.get('dataset_track')}"
        / "annotations"
        / "instances_val.json"
    )
    if not annotation.is_file():
        errors.append("evaluation annotation is missing")
    elif sha256_file(annotation) != manifest.get("annotation_sha256"):
        errors.append("annotation hash mismatch")
    return errors


def write_csv_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row}) or ["status"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _repository_commit(repo_root: str | Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _find_search_summary(
    drive_root: Path, repo_root: Path, model_id: str
) -> Path | None:
    filename = f"{model_id}_2class_search_summary.json"
    for candidate in (
        drive_root / "lr_search_configs" / filename,
        repo_root / "configs" / "lr_search" / filename,
    ):
        if candidate.is_file():
            return candidate
    return None


def _choose_evaluation(
    paths: ProjectPaths,
    dataset_track: str,
    model_id: str | None,
    run_id: str | None,
) -> tuple[str, str, list[dict[str, Any]], list[Path]]:
    matches: list[tuple[dict[str, Any], Path]] = []
    for file in sorted(paths.evaluation.glob("*__metrics.json")):
        row = read_json(file)
        if row.get("dataset_track") != dataset_track:
            continue
        if model_id and row.get("model_id") != model_id:
            continue
        if run_id and row.get("run_id") != run_id:
            continue
        matches.append((row, file))
    if not matches:
        raise RuntimeError("No matching evaluation metrics were found")
    models = {str(row["model_id"]) for row, _ in matches if row.get("model_id")}
    runs = {str(row["run_id"]) for row, _ in matches if row.get("run_id")}
    if len(models) != 1:
        raise RuntimeError(
            "Multiple evaluated models are available; pass --model-id explicitly"
        )
    chosen_model = next(iter(models))
    if run_id is None:
        registry_runs = RunRegistry(paths).list_available_runs(
            chosen_model, dataset_track, status="completed"
        )
        evaluated_run_ids = {str(row["run_id"]) for row, _ in matches}
        final_ids = []
        for run in registry_runs:
            candidate_dir = Path(
                str(run.get("run_dir") or paths.final_checkpoints / chosen_model / run["run_id"])
            )
            config = candidate_dir / "training_config.yaml"
            if (
                str(run.get("run_id")) in evaluated_run_ids
                and config.is_file()
                and read_yaml(config).get("run_kind") == "final_complete_official_train"
            ):
                final_ids.append(str(run["run_id"]))
        if not final_ids:
            raise RuntimeError("No completed evaluated final-training run was found")
        run_id = final_ids[0]
    selected = [(row, file) for row, file in matches if str(row.get("run_id")) == run_id]
    if not selected:
        raise RuntimeError(f"No evaluation metrics found for run {run_id}")
    return chosen_model, run_id, [row for row, _ in selected], [file for _, file in selected]


def _candidate_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    state = summary.get("state", {})
    candidates = state.get("candidates", {}) if isinstance(state, Mapping) else {}
    if candidates:
        return [
            {
                "candidate_id": candidate_id,
                "model_id": summary.get("model_id"),
                "dataset_track": "2class",
                "learning_rate": values.get("learning_rate"),
                "status": values.get("status"),
            }
            for candidate_id, values in sorted(candidates.items())
        ]
    return [
        {
            "candidate_id": f"candidate_{index:02d}",
            "model_id": summary.get("model_id"),
            "dataset_track": "2class",
            "learning_rate": value,
            "status": "recorded",
        }
        for index, value in enumerate(summary.get("candidates", []), start=1)
    ]


def _promotion_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    state = summary.get("state", {})
    decisions = state.get("rung_decisions", []) if isinstance(state, Mapping) else []
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        promoted = set(decision.get("promoted_candidate_ids", []))
        for candidate_id in decision.get("candidate_ids_started", []):
            rows.append(
                {
                    "model_id": summary.get("model_id"),
                    "dataset_track": "2class",
                    "epoch": decision.get("epoch"),
                    "candidate_id": candidate_id,
                    "promoted": candidate_id in promoted,
                }
            )
    return rows


def _per_class_rows(
    evaluations: list[dict[str, Any]], model_id: str, run_id: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        per_class = evaluation.get("per_class") or evaluation.get("per_class_detailed") or {}
        for class_name, metrics in per_class.items():
            values = metrics if isinstance(metrics, Mapping) else {"AP": metrics}
            rows.append(
                {
                    "model_id": model_id,
                    "dataset_track": "2class",
                    "run_id": run_id,
                    "evaluation_resolution": evaluation.get("evaluation_resolution"),
                    "class_name": class_name,
                    **{
                        key: value
                        for key, value in values.items()
                        if isinstance(value, (str, int, float, bool)) or value is None
                    },
                }
            )
    return rows or [
        {
            "model_id": model_id,
            "dataset_track": "2class",
            "run_id": run_id,
            "status": "not_reported_by_evaluator",
        }
    ]


def create_result_bundle(
    drive_root: str | Path,
    dataset_track: str,
    repo_root: str | Path,
    bundle_id: str | None = None,
    model_id: str | None = None,
    run_id: str | None = None,
) -> Path:
    """Create a strict per-model bundle from measured final-run artifacts."""
    if dataset_track != "2class":
        raise ValueError("the controlled LR workflow publishes only the 2class track")
    paths = ProjectPaths.from_value(drive_root)
    repo = Path(repo_root).expanduser().resolve()
    model_id, run_id, evaluations, evaluation_files = _choose_evaluation(
        paths, dataset_track, model_id, run_id
    )
    registry = RunRegistry(paths)
    run = registry._load().get("runs", {}).get(run_id)
    if not run or run.get("status") != "completed":
        raise RuntimeError(f"Completed registry run not found: {run_id}")
    run_dir = Path(
        str(run.get("run_dir") or paths.final_checkpoints / model_id / run_id)
    )
    final_config_path = run_dir / "resolved_config.yaml"
    selected_path = find_selected_config(paths.root, model_id, repo)
    search_summary_path = _find_search_summary(paths.root, repo, model_id)
    required_sources = {
        "selected LR configuration": selected_path,
        "search summary": search_summary_path,
        "final resolved configuration": final_config_path,
        "run environment": run_dir / "environment.json",
        "dataset split summary": paths.lr_search_manifests / "split_summary.json",
    }
    missing = [name for name, path in required_sources.items() if path is None or not Path(path).is_file()]
    if missing:
        raise RuntimeError("Missing required provenance: " + ", ".join(missing))
    selected = read_yaml(selected_path)
    selected_lr = _selected_lr(selected)
    final_config = read_yaml(final_config_path)
    applied_lr = float(final_config.get("overrides", {}).get("learning_rate"))
    if not math.isclose(selected_lr, applied_lr, rel_tol=1e-12):
        raise RuntimeError("Selected LR does not match the final resolved configuration")
    fixed = {
        "dataset_track": "2class",
        "image_size": 640,
        "seed": 42,
        "epochs": 25,
        "scheduler_horizon": 25,
        "run_kind": "final_complete_official_train",
    }
    if any(final_config.get(key) != value for key, value in fixed.items()):
        raise RuntimeError("Final run is not a compatible controlled-benchmark run")
    checkpoint = resolve_manifest_checkpoint(run, allow_legacy_aliases=True)
    validation = paths.coco(dataset_track) / "annotations" / "instances_val.json"
    if not checkpoint.is_file() or not validation.is_file():
        raise RuntimeError("Final checkpoint or official validation annotation is missing")
    split_summary = read_json(paths.lr_search_manifests / "split_summary.json")
    hashes = split_summary.get("hashes", {})
    sources = split_summary.get("sources", {})
    full_train_hash = hashes.get("official_full_train.json")
    validation_hash = hashes.get("official_validation.json")
    official_train_source_hash = sources.get("official_train", {}).get("sha256")
    official_validation_source_hash = sources.get("official_validation", {}).get("sha256")
    official_train_source_path = Path(
        str(sources.get("official_train", {}).get("path", ""))
    )
    official_validation_source_path = Path(
        str(sources.get("official_validation", {}).get("path", ""))
    )
    if not official_train_source_path.is_file() or not official_validation_source_path.is_file():
        raise RuntimeError("Official dataset sources recorded by split_summary are missing")
    official_train_source = read_json(official_train_source_path)
    official_validation_source = read_json(official_validation_source_path)
    full_train_payload = read_json(paths.lr_search_manifests / "official_full_train.json")
    validation_payload = read_json(paths.lr_search_manifests / "official_validation.json")
    image_ids = lambda payload: {int(image["id"]) for image in payload.get("images", [])}
    image_names = lambda payload: {
        str(image["file_name"]) for image in payload.get("images", [])
    }
    category_ids = lambda payload: {
        int(category["id"]) for category in payload.get("categories", [])
    }
    full_train_verified = bool(
        full_train_hash
        and official_train_source_hash
        and sha256_file(official_train_source_path) == official_train_source_hash
        and image_ids(full_train_payload) == image_ids(official_train_source)
        and image_names(full_train_payload) == image_names(official_train_source)
        and category_ids(full_train_payload) == category_ids(official_train_source)
    )
    official_validation_verified = bool(
        validation_hash
        and official_validation_source_hash
        and sha256_file(official_validation_source_path)
        == official_validation_source_hash
        and image_names(validation_payload) == image_names(official_validation_source)
        and len(validation_payload.get("annotations", []))
        == len(official_validation_source.get("annotations", []))
        and category_ids(validation_payload)
        == category_ids(official_validation_source)
    )
    if not full_train_verified or not official_validation_verified:
        raise RuntimeError("Dataset split hashes do not prove official train/validation identity")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bundle_id = bundle_id or f"{model_id}__{dataset_track}__{stamp}"
    if model_id not in bundle_id or dataset_track not in bundle_id:
        raise ValueError("bundle ID must include model_id and dataset_track")
    output = paths.result_bundles / bundle_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing bundle: {output}")
    for directory in REQUIRED_BUNDLE_DIRECTORIES:
        (output / directory).mkdir(parents=True, exist_ok=True)

    search_summary = sanitize_value(read_json(search_summary_path))
    write_yaml(output / "configs" / "selected_lr.yaml", sanitize_value(selected))
    write_yaml(
        output / "configs" / "final_resolved_config.yaml",
        sanitize_value(final_config),
    )
    write_csv_rows(output / "search" / "candidates.csv", _candidate_rows(search_summary))
    write_csv_rows(
        output / "search" / "promotion_history.csv",
        _promotion_rows(search_summary),
    )
    write_json(output / "search" / "search_summary.json", search_summary)
    final_metrics: dict[str, Any] = {
        "model_id": model_id,
        "dataset_track": dataset_track,
        "run_id": run_id,
        "evaluations": sanitize_value(evaluations),
    }
    write_json(output / "metrics" / "final_metrics.json", final_metrics)
    write_csv_rows(
        output / "metrics" / "per_class_metrics.csv",
        _per_class_rows(evaluations, model_id, run_id),
    )
    profile_fields = (
        "evaluation_resolution", "mean_latency_ms", "p50_latency_ms",
        "p95_latency_ms", "fps", "peak_vram_mb", "total_parameters",
        "trainable_parameters", "total_training_seconds", "evaluation_hardware",
    )
    write_json(
        output / "metrics" / "profiling_summary.json",
        {
            "model_id": model_id,
            "dataset_track": dataset_track,
            "run_id": run_id,
            "measurements": [
                {key: row.get(key) for key in profile_fields if key in row}
                for row in evaluations
            ],
        },
    )
    environment = sanitize_value(read_json(run_dir / "environment.json"))
    write_json(output / "provenance" / "environment_summary.json", environment)
    write_json(
        output / "provenance" / "dataset_hashes.json",
        {
            "model_id": model_id,
            "dataset_track": dataset_track,
            "run_id": run_id,
            "official_train_sha256": official_train_source_hash,
            "official_validation_sha256": official_validation_source_hash,
            "official_full_train_manifest_sha256": full_train_hash,
            "official_validation_manifest_sha256": validation_hash,
            "evaluation_annotation_sha256": sha256_file(validation),
            "official_full_train_verified": full_train_verified,
            "official_validation_verified": official_validation_verified,
            "split_verification": split_summary.get("verification", {}),
            "split_statistics": split_summary.get("statistics", {}),
        },
    )
    evaluation_commit = _repository_commit(repo)
    training_commit = str(run.get("git_commit", ""))
    (output / "provenance" / "git_commit.txt").write_text(
        f"evaluation_git_commit={evaluation_commit}\n"
        f"training_git_commit={training_commit}\n",
        encoding="utf-8",
    )
    metric = evaluations[0]
    report_lines = [
        f"# {model_id} final benchmark result",
        "",
        f"- Bundle ID: `{bundle_id}`",
        f"- Run ID: `{run_id}`",
        f"- Dataset track: `{dataset_track}`",
        "- Seed status: single-seed (seed 42)",
        f"- Selected learning rate: `{selected_lr:.12g}`",
        f"- Evaluation resolution: `{metric.get('evaluation_resolution', 'recorded in metrics')}`",
    ]
    for label, key in (("mAP", "mAP"), ("AP50", "AP50"), ("APtiny", "APtiny")):
        if metric.get(key) is not None:
            report_lines.append(f"- {label}: `{float(metric[key]):.6f}`")
    report_lines.extend(
        [
            "",
            "This bundle contains lightweight measured summaries only. "
            "Checkpoints, raw predictions, datasets, and credentials remain outside Git.",
            "",
        ]
    )
    report = "\n".join(report_lines)
    (output / "reports" / "model_report.md").write_text(report, encoding="utf-8")
    (output / "README.md").write_text(report, encoding="utf-8")
    figure_sources = list(
        (
            paths.reports / "models" / model_id / run_id / "figures"
        ).glob("*.png")
    )
    for figure in figure_sources:
        if figure.stat().st_size <= 20 * 1024 * 1024:
            shutil.copy2(figure, output / "reports" / "figures" / figure.name)

    manifest = {
        "schema_version": 2,
        "result_bundle_id": bundle_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "architecture_family": run.get("architecture_family"),
        "dataset_track": dataset_track,
        "class_names": run.get("class_names", []),
        "run_id": run_id,
        "seed": run.get("seed"),
        "seed_status": "single-seed",
        "selected_learning_rate": selected_lr,
        "checkpoint_sha256": sha256_file(checkpoint),
        "annotation_sha256": sha256_file(validation),
        "official_full_train_verified": full_train_verified,
        "evaluation_git_commit": evaluation_commit,
        "training_git_commit": training_commit,
        "evaluation_metric_sha256": {
            file.name: sha256_file(file) for file in evaluation_files
        },
        "generated_files": sorted(
            {
                *REQUIRED_BUNDLE_FILES,
                *[
                    f"reports/figures/{path.name}"
                    for path in (output / "reports" / "figures").glob("*")
                    if path.is_file()
                ],
            }
        ),
        "intentionally_excluded_files": sorted(EXCLUDED_EXTENSIONS),
        "export_status": "created",
    }
    write_json(output / "bundle_manifest.json", manifest)
    errors = validate_bundle(output)
    errors.extend(_verify_bundle_registry(paths.root, output))
    if errors:
        raise RuntimeError("Created bundle failed validation:\n" + "\n".join(errors))
    return output
