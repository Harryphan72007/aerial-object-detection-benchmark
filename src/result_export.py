"""Validated, portable export of lightweight benchmark results."""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.paths import ProjectPaths
from src.training.checkpointing import RunRegistry
from src.utils.serialization import read_json, sha256_file, write_json

REQUIRED_BUNDLE_FILES = {
    "bundle_manifest.json",
    "selected_runs.csv",
    "final_results.csv",
    "final_results.json",
    "training_efficiency.csv",
    "inference_efficiency.csv",
    "per_class_metrics.csv",
    "per_size_metrics.csv",
    "statistical_summary.csv",
    "recommendation_matrix.csv",
}
REQUIRED_BUNDLE_DIRECTORIES = {"figures", "reports", "samples"}
REQUIRED_MANIFEST_FIELDS = {
    "result_bundle_id", "evaluation_date", "evaluation_git_commit", "selected_run_ids",
    "model_ids", "architecture_families", "checkpoint_sha256", "training_git_commits",
    "dataset_track", "class_names", "annotation_sha256", "hardware_information",
    "framework_versions", "resolutions", "confidence_settings", "iou_settings",
    "metric_configuration", "generated_files", "intentionally_excluded_files",
    "failed_models", "seed_status", "export_status",
}
APPROVED_EXTENSIONS = {".csv", ".json", ".md", ".html", ".png", ".jpg", ".jpeg", ".txt"}
EXCLUDED_EXTENSIONS = {
    ".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".engine", ".weights", ".db",
    ".zip", ".tar", ".gz", ".npz", ".npy",
}
EXCLUDED_PARTS = {
    "datasets", "predictions", "raw_predictions", "profiling", "tensorboard",
    "credentials", "checkpoints", "optuna",
}
SECRET_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,]{8,}"),
)
ABSOLUTE_PATH_PATTERN = re.compile(r"(?:[A-Za-z]:[\\/]|/content/drive/|/mnt/|/home/)[^\s\"']+")
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
    if any(token in key for token in ("time", "latency", "memory", "vram", "bytes", "seconds")) and number < 0:
        return [f"{name}={number} must be nonnegative"]
    return []


def validate_bundle(bundle_dir: str | Path, max_file_size_mb: float = 20) -> list[str]:
    """Validate a Drive result bundle and return all actionable errors."""
    root = Path(bundle_dir).expanduser().resolve()
    errors: list[str] = []
    if not root.exists() or not root.is_dir():
        return [f"missing result bundle directory: {root}"]
    manifest_path = root / "bundle_manifest.json"
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid bundle manifest: {exc}"]
    for required in REQUIRED_BUNDLE_FILES:
        if not (root / required).exists():
            errors.append(f"missing required bundle file: {required}")
    for required in REQUIRED_BUNDLE_DIRECTORIES:
        if not (root / required).is_dir():
            errors.append(f"missing required bundle directory: {required}")
    missing_manifest_fields = REQUIRED_MANIFEST_FIELDS - set(manifest)
    errors.extend(f"manifest missing field: {field}" for field in sorted(missing_manifest_fields))
    track = manifest.get("dataset_track")
    if track not in {"2class", "10class"}:
        errors.append("manifest has invalid dataset_track")
    if manifest.get("result_bundle_id") and track and track not in str(manifest["result_bundle_id"]):
        errors.append("bundle ID does not match dataset track")
    if not manifest.get("selected_run_ids"):
        errors.append("manifest has no selected_run_ids")
    if not manifest.get("class_names"):
        errors.append("manifest has no class_names")
    if not manifest.get("model_ids"):
        errors.append("manifest has no model_ids")
    if not manifest.get("architecture_families"):
        errors.append("manifest has no architecture_families")
    for file in root.rglob("*"):
        if not file.is_file():
            continue
        relative = file.relative_to(root).as_posix().lower()
        if file.suffix.lower() in EXCLUDED_EXTENSIONS or any(part in relative for part in EXCLUDED_PARTS):
            errors.append(f"excluded artifact present: {file.relative_to(root)}")
        if file.stat().st_size > max_file_size_mb * 1024 * 1024:
            errors.append(f"oversized file: {file.relative_to(root)}")
        try:
            content = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        errors.extend(
            f"secret-like content in {file.relative_to(root)}: {pattern}"
            for pattern in find_secret_like_content(content)
        )
        if "/content/drive/" in content or re.search(r"[A-Za-z]:[\\/]Users[\\/]", content):
            errors.append(f"private absolute path in {file.relative_to(root)}")
        if file.suffix.lower() == ".csv":
            try:
                rows = csv.DictReader(content.splitlines())
                for row in rows:
                    for name, value in row.items():
                        errors.extend(validate_metric_value(name, value))
                    if row.get("dataset_track") and row["dataset_track"] != track:
                        errors.append(f"mixed dataset tracks in {file.relative_to(root)}")
            except csv.Error as exc:
                errors.append(f"invalid CSV {file.relative_to(root)}: {exc}")
        if file.suffix.lower() == ".json":
            try:
                parsed = json.loads(content)
                rows = parsed if isinstance(parsed, list) else [parsed]
                for row in rows:
                    if isinstance(row, dict) and row.get("dataset_track") and row["dataset_track"] != track:
                        errors.append(f"mixed dataset tracks in {file.relative_to(root)}")
            except json.JSONDecodeError as exc:
                errors.append(f"invalid JSON {file.relative_to(root)}: {exc}")
        if file.suffix.lower() == ".csv":
            try:
                rows = csv.DictReader(content.splitlines())
                if rows.fieldnames is None:
                    errors.append(f"CSV has no header: {file.relative_to(root)}")
            except csv.Error as exc:
                errors.append(f"invalid CSV {file.relative_to(root)}: {exc}")
    return sorted(set(errors))


def _copy_approved(source: Path, destination: Path, copied: list[str], excluded: list[str]) -> None:
    relative = source.as_posix().lower()
    if source.suffix.lower() in EXCLUDED_EXTENSIONS or any(part in relative for part in EXCLUDED_PARTS):
        excluded.append(str(source))
        return
    if source.suffix.lower() not in APPROVED_EXTENSIONS:
        excluded.append(str(source))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".json", ".csv", ".md", ".html", ".txt"}:
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
) -> dict[str, list[str] | str]:
    """Copy only validated lightweight files into the Git-side results tree."""
    bundle = Path(drive_root).expanduser() / "result_bundles" / bundle_id
    errors = validate_bundle(bundle, max_file_size_mb)
    errors.extend(_verify_bundle_registry(Path(drive_root), bundle))
    if errors:
        raise ValueError("result bundle validation failed:\n" + "\n".join(errors))
    target = Path(repo_root) / "results"
    manifest = read_json(bundle / "bundle_manifest.json")
    previous = None
    latest = target / "manifests" / "latest_result_manifest.json"
    if latest.exists():
        try:
            previous = read_json(latest).get("result_bundle_id")
        except json.JSONDecodeError:
            previous = None
    copied: list[str] = []
    excluded: list[str] = []
    files = [path for path in bundle.rglob("*") if path.is_file()]
    preview: list[dict[str, Any]] = []
    for source in files:
        relative = source.relative_to(bundle)
        if relative.name == "bundle_manifest.json":
            destination_relative = Path("manifests") / relative.name
        elif relative.parts and relative.parts[0] in {"figures", "reports", "samples"}:
            destination_relative = relative
        elif relative.suffix.lower() in {".csv", ".json"}:
            destination_relative = Path("tables") / relative.name
        else:
            destination_relative = Path("summary") / relative.name
        is_excluded = (
            source.suffix.lower() not in APPROVED_EXTENSIONS
            or source.suffix.lower() in EXCLUDED_EXTENSIONS
            or any(part in relative.as_posix().lower().split("/") for part in EXCLUDED_PARTS)
        )
        preview.append({
            "source": str(relative).replace("\\", "/"),
            "destination": str(destination_relative).replace("\\", "/"),
            "size_bytes": source.stat().st_size,
            "action": "exclude" if is_excluded else "copy",
        })
    if not dry_run:
        if clean_target and target.exists():
            archive = target / "archive" / str(previous or "previous")
            archive.mkdir(parents=True, exist_ok=True)
            for child in target.iterdir():
                if child.name != "archive":
                    shutil.move(str(child), str(archive / child.name))
        for source in files:
            relative = source.relative_to(bundle)
            if relative.name == "bundle_manifest.json":
                destination_relative = Path("manifests") / relative.name
            elif relative.parts and relative.parts[0] in {"figures", "reports", "samples"}:
                destination_relative = relative
            elif relative.suffix.lower() in {".csv", ".json"}:
                destination_relative = Path("tables") / relative.name
            else:
                destination_relative = Path("summary") / relative.name
            _copy_approved(source, target / destination_relative, copied, excluded)
        git_manifest = sanitize_value({
            "result_bundle_id": bundle_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "dataset_track": manifest.get("dataset_track"),
            "selected_run_ids": manifest.get("selected_run_ids", []),
            "model_ids": manifest.get("model_ids", []),
            "architecture_families": manifest.get("architecture_families", []),
            "class_names": manifest.get("class_names", []),
            "resolutions": manifest.get("resolutions", []),
            "seed_status": manifest.get("seed_status", "unknown"),
            "failed_models": manifest.get("failed_models", []),
            "training_git_commits": manifest.get("training_git_commits", {}),
            "copied_files": [str(Path(path).relative_to(target)).replace("\\", "/") for path in copied],
            "excluded_files": excluded,
            "checkpoint_policy": "checkpoints remain in Google Drive",
        })
        write_json(target / "manifests" / "latest_result_manifest.json", git_manifest)
    else:
        for item in preview:
            if item["action"] == "copy":
                copied.append(str(target / item["destination"]))
            else:
                excluded.append(item["source"])
    return {
        "copied": copied,
        "excluded": excluded,
        "preview": preview,
        "bundle_id": bundle_id,
    }


def _verify_bundle_registry(drive_root: Path, bundle: Path) -> list[str]:
    """Verify selected registry runs, checkpoint hashes, and class compatibility."""
    manifest = read_json(bundle / "bundle_manifest.json")
    registry_path = drive_root / "experiment_registry" / "checkpoint_registry.json"
    if not registry_path.exists():
        return [f"missing registry: {registry_path}"]
    registry = read_json(registry_path).get("runs", {})
    errors: list[str] = []
    classes = tuple(manifest.get("class_names", []))
    for run_id in manifest.get("selected_run_ids", []):
        run = registry.get(run_id)
        if not run:
            errors.append(f"selected run is absent from registry: {run_id}")
            continue
        if run.get("dataset_track") != manifest.get("dataset_track"):
            errors.append(f"selected run has incompatible track: {run_id}")
        if tuple(run.get("class_names", [])) != classes:
            errors.append(f"selected run has incompatible classes: {run_id}")
        checkpoint = Path(run.get("checkpoint_best_map", ""))
        if not checkpoint.exists():
            errors.append(f"selected checkpoint is missing: {checkpoint}")
            continue
        expected = manifest.get("checkpoint_sha256", {}).get(run_id)
        if not expected:
            errors.append(f"selected checkpoint hash is missing: {run_id}")
        elif sha256_file(checkpoint) != expected:
            errors.append(f"checkpoint hash mismatch: {run_id}")
    annotation_hash = manifest.get("annotation_sha256")
    if annotation_hash:
        annotation_candidates = list((drive_root / "datasets").glob(f"coco_{manifest.get('dataset_track')}/annotations/instances_val.json"))
        if not annotation_candidates:
            errors.append("evaluation annotation is missing")
        elif sha256_file(annotation_candidates[0]) != annotation_hash:
            errors.append("annotation hash mismatch")
    else:
        errors.append("annotation hash is missing")
    return errors


def write_csv_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row}) or ["status"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_result_bundle(
    drive_root: str | Path,
    dataset_track: str,
    repo_root: str | Path,
    bundle_id: str | None = None,
) -> Path:
    """Create a versioned bundle from existing evaluation JSON outputs."""
    paths = Path(drive_root)
    bundle_id = bundle_id or f"evaluation__{dataset_track}__{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output = paths / "result_bundles" / bundle_id
    output.mkdir(parents=True, exist_ok=True)
    evaluation = []
    for file in sorted((paths / "evaluation").glob("*__metrics.json")):
        data = read_json(file)
        if data.get("dataset_track", dataset_track) == dataset_track:
            evaluation.append(data)
    if not evaluation:
        raise RuntimeError(f"No evaluation metrics found for {dataset_track}")
    registry = RunRegistry(ProjectPaths.from_value(paths))
    selected_ids = sorted({str(row["run_id"]) for row in evaluation if row.get("run_id")})
    selected_runs = []
    for run_id in selected_ids:
        manifest = registry._load().get("runs", {}).get(run_id, {})
        selected_runs.append(manifest)
    annotation = paths / "datasets" / f"coco_{dataset_track}" / "annotations" / "instances_val.json"
    class_names = evaluation[0].get("class_names") or (selected_runs[0].get("class_names") if selected_runs else [])
    registry_data = registry._load().get("runs", {})
    checkpoint_hashes = {}
    for run_id in selected_ids:
        checkpoint_value = registry_data.get(run_id, {}).get("checkpoint_best_map")
        if checkpoint_value and Path(checkpoint_value).exists():
            checkpoint_hashes[run_id] = sha256_file(checkpoint_value)
    manifest = {
        "result_bundle_id": bundle_id, "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "evaluation_git_commit": _repository_commit(repo_root), "selected_run_ids": selected_ids,
        "model_ids": sorted({row.get("model_id") for row in evaluation}),
        "architecture_families": sorted({row.get("architecture_family") for row in evaluation}),
        "checkpoint_sha256": checkpoint_hashes,
        "training_git_commits": {run["run_id"]: run.get("git_commit", "unknown") for run in selected_runs},
        "dataset_track": dataset_track, "class_names": class_names,
        "annotation_sha256": sha256_file(annotation) if annotation.exists() else None,
        "hardware_information": sorted({row.get("evaluation_hardware") for row in evaluation}),
        "resolutions": sorted({row.get("evaluation_resolution") for row in evaluation}),
        "confidence_settings": {"threshold": 0.001}, "iou_settings": {"default": "COCO"},
        "metric_configuration": {"evaluator": "COCO + VisDrone slices"},
        "framework_versions": sorted({
            str(row.get("framework_version")) for row in evaluation
            if row.get("framework_version")
        }),
        "generated_files": sorted({
            "bundle_manifest.json", "selected_runs.csv", "final_results.csv",
            "final_results.json", "training_efficiency.csv", "inference_efficiency.csv",
            "per_class_metrics.csv", "per_size_metrics.csv", "statistical_summary.csv",
            "recommendation_matrix.csv", "figures/", "reports/", "reports/pull_request_summary.md", "samples/",
        }),
        "failed_models": read_json(paths / "evaluation" / "evaluation_failures.json") if (paths / "evaluation" / "evaluation_failures.json").exists() else [],
        "seed_status": "multi-seed" if len({run.get("seed") for run in selected_runs}) > 1 else "single-seed",
        "intentionally_excluded_files": sorted(EXCLUDED_EXTENSIONS), "export_status": "created",
    }
    write_json(output / "bundle_manifest.json", sanitize_value(manifest))
    write_csv_rows(output / "selected_runs.csv", [sanitize_value(run) for run in selected_runs])
    write_csv_rows(output / "final_results.csv", [sanitize_value(row) for row in evaluation])
    write_json(output / "final_results.json", sanitize_value(evaluation))
    for name in ("training_efficiency", "inference_efficiency", "per_class_metrics", "per_size_metrics", "statistical_summary", "recommendation_matrix"):
        write_csv_rows(output / f"{name}.csv", [sanitize_value(row) for row in evaluation])
    for name in ("figures", "reports", "samples"):
        (output / name).mkdir(exist_ok=True)
    (output / "reports" / "pull_request_summary.md").write_text(
        f"# VisDrone benchmark result bundle\n\n"
        f"Bundle ID: `{bundle_id}`\n\n"
        f"Dataset track: `{dataset_track}`\n\n"
        f"Seed status: {manifest['seed_status']}\n\n"
        "Checkpoints and raw predictions remain outside Git on Drive.\n",
        encoding="utf-8",
    )
    return output


def _repository_commit(repo_root: str | Path) -> str:
    """Capture the evaluation source commit without making Git a hard dependency."""
    try:
        import subprocess

        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
