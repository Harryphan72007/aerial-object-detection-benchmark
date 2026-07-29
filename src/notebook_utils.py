"""Shared repository and preflight helpers for independently runnable notebooks."""
from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from src.data.validate_annotations import validate_coco
from src.paths import ProjectPaths


def in_colab() -> bool:
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


def find_repository_root(start: str | Path | None = None) -> Path:
    override = os.environ.get("BENCHMARK_REPO_ROOT")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    current = Path(start or Path.cwd()).resolve()
    candidates.extend([current, *current.parents])
    candidates.append(Path("/content/aerial-object-detection-benchmark"))
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "__init__.py"
        ).is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot locate the benchmark repository. Set BENCHMARK_REPO_ROOT to the clone path."
    )


@dataclass(frozen=True)
class PreflightReport:
    valid: bool
    errors: list[str]
    warnings: list[str]
    free_disk_gb: float

    def raise_for_errors(self) -> None:
        if self.errors:
            raise RuntimeError("preflight failed:\n- " + "\n- ".join(self.errors))


def preflight_dataset(
    paths: ProjectPaths,
    track: str,
    *,
    split: str = "train",
    minimum_free_gb: float = 2.0,
) -> PreflightReport:
    ProjectPaths.validate_track(track)
    annotation_file = paths.coco(track) / "annotations" / f"instances_{split}.json"
    image_dir = paths.coco(track) / split
    errors: list[str] = []
    warnings: list[str] = []
    if not annotation_file.is_file():
        errors.append(
            f"missing {track} annotations: {annotation_file}; run "
            "notebooks/00_visdrone_colab_setup.ipynb first"
        )
    if not image_dir.is_dir():
        errors.append(f"missing image directory: {image_dir}")
    if not errors:
        report = validate_coco(annotation_file, image_dir)
        errors.extend(report.errors)
        warnings.extend(report.warnings)
        payload = json.loads(annotation_file.read_text(encoding="utf-8"))
        expected_ids = {1, 2} if track == "2class" else set(range(1, 11))
        actual_ids = {int(category["id"]) for category in payload["categories"]}
        if actual_ids != expected_ids:
            errors.append(
                f"annotation track mismatch: expected categories "
                f"{sorted(expected_ids)}, got {sorted(actual_ids)}"
            )
    usage = shutil.disk_usage(paths.root if paths.root.exists() else paths.root.parent)
    free_gb = usage.free / 2**30
    if free_gb < minimum_free_gb:
        errors.append(
            f"only {free_gb:.2f} GiB free; at least {minimum_free_gb:.2f} GiB required"
        )
    return PreflightReport(not errors, errors, warnings, free_gb)


def require_gpu(runtime_name: str) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(f"{runtime_name} requires PyTorch and a Colab GPU runtime") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{runtime_name} requires a CUDA GPU. In Colab choose Runtime > Change runtime type > GPU."
        )


def require_model_environment(family: str) -> None:
    """Fail before model construction when a binary/package stack is incompatible."""
    errors: list[str] = []
    if family == "openmmlab":
        if sys.version_info[:2] != (3, 10):
            errors.append(
                f"Python 3.10 required, active version is {sys.version_info.major}."
                f"{sys.version_info.minor}"
            )
        expected = {
            "torch": "2.1.",
            "mmcv": "2.1.0",
            "mmengine": "0.10.",
            "mmdet": "3.3.0",
        }
    elif family == "rtdetr":
        expected = {"transformers": "4.52.4", "accelerate": "1.7.0"}
    else:
        raise ValueError(f"unknown model environment family: {family}")
    for package, prefix in expected.items():
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{package} is not installed")
            continue
        if not version.startswith(prefix):
            errors.append(f"{package} {prefix} required, active version is {version}")
    if errors:
        requirements = (
            "requirements-openmmlab-py310-cu118.txt"
            if family == "openmmlab"
            else "requirements-rtdetr-colab.txt"
        )
        raise RuntimeError(
            f"incompatible {family} environment:\n- "
            + "\n- ".join(errors)
            + f"\nUse the isolated stack documented in {requirements}."
        )


def smoke_test_enabled(default: bool = False) -> bool:
    value = os.environ.get("SMOKE_TEST")
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
