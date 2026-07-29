"""Authoritative verification of the persistent and optional local data contract."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.conversion import conversion_manifest_status
from src.data.download import (
    VISDRONE_ARCHIVES,
    verify_archive_manifest,
    verify_extraction_manifest,
)
from src.data.local_cache import DataAccessPaths, verify_local_cache
from src.data.validate_annotations import validate_coco
from src.drive_sync import validate_drive_writable
from src.paths import ProjectPaths
from src.training.lr_search import (
    assert_final_training_uses_official_train,
    validate_lr_search_manifests,
)


@dataclass
class DataContractReport:
    verified: bool = False
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def raise_for_errors(self) -> None:
        if not self.verified:
            failures = self.errors or [
                f"check failed: {name}"
                for name, valid in self.checks.items()
                if not valid
            ]
            raise RuntimeError(
                "DATA CONTRACT VERIFIED: NO\n- " + "\n- ".join(failures)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "checks": self.checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


def _discover_repo_root() -> Path:
    override = os.environ.get("BENCHMARK_REPO_ROOT")
    candidates = [Path(override)] if override else []
    current = Path.cwd().resolve()
    candidates.extend([current, *current.parents])
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / ".git").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot locate repository root for conversion provenance; "
        "set BENCHMARK_REPO_ROOT."
    )


def verify_complete_data_contract(
    paths: ProjectPaths,
    *,
    repo_root: str | Path | None = None,
    local_cache: DataAccessPaths | str | Path | None = None,
    max_images_per_split: int | None = None,
) -> DataContractReport:
    """Verify every persistent dataset transition before expensive execution."""
    if not isinstance(paths, ProjectPaths):
        raise TypeError("paths must be a ProjectPaths instance")
    report = DataContractReport()
    repo = Path(repo_root).resolve() if repo_root else _discover_repo_root()

    try:
        validate_drive_writable(paths.root)
        report.checks["drive_root_mounted_and_writable"] = True
    except RuntimeError as exc:
        report.checks["drive_root_mounted_and_writable"] = False
        report.errors.append(str(exc))

    archive_reports: dict[str, Any] = {}
    extraction_reports: dict[str, Any] = {}
    conversion_reports: dict[str, Any] = {}
    for split, spec in VISDRONE_ARCHIVES.items():
        archive_result = verify_archive_manifest(
            split, paths.archives, paths.dataset_manifests
        )
        archive_reports[split] = archive_result
        report.checks[f"{split}_archive_valid"] = bool(archive_result["valid"])
        report.errors.extend(
            f"{split} archive: {error}" for error in archive_result["errors"]
        )

        extraction_result = verify_extraction_manifest(
            split=split,
            archive_path=paths.archives / str(spec["filename"]),
            split_root=paths.official_split(split),
            expected_folder=str(spec["folder"]),
            manifest_path=paths.dataset_manifests / f"{split}_extraction.json",
            verified_archive_sha256=(
                str(archive_result["sha256"]) if archive_result["valid"] else None
            ),
            verified_archive_size_bytes=(
                int(archive_result["size_bytes"])
                if archive_result["valid"]
                else None
            ),
        )
        extraction_reports[split] = extraction_result
        report.checks[f"{split}_extraction_manifest_matches"] = bool(
            extraction_result["valid"]
        )
        report.errors.extend(
            f"{split} extraction: {error}" for error in extraction_result["errors"]
        )

        conversion_result = conversion_manifest_status(
            paths,
            repo,
            "2class",
            split,
            max_images=max_images_per_split,
        )
        conversion_reports[split] = conversion_result
        report.checks[f"2class_{split}_conversion_current"] = bool(
            conversion_result["valid"]
        )
        report.errors.extend(
            f"{split} conversion: {error}" for error in conversion_result["errors"]
        )

    train_json = paths.coco("2class") / "annotations" / "instances_train.json"
    validation_json = paths.coco("2class") / "annotations" / "instances_val.json"
    try:
        split_checks = validate_lr_search_manifests(
            paths.lr_search_manifests,
            official_train_json=train_json,
            official_validation_json=validation_json,
        )
        report.checks["lr_search_manifests_current"] = True
        for name, valid in split_checks.items():
            report.checks[f"lr_{name}"] = bool(valid)
    except (
        AssertionError,
        FileNotFoundError,
        KeyError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        report.checks["lr_search_manifests_current"] = False
        report.errors.append(f"LR-search manifests: {exc}")
        split_checks = {}

    referenced_reports: dict[str, Any] = {}
    datasets = (
        ("coco_train", train_json, paths.images("train")),
        ("coco_validation", validation_json, paths.images("val")),
        (
            "search_train",
            paths.lr_search_manifests / "search_train_seed42.json",
            paths.images("train"),
        ),
        (
            "search_validation",
            paths.lr_search_manifests / "search_validation_seed42.json",
            paths.images("train"),
        ),
        (
            "official_full_train",
            paths.lr_search_manifests / "official_full_train.json",
            paths.images("train"),
        ),
        (
            "official_validation",
            paths.lr_search_manifests / "official_validation.json",
            paths.images("val"),
        ),
    )
    categories_exact = True
    all_images_exist = True
    for name, annotation, image_root in datasets:
        try:
            validation = validate_coco(annotation, image_root)
            referenced_reports[name] = validation.__dict__
            if validation.errors:
                all_images_exist = False
                report.errors.extend(f"{name}: {error}" for error in validation.errors)
            payload = json.loads(annotation.read_text(encoding="utf-8"))
            category_ids = [int(row["id"]) for row in payload.get("categories", [])]
            if category_ids != [1, 2]:
                categories_exact = False
                report.errors.append(
                    f"{name}: category IDs are {category_ids}, expected [1, 2]"
                )
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
            all_images_exist = False
            categories_exact = False
            report.errors.append(f"{name}: {exc}")
    report.checks["all_referenced_image_files_exist"] = all_images_exist
    report.checks["category_ids_exactly_1_2"] = categories_exact

    try:
        assert_final_training_uses_official_train(
            paths.lr_search_manifests, train_json
        )
        report.checks["final_train_equals_complete_official_train"] = True
        report.checks["official_validation_excluded_from_training"] = True
    except (AssertionError, FileNotFoundError, KeyError, ValueError) as exc:
        report.checks["final_train_equals_complete_official_train"] = False
        report.checks["official_validation_excluded_from_training"] = False
        report.errors.append(f"final dataset identity: {exc}")

    if local_cache is None:
        cache_report = {"enabled": False, "valid": True, "root": None}
    else:
        cache_root = (
            local_cache.cache_root
            if isinstance(local_cache, DataAccessPaths)
            else Path(local_cache)
        )
        if cache_root is None:
            cache_report = {"enabled": False, "valid": True, "root": None}
        else:
            cache_report = verify_local_cache(paths, cache_root)
            report.errors.extend(
                f"local cache: {error}" for error in cache_report["errors"]
            )
    report.checks["local_cache_valid_when_enabled"] = bool(cache_report["valid"])

    report.details = {
        "drive_root": str(paths.root),
        "archives": archive_reports,
        "extractions": extraction_reports,
        "conversions": conversion_reports,
        "lr_verification": split_checks,
        "referenced_datasets": referenced_reports,
        "local_cache": cache_report,
        "paths": {
            "train_archive": str(
                paths.archives / str(VISDRONE_ARCHIVES["train"]["filename"])
            ),
            "validation_archive": str(
                paths.archives / str(VISDRONE_ARCHIVES["val"]["filename"])
            ),
            "train_images": str(paths.images("train")),
            "validation_images": str(paths.images("val")),
            "coco_train": str(train_json),
            "coco_validation": str(validation_json),
            "lr_search_manifests": str(paths.lr_search_manifests),
        },
    }
    report.verified = bool(report.checks) and all(report.checks.values()) and not report.errors
    return report
