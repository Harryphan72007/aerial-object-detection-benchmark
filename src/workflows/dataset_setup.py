"""Idempotent, provenance-aware dataset preparation used by notebook 00."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.contract import verify_complete_data_contract
from src.data.conversion import ensure_conversion
from src.data.download import VISDRONE_ARCHIVES, ensure_archive, extract_idempotent
from src.paths import ProjectPaths
from src.training.lr_search import validate_lr_search_manifests
from src.training.lr_workflow import LRControlledBenchmark
from src.utils.serialization import write_json


def _lr_manifests_current(paths: ProjectPaths) -> bool:
    annotations = paths.coco("2class") / "annotations"
    try:
        validate_lr_search_manifests(
            paths.lr_search_manifests,
            official_train_json=annotations / "instances_train.json",
            official_validation_json=annotations / "instances_val.json",
        )
        return True
    except (
        AssertionError,
        FileNotFoundError,
        KeyError,
        ValueError,
    ):
        return False


def prepare_visdrone(
    repo_root: str | Path,
    drive_root: str | Path,
    *,
    dataset_source: str = "auto",
    prepare_10class_track: bool = False,
    redownload: bool = False,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Prepare and verify the exact persistent data contract used by every model."""
    if dataset_source not in {"auto", "download", "drive", "manual"}:
        raise ValueError("DATASET_SOURCE must be auto, download, drive, or manual")
    repo = Path(repo_root).resolve()
    paths = ProjectPaths.from_value(drive_root).create()
    operations: dict[str, list[str]] = {
        "downloads": [],
        "archive_manifest_refreshes": [],
        "extractions": [],
        "conversions": [],
        "lr_manifest_generations": [],
    }
    archive_records: list[dict[str, Any]] = []
    for split, spec in VISDRONE_ARCHIVES.items():
        ensured = ensure_archive(
            split,
            paths.archives,
            paths.dataset_manifests,
            source_mode=dataset_source,
            redownload=redownload,
        )
        if ensured.action == "downloaded":
            operations["downloads"].append(split)
        elif ensured.action == "manifest_refreshed":
            operations["archive_manifest_refreshes"].append(split)
        archive_records.append(ensured.manifest.__dict__)
        _, extraction_action = extract_idempotent(
            ensured.manifest.archive_path,
            paths.raw,
            str(spec["folder"]),
            split=split,
            manifest_path=paths.dataset_manifests / f"{split}_extraction.json",
            verified_archive_sha256=ensured.manifest.sha256,
            verified_archive_size_bytes=ensured.manifest.size_bytes,
        )
        if extraction_action in {"extracted", "recovered_staging"}:
            operations["extractions"].append(split)

    tracks = ["2class", *(["10class"] if prepare_10class_track else [])]
    max_images = 12 if smoke_test else None
    conversion_records: dict[str, dict[str, Any]] = {}
    for track in tracks:
        conversion_records[track] = {}
        for split in ("train", "val"):
            manifest, action = ensure_conversion(
                paths,
                repo,
                track,
                split,
                max_images=max_images,
            )
            conversion_records[track][split] = manifest
            if action == "converted":
                operations["conversions"].append(f"{track}:{split}")

    lr_was_current = _lr_manifests_current(paths)
    split_summary = LRControlledBenchmark(repo, paths.root).prepare_manifests(
        force=not lr_was_current
    )
    if not lr_was_current:
        operations["lr_manifest_generations"].append("2class")

    contract = verify_complete_data_contract(
        paths,
        repo_root=repo,
        max_images_per_split=max_images,
    )
    contract.raise_for_errors()
    result = {
        "drive_root": str(paths.root),
        "official_train": str(paths.official_split("train")),
        "official_validation": str(paths.official_split("val")),
        "train_images": str(paths.images("train")),
        "validation_images": str(paths.images("val")),
        "coco_2class": str(paths.coco("2class")),
        "coco_10class": str(paths.coco("10class"))
        if prepare_10class_track
        else None,
        "lr_search_manifests": str(paths.lr_search_manifests),
        "archives": archive_records,
        "conversions": conversion_records,
        "split_summary": split_summary,
        "operations": operations,
        "data_contract": contract.to_dict(),
    }
    write_json(paths.dataset_manifests / "dataset_setup_summary.json", result)
    return result
