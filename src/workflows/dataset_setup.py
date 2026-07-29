"""Idempotent dataset preparation used by notebook 00."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.data.download import VISDRONE_ARCHIVES, ensure_archive, extract_idempotent
from src.data.validate_annotations import validate_coco
from src.paths import ProjectPaths
from src.training.lr_workflow import LRControlledBenchmark


def _valid_track(paths: ProjectPaths, track: str) -> bool:
    annotation_root = paths.coco(track) / "annotations"
    try:
        for split in ("train", "val"):
            report = validate_coco(
                annotation_root / f"instances_{split}.json",
                paths.coco(track) / split,
            )
            report.raise_for_errors()
    except (FileNotFoundError, KeyError, RuntimeError, ValueError):
        return False
    return True


def prepare_visdrone(
    repo_root: str | Path,
    drive_root: str | Path,
    *,
    dataset_source: str = "auto",
    prepare_10class_track: bool = False,
    redownload: bool = False,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Download/restore, verify, extract, convert, validate, and make search manifests."""
    if dataset_source not in {"auto", "download", "drive"}:
        raise ValueError("DATASET_SOURCE must be auto, download, or drive")
    repo = Path(repo_root).resolve()
    paths = ProjectPaths.from_value(drive_root).create()
    archive_records = []
    for split, spec in VISDRONE_ARCHIVES.items():
        archive = paths.archives / str(spec["filename"])
        if dataset_source == "drive" and not archive.is_file():
            raise FileNotFoundError(f"Drive archive is missing: {archive}")
        record = ensure_archive(
            split,
            paths.archives,
            paths.dataset_manifests,
            redownload=redownload,
        )
        archive_records.append(record.__dict__)
        extract_idempotent(record.archive_path, paths.raw, str(spec["folder"]))

    tracks = ["2class", *(["10class"] if prepare_10class_track else [])]
    pending = [track for track in tracks if not _valid_track(paths, track)]
    if pending:
        command = [
            sys.executable,
            "-m",
            "scripts.prepare_data",
            "--drive-root",
            str(paths.root),
            "--tracks",
            *pending,
            "--validate",
        ]
        if smoke_test:
            command.extend(["--max-images-per-split", "4"])
        subprocess.run(command, check=True, cwd=repo)
    validation = {}
    for track in tracks:
        validation[track] = {}
        for split in ("train", "val"):
            report = validate_coco(
                paths.coco(track) / "annotations" / f"instances_{split}.json",
                paths.coco(track) / split,
            )
            report.raise_for_errors()
            validation[track][split] = report.__dict__
    split_summary = LRControlledBenchmark(repo, paths.root).prepare_manifests()
    result = {
        "drive_root": str(paths.root),
        "official_train": str(paths.raw / "VisDrone2019-DET-train"),
        "official_validation": str(paths.raw / "VisDrone2019-DET-val"),
        "coco_2class": str(paths.coco("2class")),
        "coco_10class": str(paths.coco("10class"))
        if prepare_10class_track
        else None,
        "lr_search_manifests": str(paths.lr_search_manifests),
        "archives": archive_records,
        "validation": validation,
        "split_summary": split_summary,
    }
    (paths.dataset_manifests / "dataset_setup_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
