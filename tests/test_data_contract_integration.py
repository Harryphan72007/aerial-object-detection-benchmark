from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate import discover_evaluation_dataset
from src.data.contract import verify_complete_data_contract
from src.data.dataloaders import CocoDetectionRecords
from src.data.download import VISDRONE_ARCHIVES, sha256_file
from src.data.local_cache import resolve_data_access
from src.data.smoke_dataset import create_smoke_archives
from src.paths import ProjectPaths
from src.workflows.dataset_setup import prepare_visdrone
from src.workflows.model_day import inspect_model_day


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "rtdetrv2_l"


def _assert_record(records: CocoDetectionRecords) -> None:
    assert len(records) > 0
    record = records[0]
    assert record["image"].size == (160, 96)
    assert record["file_name"]


def test_synthetic_end_to_end_contract_repairs_and_is_idempotent(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("SMOKE_TEST", "1")
    drive_root = tmp_path / "drive"
    paths = ProjectPaths.from_value(drive_root).create()
    create_smoke_archives(paths.archives)
    for spec in VISDRONE_ARCHIVES.values():
        monkeypatch.setitem(spec, "minimum_bytes", 1)

    first = prepare_visdrone(
        ROOT,
        drive_root,
        dataset_source="drive",
        smoke_test=True,
    )
    assert first["data_contract"]["verified"]
    assert first["operations"]["extractions"] == ["train", "val"]
    assert sorted(first["operations"]["conversions"]) == [
        "2class:train",
        "2class:val",
    ]
    for split, spec in VISDRONE_ARCHIVES.items():
        assert (paths.archives / str(spec["filename"])).is_file()
        assert paths.official_split(split).is_dir()
        assert (
            paths.dataset_manifests / f"{split}_extraction.json"
        ).is_file()
        assert (
            paths.coco("2class")
            / "annotations"
            / f"conversion_manifest_{split}.json"
        ).is_file()

    annotation_root = paths.coco("2class") / "annotations"
    manifest_root = paths.lr_search_manifests
    datasets = (
        CocoDetectionRecords(
            paths.images("train"), annotation_root / "instances_train.json"
        ),
        CocoDetectionRecords(
            paths.images("val"), annotation_root / "instances_val.json"
        ),
        CocoDetectionRecords(
            paths.images("train"), manifest_root / "search_train_seed42.json"
        ),
        CocoDetectionRecords(
            paths.images("train"),
            manifest_root / "search_validation_seed42.json",
        ),
    )
    for records in datasets:
        _assert_record(records)

    inspected = inspect_model_day(drive_root, MODEL_ID, ROOT)
    assert inspected["stage"] == "ENVIRONMENT"
    assert inspected["data_contract"]["verified"]
    discovered = discover_evaluation_dataset(paths, "2class", "val")
    assert discovered["record_count"] == 12
    _assert_record(discovered["records"])

    forbidden = (
        "coco_2class" + "/train",
        "coco_2class" + "/val",
        "processed/" + "coco_2class" + "/train",
        "processed/" + "coco_2class" + "/val",
    )
    searchable = [
        *ROOT.glob("*.md"),
        *ROOT.glob("src/**/*.py"),
        *ROOT.glob("scripts/**/*.py"),
        *ROOT.glob("tests/**/*.py"),
        *ROOT.glob("docs/**/*.md"),
        *ROOT.glob("notebooks/**/*.ipynb"),
    ]
    for path in searchable:
        if path == Path(__file__):
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden), path

    train_archive = paths.archives / str(VISDRONE_ARCHIVES["train"]["filename"])
    archive_hash = sha256_file(train_archive)
    deleted = next(paths.images("train").glob("*.jpg"))
    deleted.unlink()
    repaired = prepare_visdrone(
        ROOT,
        drive_root,
        dataset_source="drive",
        smoke_test=True,
    )
    assert repaired["operations"]["extractions"] == ["train"]
    assert deleted.is_file()
    assert sha256_file(train_archive) == archive_hash

    conversion_manifest = (
        annotation_root / "conversion_manifest_train.json"
    )
    stale = json.loads(conversion_manifest.read_text(encoding="utf-8"))
    stale["source_image_count"] += 1
    conversion_manifest.write_text(
        json.dumps(stale, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reconverted = prepare_visdrone(
        ROOT,
        drive_root,
        dataset_source="drive",
        smoke_test=True,
    )
    assert reconverted["operations"]["conversions"] == ["2class:train"]

    unchanged = prepare_visdrone(
        ROOT,
        drive_root,
        dataset_source="drive",
        smoke_test=True,
    )
    assert not unchanged["operations"]["downloads"]
    assert not unchanged["operations"]["extractions"]
    assert not unchanged["operations"]["conversions"]
    assert not unchanged["operations"]["lr_manifest_generations"]

    access = resolve_data_access(
        paths, "local_cache", cache_root=tmp_path / "local-cache"
    )
    local_contract = verify_complete_data_contract(
        paths,
        repo_root=ROOT,
        local_cache=access,
        max_images_per_split=12,
    )
    assert local_contract.verified, local_contract.errors
    _assert_record(
        CocoDetectionRecords(
            access.train_images,
            access.lr_manifest_dir / "search_train_seed42.json",
        )
    )
