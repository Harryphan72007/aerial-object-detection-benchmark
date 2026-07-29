import json
import zipfile

import pytest

from src.data.download import (
    ManualArchiveRequired,
    dataset_split_ready,
    ensure_archive,
    extract_idempotent,
    sha256_file,
    validate_zip,
)
from src.notebook_utils import find_repository_root, preflight_dataset
from src.paths import ProjectPaths


def test_dataset_path_resolution(tmp_path, monkeypatch):
    paths = ProjectPaths.from_value(tmp_path / "drive").create()
    assert paths.archives == paths.root / "datasets/VisDrone2019-DET/archives"
    assert paths.raw == paths.root / "datasets/VisDrone2019-DET/raw"
    assert paths.images("train") == (
        paths.root / "datasets/VisDrone2019-DET/raw/VisDrone2019-DET-train/images"
    )
    assert paths.coco("2class") == (
        paths.root / "datasets/VisDrone2019-DET/processed/coco_2class"
    )
    monkeypatch.setenv("BENCHMARK_REPO_ROOT", str(find_repository_root()))
    assert find_repository_root() == find_repository_root().resolve()


def test_archive_validation_and_idempotent_extraction(tmp_path):
    archive = tmp_path / "split.zip"
    expected = "VisDrone2019-DET-val"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(f"{expected}/images/0001.jpg", b"fake-jpeg")
        output.writestr(f"{expected}/annotations/0001.txt", b"1,1,1,1,1,1,0,0")
    validate_zip(archive, expected)
    assert len(sha256_file(archive)) == 64
    raw = tmp_path / "raw"
    manifest = tmp_path / "val_extraction.json"
    destination, action = extract_idempotent(
        archive,
        raw,
        expected,
        split="val",
        manifest_path=manifest,
    )
    assert action == "extracted"
    assert dataset_split_ready(
        destination,
        archive_path=archive,
        manifest_path=manifest,
        split="val",
    )
    second_destination, second_action = extract_idempotent(
        archive,
        raw,
        expected,
        split="val",
        manifest_path=manifest,
    )
    assert second_destination == destination
    assert second_action == "reused"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["archive_sha256"] == sha256_file(archive)
    assert payload["image_count"] == 1
    assert payload["annotation_count"] == 1
    assert len(payload["relative_filename_inventory_sha256"]) == 64
    assert payload["total_extracted_bytes"] > 0


def test_manual_archive_mode_prints_exact_placement_contract(tmp_path):
    with pytest.raises(ManualArchiveRequired) as error:
        ensure_archive(
            "train",
            tmp_path / "archives",
            tmp_path / "manifests",
            source_mode="manual",
        )
    message = str(error.value)
    assert "VisDrone2019-DET-train.zip" in message
    assert "VisDrone2019-DET-val.zip" in message
    assert str(tmp_path / "archives") in message


def test_preflight_detects_annotation_track_mismatch(tmp_path):
    paths = ProjectPaths.from_value(tmp_path / "drive").create()
    image_dir = paths.images("train")
    image_dir.mkdir(parents=True)
    annotation_dir = paths.coco("2class") / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    (annotation_dir / "instances_train.json").write_text(
        json.dumps(
            {
                "images": [],
                "annotations": [],
                "categories": [{"id": 1, "name": "pedestrian"}],
            }
        ),
        encoding="utf-8",
    )
    report = preflight_dataset(paths, "2class", minimum_free_gb=0)
    assert not report.valid
    assert any("track mismatch" in error for error in report.errors)
