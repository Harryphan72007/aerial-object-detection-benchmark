from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.data.class_mapping import IGNORED_CATEGORY_IDS, ClassMapping
from src.data.dataloaders import CocoDetectionRecords as LegacyRecords
from src.data.dataset import CocoDetectionDataset, detection_collate
from src.data.manifests import build_dataset_manifest, write_dataset_manifest
from src.utils.serialization import read_json

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "tests" / "fixtures" / "tiny_coco" / "instances_tiny.json"


def _images(root: Path) -> Path:
    root.mkdir()
    Image.new("RGB", (20, 16), "white").save(root / "image_7.jpg")
    Image.new("RGB", (24, 18), "black").save(root / "image_9.jpg")
    return root


def test_canonical_mapping_covers_ids_and_ignored_regions() -> None:
    mapping = ClassMapping("2class")
    assert [mapping.map_category(value) for value in range(1, 11)] == [
        1, 1, 2, 2, 2, 2, 2, 2, 2, 2
    ]
    assert IGNORED_CATEGORY_IDS == {0, 11}
    assert ClassMapping("10class").class_names[0] == "pedestrian"


def test_one_batch_preserves_image_ids_boxes_areas_and_legacy_parity(
    tmp_path: Path,
) -> None:
    images = _images(tmp_path / "images")
    canonical = CocoDetectionDataset(images, ANNOTATIONS)
    legacy = LegacyRecords(images, ANNOTATIONS)
    batch = detection_collate([canonical[0], canonical[1]])
    assert [item["image_id"] for item in batch] == [7, 9]
    assert [item["annotations"][0]["category_id"] for item in batch] == [1, 2]
    assert [item["annotations"][0]["bbox"] for item in batch] == [
        [1, 2, 5, 6], [3, 4, 8, 5]
    ]
    assert [item["annotations"][0]["area"] for item in batch] == [30, 40]
    for index in range(2):
        assert canonical[index]["image_id"] == legacy[index]["image_id"]
        assert canonical[index]["annotations"] == legacy[index]["annotations"]


def test_dataset_manifest_round_trip_is_deterministic(tmp_path: Path) -> None:
    expected = build_dataset_manifest(
        ANNOTATIONS, split="train", track="2class", ignored_regions=2
    )
    destination = tmp_path / "dataset_manifest.json"
    actual = write_dataset_manifest(
        destination,
        ANNOTATIONS,
        split="train",
        track="2class",
        ignored_regions=2,
    )
    assert actual == expected == read_json(destination)
    assert actual["image_ids"] == [7, 9]
    assert actual["annotation_count"] == 2
    assert actual["annotations_by_class"] == {"person": 1, "vehicle": 1}
    assert actual["total_annotation_area"] == 70.0
    assert actual["ignored_regions"] == 2
