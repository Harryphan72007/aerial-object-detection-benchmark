"""Deterministic manifests for converted COCO datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.class_mapping import ClassMapping
from src.utils.serialization import sha256_file, write_json


def build_dataset_manifest(
    annotation_file: str | Path,
    *,
    split: str,
    track: str,
    ignored_regions: int,
) -> dict[str, Any]:
    """Describe IDs, boxes, areas, mapping, and ignored-region provenance."""

    source = Path(annotation_file)
    payload = json.loads(source.read_text(encoding="utf-8"))
    mapping = ClassMapping(track)
    expected_categories = mapping.coco_categories()
    if payload.get("categories") != expected_categories:
        raise ValueError("COCO categories do not match the canonical class mapping")
    image_ids = [int(image["id"]) for image in payload.get("images", [])]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("COCO image IDs must be unique")
    known_images = set(image_ids)
    class_counts: Counter[str] = Counter()
    total_area = 0.0
    annotation_ids: list[int] = []
    for annotation in payload.get("annotations", []):
        annotation_id = int(annotation["id"])
        annotation_ids.append(annotation_id)
        if int(annotation["image_id"]) not in known_images:
            raise ValueError(f"annotation {annotation_id} references an unknown image")
        category_id = int(annotation["category_id"])
        if category_id < 1 or category_id > len(mapping.class_names):
            raise ValueError(f"annotation {annotation_id} has an invalid category")
        width, height = map(float, annotation["bbox"][2:])
        expected_area = width * height
        area = float(annotation["area"])
        if abs(area - expected_area) > max(1e-9, expected_area * 1e-9):
            raise ValueError(f"annotation {annotation_id} area does not match bbox")
        total_area += area
        class_counts[mapping.class_names[category_id - 1]] += 1
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("COCO annotation IDs must be unique")
    return {
        "schema_version": 1,
        "split": split,
        "track": track,
        "annotation_sha256": sha256_file(source),
        "class_names": mapping.class_names,
        "image_ids": image_ids,
        "image_count": len(image_ids),
        "annotation_count": len(annotation_ids),
        "annotations_by_class": dict(sorted(class_counts.items())),
        "total_annotation_area": total_area,
        "ignored_regions": int(ignored_regions),
    }


def write_dataset_manifest(
    destination: str | Path,
    annotation_file: str | Path,
    *,
    split: str,
    track: str,
    ignored_regions: int,
) -> dict[str, Any]:
    manifest = build_dataset_manifest(
        annotation_file,
        split=split,
        track=track,
        ignored_regions=ignored_regions,
    )
    write_json(destination, manifest)
    return manifest
