"""Versioned mapping from original VisDrone classes to merged evaluation labels."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from src.data.class_mapping import PERSON, VEHICLE, VISDRONE_CLASSES

MERGED_CLASS_IDS = {"person": 1, "vehicle": 2}
ABLATION_DIRECT = "direct_2class"
ABLATION_MERGED = "original_10class_to_merged_2class"


def original_to_merged_mapping() -> dict[int, int]:
    mapping: dict[int, int] = {}
    for category_id, name in VISDRONE_CLASSES.items():
        if name in PERSON:
            mapping[category_id] = MERGED_CLASS_IDS["person"]
        elif name in VEHICLE:
            mapping[category_id] = MERGED_CLASS_IDS["vehicle"]
        else:
            raise ValueError(f"unmapped original VisDrone category: {category_id}/{name}")
    return mapping


def map_original_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    mapping = original_to_merged_mapping()
    output: list[dict[str, Any]] = []
    for record in records:
        source_id = int(record["category_id"])
        if source_id in {0, 11}:
            continue
        if source_id not in mapping:
            raise ValueError(f"unknown original VisDrone category id: {source_id}")
        mapped = dict(record)
        mapped["source_category_id"] = source_id
        mapped["source_category_name"] = VISDRONE_CLASSES[source_id]
        mapped["category_id"] = mapping[source_id]
        output.append(mapped)
    return output


def label_space_manifest(
    *,
    ablation_id: str,
    training_class_space: str,
    evaluation_class_space: str,
) -> dict[str, Any]:
    allowed = {ABLATION_DIRECT, ABLATION_MERGED}
    if ablation_id not in allowed:
        raise ValueError(f"unknown label-granularity ablation: {ablation_id}")
    if ablation_id == ABLATION_DIRECT and training_class_space != "merged_2class":
        raise ValueError("direct two-class evaluation requires merged_2class training")
    if ablation_id == ABLATION_MERGED and training_class_space != "original_10class":
        raise ValueError("merged ablation requires original_10class training")
    if evaluation_class_space != "merged_2class":
        raise ValueError("label-granularity v1 evaluates in merged_2class space")
    mapping = original_to_merged_mapping() if ablation_id == ABLATION_MERGED else {1: 1, 2: 2}
    mapping_hash = hashlib.sha256(
        json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "ablation_id": ablation_id,
        "training_class_space": training_class_space,
        "evaluation_class_space": evaluation_class_space,
        "evaluation_class_names": ["person", "vehicle"],
        "category_mapping": {str(key): value for key, value in mapping.items()},
        "category_mapping_hash": mapping_hash,
    }


def require_same_label_granularity(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> None:
    fields = ("ablation_id", "training_class_space", "evaluation_class_space")
    changed = {
        field: (first.get(field), second.get(field))
        for field in fields
        if first.get(field) != second.get(field)
    }
    if changed:
        raise ValueError(f"label-granularity results are not directly comparable: {changed}")
