from __future__ import annotations

import pytest

from src.evaluation.label_granularity import (
    ABLATION_DIRECT,
    ABLATION_MERGED,
    label_space_manifest,
    map_original_records,
    original_to_merged_mapping,
    require_same_label_granularity,
)


def test_all_original_classes_map_to_person_or_vehicle() -> None:
    mapping = original_to_merged_mapping()
    assert mapping == {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2}
    predictions = [
        {"image_id": 1, "category_id": category_id, "bbox": [0, 0, 1, 1], "score": 0.5}
        for category_id in range(1, 11)
    ]
    mapped = map_original_records(predictions)
    assert [row["category_id"] for row in mapped[:2]] == [1, 1]
    assert all(row["category_id"] == 2 for row in mapped[2:])
    assert mapped[0]["source_category_name"] == "pedestrian"


def test_ignored_categories_are_excluded_and_unknown_is_rejected() -> None:
    assert map_original_records([{"category_id": 0}, {"category_id": 11}]) == []
    with pytest.raises(ValueError, match="unknown"):
        map_original_records([{"category_id": 12}])


def test_direct_and_merged_manifests_remain_distinguishable() -> None:
    direct = label_space_manifest(
        ablation_id=ABLATION_DIRECT,
        training_class_space="merged_2class",
        evaluation_class_space="merged_2class",
    )
    merged = label_space_manifest(
        ablation_id=ABLATION_MERGED,
        training_class_space="original_10class",
        evaluation_class_space="merged_2class",
    )
    assert direct["category_mapping_hash"] != merged["category_mapping_hash"]
    with pytest.raises(ValueError, match="not directly comparable"):
        require_same_label_granularity(direct, merged)
