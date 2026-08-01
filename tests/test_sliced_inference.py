from __future__ import annotations

from src.data.tiling import TileWindow
from src.inference.sliced import (
    merge_predictions,
    restore_prediction,
    run_sliced_inference,
    sliced_output_paths,
)


def test_one_slice_restores_equivalent_box_and_class() -> None:
    local = {"image_id": 99, "category_id": 2, "bbox": [10, 20, 30, 40], "score": 0.9}
    restored = restore_prediction(
        local,
        TileWindow(0, 0, 100, 100),
        image_id=7,
        image_width=100,
        image_height=100,
    )
    assert restored["bbox"] == local["bbox"]
    assert restored["category_id"] == 2
    assert restored["image_id"] == 7


def test_multi_slice_offsets_and_class_aware_merge() -> None:
    first = restore_prediction(
        {"category_id": 1, "bbox": [40, 10, 20, 20], "score": 0.9},
        TileWindow(0, 0, 60, 60),
        image_id=1,
        image_width=100,
        image_height=60,
    )
    duplicate = restore_prediction(
        {"category_id": 1, "bbox": [0, 10, 20, 20], "score": 0.8},
        TileWindow(40, 0, 60, 60),
        image_id=1,
        image_width=100,
        image_height=60,
    )
    other_class = {**duplicate, "category_id": 2, "score": 0.7}
    merged = merge_predictions([duplicate, other_class, first], iou_threshold=0.5)
    assert [(row["category_id"], row["score"]) for row in merged] == [(1, 0.9), (2, 0.7)]
    assert merged[0]["bbox"] == [40.0, 10.0, 20.0, 20.0]


def test_sliced_latency_and_output_namespaces_are_isolated(tmp_path) -> None:
    result = run_sliced_inference(
        image_id=1,
        image_width=60,
        image_height=60,
        tile_size=60,
        overlap=0,
        iou_threshold=0.5,
        predict_slice=lambda tile: [
            {"category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.5}
        ],
    )
    assert result["latency"]["slice_count"] == 1
    assert result["inference_mode"] == "sliced"
    paths = sliced_output_paths(tmp_path, "run-1")
    assert "sliced" in paths["predictions"].parts
    assert paths["predictions"] != tmp_path / "predictions" / "run-1" / "predictions.json"
    assert paths["metrics"] != tmp_path / "evaluation" / "run-1" / "metrics.json"
