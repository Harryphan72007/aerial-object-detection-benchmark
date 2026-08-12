from __future__ import annotations

import json
from pathlib import Path

import yaml

import pytest

from scripts.run_mmdetection import (
    AERIAL_METRIC_MODULE,
    apply_detection_policy,
    append_aerial_evaluator,
    best_metrics_from_rows,
    configure_evaluator_annotation,
    configure_faster_rcnn_from_mask,
    disable_default_checkpoint_hook,
    merge_custom_imports,
    _read_scalar_rows,
    restore_selection_state,
)
from src.evaluation.mmdet_aerial_metric import (
    flatten_image_results,
    image_record,
)
from src.evaluation.policy import (
    DETECTION_SCORE_THRESHOLD,
    MAX_DETECTIONS_PER_IMAGE,
)

ROOT = Path(__file__).resolve().parents[1]


class AttrDict(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


def test_checked_in_evaluation_defaults_match_the_central_policy() -> None:
    defaults = yaml.safe_load(
        (ROOT / "configs" / "common" / "evaluation_defaults.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert defaults["score_thr"] == DETECTION_SCORE_THRESHOLD
    assert defaults["confidence_threshold"] == DETECTION_SCORE_THRESHOLD
    assert defaults["max_per_img"] == MAX_DETECTIONS_PER_IMAGE
    assert defaults["max_detections"][-1] == MAX_DETECTIONS_PER_IMAGE


def test_mask_rcnn_config_is_converted_to_bbox_only_recursively() -> None:
    cfg = AttrDict(
        model=AttrDict(
            type="MaskRCNN",
            roi_head={
                "bbox_head": {"num_classes": 80},
                "mask_roi_extractor": {"type": "SingleRoIExtractor"},
                "mask_head": {"type": "FCNMaskHead"},
            },
        ),
        train_dataloader={
            "dataset": {
                "pipeline": [
                    {"type": "LoadAnnotations", "with_mask": True, "poly2mask": True},
                    {"type": "CopyPaste"},
                ]
            }
        },
        val_dataloader={"dataset": {"pipeline": []}},
        test_dataloader={"dataset": {"pipeline": []}},
        val_evaluator={"type": "CocoMetric", "metric": ["bbox", "segm"]},
        test_evaluator=[{"type": "CocoMetric", "metric": "segm"}],
    )

    report = configure_faster_rcnn_from_mask(cfg)

    assert cfg.model.type == "FasterRCNN"
    assert "mask_head" not in cfg.model["roi_head"]
    assert "mask_roi_extractor" not in cfg.model["roi_head"]
    pipeline = cfg.train_dataloader["dataset"]["pipeline"]
    assert pipeline == [{"type": "LoadAnnotations", "with_mask": False}]
    assert cfg.val_evaluator["metric"] == "bbox"
    assert cfg.test_evaluator[0]["metric"] == "bbox"
    assert all(value > 0 for value in report.values())


def test_detection_policy_and_nested_evaluator_annotation_are_consistent() -> None:
    model = {
        "test_cfg": {
            "rpn": {"score_thr": 0.0, "max_per_img": 1000},
            "rcnn": {"score_thr": 0.05, "max_per_img": 100},
        },
    }
    report = apply_detection_policy(model)
    assert model["test_cfg"]["rcnn"]["score_thr"] == DETECTION_SCORE_THRESHOLD
    assert model["test_cfg"]["rcnn"]["max_per_img"] == MAX_DETECTIONS_PER_IMAGE
    assert model["test_cfg"]["rpn"] == {"score_thr": 0.0, "max_per_img": 1000}
    assert report["max_detections_per_image"] == MAX_DETECTIONS_PER_IMAGE

    evaluators = [{"type": "CocoMetric"}, {"metrics": [{"ann_file": "old"}]}]
    assert configure_evaluator_annotation(evaluators, "validation.json") == 2
    assert evaluators[0]["ann_file"] == "validation.json"
    assert evaluators[1]["metrics"][0]["ann_file"] == "validation.json"
    appended = append_aerial_evaluator(evaluators, "validation.json")
    assert len(appended) == 3
    assert appended[-1]["type"] == "AerialCocoMetric"
    assert all(not isinstance(value, list) for value in appended)


class FakeTensor:
    """Stand in for the device-resident tensors MMDetection puts in predictions."""

    def __init__(self, rows: list) -> None:
        self.rows = rows
        self.moved = False

    def cpu(self) -> "FakeTensor":
        moved = FakeTensor(self.rows)
        moved.moved = True
        return moved

    def numpy(self) -> list:
        if not self.moved:
            raise AssertionError("prediction fields must be moved to CPU first")
        return self.rows


def test_metric_reads_the_flattened_data_samples_mmengine_actually_passes() -> None:
    # ``Evaluator.process`` converts every DetDataSample with ``to_dict()``,
    # which lifts metainfo to the top level and turns pred_instances into a
    # plain dict of tensors.
    sample = {
        "img_id": 77,
        "ori_shape": (1080, 1920),
        "pred_instances": {
            "bboxes": FakeTensor([[10.0, 20.0, 30.0, 50.0]]),
            "scores": FakeTensor([0.75]),
            "labels": FakeTensor([1]),
        },
    }

    record = image_record(sample)

    assert record == {
        "image_id": 77,
        "detections": [
            {
                "image_id": 77,
                "category_id": 2,
                "bbox": [10.0, 20.0, 20.0, 30.0],
                "score": 0.75,
            }
        ],
    }


def test_metric_still_reads_data_elements_that_nest_metainfo() -> None:
    class Sample:
        metainfo = {"img_id": 5}
        pred_instances = None

    assert image_record(Sample()) == {"image_id": 5, "detections": []}


def test_metric_refuses_samples_without_an_image_id_instead_of_guessing() -> None:
    with pytest.raises(KeyError, match="img_id"):
        image_record({"pred_instances": None})


def test_custom_imports_registers_the_metric_without_dropping_upstream_modules() -> None:
    merged = merge_custom_imports({"imports": ["model"], "allow_failed_imports": True})
    assert merged == {
        "imports": ["model", AERIAL_METRIC_MODULE],
        "allow_failed_imports": False,
    }
    assert merge_custom_imports(None)["imports"] == [AERIAL_METRIC_MODULE]
    assert merge_custom_imports(merged)["imports"] == ["model", AERIAL_METRIC_MODULE]


def test_default_checkpoint_hook_is_disabled_rather_than_removed() -> None:
    # MMEngine re-adds a stock CheckpointHook for any default name that is
    # simply absent, so only an explicit None keeps epoch_*.pth off disk.
    hooks = {"checkpoint": {"type": "CheckpointHook", "interval": 1}}
    disable_default_checkpoint_hook(hooks)
    assert hooks == {"checkpoint": None}


def test_metric_results_keep_empty_images_until_compute_boundary() -> None:
    bundled = [
        {"image_id": 1, "detections": []},
        {
            "image_id": 2,
            "detections": [
                {
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "score": 0.9,
                }
                for _ in range(7)
            ],
        },
    ]
    assert len(bundled) == 2
    flattened = flatten_image_results(bundled)
    assert flattened == bundled[1]["detections"]
    assert len(flattened) == 7
    assert len(flattened) > len(bundled)


def test_timestamped_scalar_history_selects_one_authoritative_metric_pair(
    tmp_path: Path,
) -> None:
    scalar = tmp_path / "20260803_120000" / "vis_data" / "scalars.json"
    scalar.parent.mkdir(parents=True)
    rows = [
        {"epoch": 1, "coco/bbox_mAP": 0.20, "aerial_coco/APtiny": 0.19},
        {"epoch": 2, "coco/bbox_mAP": 0.25, "aerial_coco/APtiny": 0.10},
        {"epoch": 3, "coco/bbox_mAP": 0.22, "aerial_coco/APtiny": 0.30},
    ]
    scalar.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    loaded = _read_scalar_rows(tmp_path)
    assert loaded == rows
    assert best_metrics_from_rows(loaded) == (0.25, 0.10, 2)


def test_missing_metric_pair_fails_instead_of_silently_returning_zeros() -> None:
    try:
        best_metrics_from_rows([{"epoch": 1, "loss": 2.0}])
    except RuntimeError as error:
        assert "no same-epoch" in str(error)
    else:
        raise AssertionError("missing validation metrics must fail")


def test_resume_restores_map_aptiny_and_epoch_as_one_selection_state() -> None:
    state = {
        "best_checkpoint_metadata": {
            "metric_value": 0.31,
            "aptiny": 0.17,
            "epoch": 4,
        },
        "best_model_state_dict": {"weight": "fixture"},
    }
    selected: dict[str, object] = {}
    restore_selection_state(state, selected)
    assert selected == {
        "best_metric": 0.31,
        "best_aptiny": 0.17,
        "best_epoch": 4,
        "best_model_state_dict": {"weight": "fixture"},
    }
