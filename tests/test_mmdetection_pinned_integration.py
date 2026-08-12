"""Optional integration checks against the exact pinned MMDetection checkout."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from scripts.run_mmdetection import (
    AERIAL_METRIC_NAME,
    append_aerial_evaluator,
    configure_dataset,
    configure_evaluator_annotation,
    configure_faster_rcnn_from_mask,
    register_aerial_metric,
)


def _pinned_config() -> Path:
    root = Path(os.environ.get("MMDET_ROOT", ""))
    return root / "configs" / "swin" / "mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py"


def test_pinned_mask_pipeline_accepts_empty_visdrone_segmentations(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mmengine")
    pytest.importorskip("mmdet")
    config_path = _pinned_config()
    if not config_path.is_file():
        pytest.skip("set MMDET_ROOT to the pinned MMDetection checkout")

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner

    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (64, 64), "white").save(image_root / "sample.jpg")
    annotation = tmp_path / "instances.json"
    annotation.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "sample.jpg", "width": 64, "height": 64}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [4, 5, 12, 13],
                        "area": 156,
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                ],
                "categories": [{"id": 1, "name": "person"}],
            }
        ),
        encoding="utf-8",
    )

    cfg = Config.fromfile(str(config_path))
    configure_faster_rcnn_from_mask(cfg)
    for name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        loader = getattr(cfg, name)
        configure_dataset(loader.dataset, annotation, image_root, ["person"], 64)
        loader.batch_size = 1
        loader.num_workers = 0
        loader.persistent_workers = False
    assert configure_evaluator_annotation(cfg.val_evaluator, str(annotation)) > 0
    # The production registration path, not a hand-rolled import: assigning
    # custom_imports to a loaded config imports nothing, and building the
    # evaluator is the step that used to fail with an unregistered metric.
    imports = register_aerial_metric(cfg)
    assert "src.evaluation.mmdet_aerial_metric" in imports["imports"]
    cfg.val_evaluator = append_aerial_evaluator(cfg.val_evaluator, str(annotation))

    init_default_scope("mmdet")
    batch = next(iter(Runner.build_dataloader(cfg.train_dataloader)))
    evaluator = Runner.build_evaluator(cfg.val_evaluator)

    assert batch["data_samples"][0].gt_instances.bboxes.shape[0] == 1
    assert evaluator is not None
    assert AERIAL_METRIC_NAME in {
        type(metric).__name__ for metric in evaluator.metrics
    }
