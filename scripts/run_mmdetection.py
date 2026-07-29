#!/usr/bin/env python
"""Build a runtime MMDetection 3.x config and train with MMEngine Runner."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from src.training.callbacks import save_training_curves
from src.training.checkpointing import materialize_checkpoint_alias


def update_resize(pipeline: Any, image_size: int) -> Any:
    """Recursively set public MMDetection resize transforms."""
    if not isinstance(pipeline, list):
        return pipeline
    for transform in pipeline:
        if not isinstance(transform, dict):
            continue
        if transform.get("type") in {"Resize", "RandomResize"}:
            transform["scale"] = (image_size, image_size)
            transform["keep_ratio"] = True
        for key in ("transforms", "pipeline"):
            if key in transform:
                update_resize(transform[key], image_size)
    return pipeline


def configure_dataset(
    cfg: Any,
    annotation_file: Path,
    image_root: Path,
    classes: list[str],
    image_size: int,
) -> None:
    """Configure nested MMDetection dataset wrappers for COCO input."""
    if "dataset" in cfg:
        configure_dataset(
            cfg["dataset"], annotation_file, image_root, classes, image_size
        )
        return
    cfg["type"] = "CocoDataset"
    cfg["data_root"] = ""
    cfg["ann_file"] = str(annotation_file)
    cfg["data_prefix"] = {"img": str(image_root) + "/"}
    cfg["metainfo"] = {"classes": tuple(classes)}
    if "pipeline" in cfg:
        update_resize(cfg["pipeline"], image_size)


def set_num_classes(node: Any, number_of_classes: int) -> None:
    """Set every detector head's ``num_classes`` field recursively."""
    if isinstance(node, dict):
        if "num_classes" in node:
            node["num_classes"] = number_of_classes
        for value in node.values():
            set_num_classes(value, number_of_classes)
    elif isinstance(node, list):
        for value in node:
            set_num_classes(value, number_of_classes)


def _last_epoch_checkpoint(run_dir: Path) -> Path | None:
    pointer = run_dir / "last_checkpoint"
    if pointer.exists():
        raw = pointer.read_text(encoding="utf-8").strip()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        if candidate.exists():
            return candidate
    epoch_files = sorted(
        run_dir.glob("epoch_*.pth"),
        key=lambda path: int(re.search(r"epoch_(\d+)", path.name).group(1))
        if re.search(r"epoch_(\d+)", path.name)
        else -1,
    )
    return epoch_files[-1] if epoch_files else None


def _best_checkpoint(run_dir: Path, token: str) -> Path | None:
    candidates = sorted(
        [
            path
            for path in run_dir.glob("best_*.pth")
            if token.lower() in path.name.lower()
        ],
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _read_scalar_rows(run_dir: Path) -> list[dict[str, Any]]:
    scalar_file = run_dir / "vis_data" / "scalars.json"
    if not scalar_file.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in scalar_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _write_history(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with (run_dir / "epoch_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    fields = sorted({key for row in rows for key in row})
    with (run_dir / "metrics_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    save_training_curves(rows, run_dir / "training_curves.png")



def _set_recursive(node: Any, key: str, value: Any) -> int:
    changed = 0
    if isinstance(node, dict):
        if key in node:
            node[key] = value
            changed += 1
        for child in node.values():
            changed += _set_recursive(child, key, value)
    elif isinstance(node, list):
        for child in node:
            changed += _set_recursive(child, key, value)
    return changed


def apply_overrides(cfg: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply only documented backend fields and report unsupported keys."""
    applied: dict[str, Any] = {}
    unsupported: dict[str, Any] = {}
    optimizer = cfg.optim_wrapper.setdefault("optimizer", {})
    if "learning_rate" in overrides:
        optimizer["lr"] = float(overrides["learning_rate"]); applied["learning_rate"] = optimizer["lr"]
    if "weight_decay" in overrides:
        optimizer["weight_decay"] = float(overrides["weight_decay"]); applied["weight_decay"] = optimizer["weight_decay"]
    if "gradient_clip" in overrides:
        cfg.optim_wrapper["clip_grad"] = {"max_norm": float(overrides["gradient_clip"]), "norm_type": 2}; applied["gradient_clip"] = overrides["gradient_clip"]
    if "backbone_lr_multiplier" in overrides:
        paramwise = dict(cfg.optim_wrapper.get("paramwise_cfg", {})); custom = dict(paramwise.get("custom_keys", {})); custom["backbone"] = {"lr_mult": float(overrides["backbone_lr_multiplier"])}; paramwise["custom_keys"] = custom; cfg.optim_wrapper["paramwise_cfg"] = paramwise; applied["backbone_lr_multiplier"] = overrides["backbone_lr_multiplier"]
    if "max_detections" in overrides:
        count = _set_recursive(cfg.model, "max_per_img", int(overrides["max_detections"])); (applied if count else unsupported)["max_detections"] = overrides["max_detections"]
    if "anchor_sizes" in overrides:
        count = _set_recursive(cfg.model, "base_sizes", list(overrides["anchor_sizes"])); (applied if count else unsupported)["anchor_sizes"] = overrides["anchor_sizes"]
    if "anchor_ratios" in overrides:
        count = _set_recursive(cfg.model, "ratios", list(overrides["anchor_ratios"])); (applied if count else unsupported)["anchor_ratios"] = overrides["anchor_ratios"]
    if "rpn_nms_threshold" in overrides:
        count = _set_recursive(cfg.model, "iou_threshold", float(overrides["rpn_nms_threshold"])); (applied if count else unsupported)["rpn_nms_threshold"] = overrides["rpn_nms_threshold"]
    if "rpn_proposals" in overrides:
        count = _set_recursive(cfg.model, "nms_pre", int(overrides["rpn_proposals"])); (applied if count else unsupported)["rpn_proposals"] = overrides["rpn_proposals"]
    if "roi_score_threshold" in overrides:
        count = _set_recursive(cfg.model, "score_thr", float(overrides["roi_score_threshold"])); (applied if count else unsupported)["roi_score_threshold"] = overrides["roi_score_threshold"]
    if "roi_nms_threshold" in overrides:
        # This is deliberately applied only to RCNN test NMS, not every NMS node.
        try:
            cfg.model.test_cfg.rcnn.nms.iou_threshold = float(overrides["roi_nms_threshold"]); applied["roi_nms_threshold"] = overrides["roi_nms_threshold"]
        except Exception:
            unsupported["roi_nms_threshold"] = overrides["roi_nms_threshold"]
    if "drop_path_rate" in overrides:
        count = _set_recursive(cfg.model.get("backbone", {}), "drop_path_rate", float(overrides["drop_path_rate"])); (applied if count else unsupported)["drop_path_rate"] = overrides["drop_path_rate"]
    if "dropout" in overrides:
        count = _set_recursive(cfg.model, "dropout", float(overrides["dropout"])); (applied if count else unsupported)["dropout"] = overrides["dropout"]
    if "p2_enabled" in overrides:
        try:
            cfg.model.neck.start_level = 0 if bool(overrides["p2_enabled"]) else 1
            applied["p2_enabled"] = bool(overrides["p2_enabled"])
        except Exception:
            unsupported["p2_enabled"] = overrides["p2_enabled"]
    # These require recipe-specific pipelines/constructors and are intentionally not guessed.
    for key in ("augmentation_strength", "layerwise_lr_decay", "warmup_epochs"):
        if key in overrides:
            unsupported[key] = overrides[key]
    return {"applied": applied, "unsupported": unsupported}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-ann", required=True)
    parser.add_argument("--val-ann", required=True)
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--accumulation", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--strip-mask-branch", action="store_true")
    parser.add_argument("--registration-import")
    parser.add_argument("--pretrained-weight")
    parser.add_argument("--clear-pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overrides", default="{}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
    except ImportError as exc:
        raise RuntimeError(
            "Install mmengine, mmdet 3.3.0, and a CUDA-matched mmcv wheel."
        ) from exc

    base_config = Path(args.base_config).resolve()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Both the official MMDetection and VMamba detection layouts put configs two
    # directories below the import root.
    detection_root = base_config.parents[2]
    sys.path.insert(0, str(detection_root))
    if args.registration_import:
        __import__(args.registration_import)

    cfg = Config.fromfile(str(base_config))
    if args.pretrained_weight:
        cfg.model.backbone.pretrained = str(Path(args.pretrained_weight).resolve())
        if "init_cfg" in cfg.model.backbone:
            cfg.model.backbone.init_cfg = None
    elif args.clear_pretrained:
        if "pretrained" in cfg.model.backbone:
            cfg.model.backbone.pretrained = None
        if "init_cfg" in cfg.model.backbone:
            cfg.model.backbone.init_cfg = None
        (run_dir / "PRETRAINING_WARNING.txt").write_text(
            "No verified pretrained checkpoint was supplied; the backbone is "
            "trained from scratch. Do not compare this run directly with "
            "pretrained-recipe runs.\n",
            encoding="utf-8",
        )
    classes = json.loads(args.classes)
    overrides = json.loads(args.overrides)
    override_report = apply_overrides(cfg, overrides)
    (run_dir / "applied_overrides.json").write_text(
        json.dumps(override_report, indent=2), encoding="utf-8"
    )
    if args.strip_mask_branch:
        cfg.model.type = "FasterRCNN"
        if "roi_head" in cfg.model:
            cfg.model.roi_head.pop("mask_roi_extractor", None)
            cfg.model.roi_head.pop("mask_head", None)
    set_num_classes(cfg.model, len(classes))
    configure_dataset(
        cfg.train_dataloader.dataset,
        Path(args.train_ann),
        Path(args.train_images),
        classes,
        args.image_size,
    )
    configure_dataset(
        cfg.val_dataloader.dataset,
        Path(args.val_ann),
        Path(args.val_images),
        classes,
        args.image_size,
    )
    configure_dataset(
        cfg.test_dataloader.dataset,
        Path(args.val_ann),
        Path(args.val_images),
        classes,
        args.image_size,
    )

    cfg.train_dataloader.batch_size = args.batch_size
    cfg.train_dataloader.num_workers = min(
        int(cfg.train_dataloader.get("num_workers", 2)), 2
    )
    cfg.val_dataloader.num_workers = min(
        int(cfg.val_dataloader.get("num_workers", 2)), 2
    )
    cfg.test_dataloader.num_workers = min(
        int(cfg.test_dataloader.get("num_workers", 2)), 2
    )
    cfg.val_evaluator.ann_file = args.val_ann
    cfg.test_evaluator.ann_file = args.val_ann
    cfg.custom_imports = {
        "imports": ["src.evaluation.mmdet_aerial_metric"],
        "allow_failed_imports": False,
    }
    cfg.val_evaluator = [
        cfg.val_evaluator,
        {"type": "AerialCocoMetric", "ann_file": args.val_ann},
    ]
    cfg.test_evaluator = [
        cfg.test_evaluator,
        {"type": "AerialCocoMetric", "ann_file": args.val_ann},
    ]
    if hasattr(cfg, "train_cfg") and cfg.train_cfg:
        cfg.train_cfg.max_epochs = args.epochs
        cfg.train_cfg.val_interval = 1
    cfg.work_dir = str(run_dir)
    cfg.randomness = {"seed": args.seed, "deterministic": False}
    cfg.resume = bool(args.resume)
    if args.resume and (run_dir / "last.pth").exists():
        cfg.load_from = str(run_dir / "last.pth")

    if args.amp:
        base_optimizer = dict(cfg.optim_wrapper)
        base_optimizer["type"] = "AmpOptimWrapper"
        base_optimizer["loss_scale"] = "dynamic"
        cfg.optim_wrapper = base_optimizer
    cfg.optim_wrapper["accumulative_counts"] = args.accumulation
    cfg.default_hooks.checkpoint.update(
        {
            "type": "CheckpointHook",
            "interval": 1,
            "max_keep_ckpts": 5,
            "save_last": True,
            "save_best": ["coco/bbox_mAP", "aerial_coco/APtiny"],
            "rule": "greater",
        }
    )
    cfg.default_hooks.logger.interval = 20
    cfg.visualizer.vis_backends = [
        {"type": "LocalVisBackend"},
        {"type": "TensorboardVisBackend", "save_dir": str(run_dir / "tensorboard")},
    ]
    cfg.dump(str(run_dir / "runtime_config.py"))

    started = time.perf_counter()
    runner = Runner.from_cfg(cfg)
    runner.train()
    elapsed = time.perf_counter() - started

    last_checkpoint = _last_epoch_checkpoint(run_dir)
    best_map = _best_checkpoint(run_dir, "bbox_map") or last_checkpoint
    best_tiny = _best_checkpoint(run_dir, "aptiny") or best_map
    if last_checkpoint:
        materialize_checkpoint_alias(last_checkpoint, run_dir / "last.pth")
    if best_map:
        materialize_checkpoint_alias(best_map, run_dir / "best_map.pth")
    if best_tiny:
        materialize_checkpoint_alias(best_tiny, run_dir / "best_aptiny.pth")

    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameters_by_module = {
        name: int(sum(parameter.numel() for parameter in module.parameters()))
        for name, module in model.named_children()
    }

    rows = _read_scalar_rows(run_dir)
    _write_history(run_dir, rows)
    best_map_value = 0.0
    best_tiny_value = 0.0
    best_epoch = 0
    for row in rows:
        value = row.get("coco/bbox_mAP", row.get("bbox_mAP"))
        if value is not None and float(value) > best_map_value:
            best_map_value = float(value)
            best_epoch = int(row.get("epoch", row.get("step", 0)))
        tiny = row.get("aerial_coco/APtiny", row.get("APtiny"))
        if tiny is not None:
            best_tiny_value = max(best_tiny_value, float(tiny))

    summary = {
        "checkpoint_best_map": str(run_dir / "best_map.pth"),
        "checkpoint_best_aptiny": str(run_dir / "best_aptiny.pth"),
        "checkpoint_last": str(run_dir / "last.pth"),
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
        "parameters_by_module": parameters_by_module,
        "estimated_model_size_bytes_fp32": int(total * 4),
        "best_validation_map": best_map_value,
        "best_validation_aptiny": best_tiny_value,
        "best_epoch": best_epoch,
        "framework_training_seconds": elapsed,
        "notes": (
            "APtiny is computed by AerialCocoMetric with area < 16^2 pixels."
        ),
        "hyperparameter_overrides": override_report,
    }
    (run_dir / "final_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
