#!/usr/bin/env python
"""Build a runtime MMDetection 3.x config and train with MMEngine Runner."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from src.optional_outputs import run_optional_output
from src.evaluation.policy import (
    DETECTION_SCORE_THRESHOLD,
    MAX_DETECTIONS_PER_IMAGE,
    detection_policy,
)
from src.training.callbacks import safe_save_training_curves
from src.training.checkpointing import (
    atomic_torch_save,
    materialize_canonical_best,
)
from src.runtime_manifest import write_runtime_environment_manifest
from src.utils.serialization import (
    read_yaml,
    write_csv,
    write_json,
    write_text_atomic,
)


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


MASK_ONLY_TRANSFORMS = {"CopyPaste", "LoadPanopticAnnotations"}


def strip_mask_pipeline(node: Any) -> int:
    """Disable mask-only settings throughout nested MMDetection pipelines."""

    changed = 0
    if isinstance(node, list):
        retained = []
        for value in node:
            if isinstance(value, dict) and value.get("type") in MASK_ONLY_TRANSFORMS:
                changed += 1
                continue
            changed += strip_mask_pipeline(value)
            retained.append(value)
        node[:] = retained
    elif isinstance(node, dict):
        if node.get("type") == "LoadAnnotations" and node.get("with_mask") is not False:
            node["with_mask"] = False
            node.pop("poly2mask", None)
            changed += 1
        for value in node.values():
            changed += strip_mask_pipeline(value)
    return changed


def _remove_mask_model_keys(node: Any) -> int:
    changed = 0
    if isinstance(node, dict):
        for key in ("mask_roi_extractor", "mask_head"):
            if key in node:
                node.pop(key)
                changed += 1
        for value in node.values():
            changed += _remove_mask_model_keys(value)
    elif isinstance(node, list):
        for value in node:
            changed += _remove_mask_model_keys(value)
    return changed


def configure_bbox_only_evaluator(node: Any) -> int:
    """Force inherited COCO evaluators to consume detector bbox outputs only."""

    changed = 0
    if isinstance(node, dict):
        if node.get("type") == "CocoMetric" and node.get("metric") != "bbox":
            node["metric"] = "bbox"
            changed += 1
        for value in node.values():
            changed += configure_bbox_only_evaluator(value)
    elif isinstance(node, list):
        for value in node:
            changed += configure_bbox_only_evaluator(value)
    return changed


def configure_faster_rcnn_from_mask(cfg: Any) -> dict[str, int]:
    """Convert a Mask R-CNN config into a bbox-only Faster R-CNN config."""

    cfg.model.type = "FasterRCNN"
    removed_heads = _remove_mask_model_keys(cfg.model)
    pipeline_changes = sum(
        strip_mask_pipeline(getattr(cfg, name))
        for name in ("train_dataloader", "val_dataloader", "test_dataloader")
        if hasattr(cfg, name)
    )
    evaluator_changes = sum(
        configure_bbox_only_evaluator(getattr(cfg, name))
        for name in ("val_evaluator", "test_evaluator")
        if hasattr(cfg, name)
    )
    return {
        "removed_mask_heads": removed_heads,
        "pipeline_changes": pipeline_changes,
        "evaluator_changes": evaluator_changes,
    }


def configure_evaluator_annotation(node: Any, annotation_file: str) -> int:
    changed = 0
    if isinstance(node, dict):
        if node.get("type") == "CocoMetric" or "ann_file" in node:
            node["ann_file"] = annotation_file
            changed += 1
        for value in node.values():
            changed += configure_evaluator_annotation(value, annotation_file)
    elif isinstance(node, list):
        for value in node:
            changed += configure_evaluator_annotation(value, annotation_file)
    return changed


def append_aerial_evaluator(node: Any, annotation_file: str) -> list[Any]:
    """Append the custom evaluator without creating unsupported nested lists."""

    evaluators = list(node) if isinstance(node, (list, tuple)) else [node]
    evaluators.append({"type": "AerialCocoMetric", "ann_file": annotation_file})
    return evaluators


def apply_detection_policy(model: Any) -> dict[str, Any]:
    """Apply the immutable benchmark prediction policy to an RCNN config."""

    try:
        test_cfg = model["test_cfg"]
        rcnn = test_cfg["rcnn"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "MMDetection RCNN config does not expose model.test_cfg.rcnn"
        ) from error
    rcnn["max_per_img"] = MAX_DETECTIONS_PER_IMAGE
    rcnn["score_thr"] = DETECTION_SCORE_THRESHOLD
    return {
        **detection_policy(),
        "applied_nodes": {"max_per_img": 1, "score_thr": 1},
    }


def _read_scalar_rows(
    run_dir: Path, log_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    candidates = {run_dir / "vis_data" / "scalars.json"}
    candidates.update(run_dir.glob("*/vis_data/scalars.json"))
    if log_dir is not None:
        candidates.add(Path(log_dir) / "vis_data" / "scalars.json")
    rows: list[dict[str, Any]] = []
    for scalar_file in sorted(path for path in candidates if path.is_file()):
        for line in scalar_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def validation_metric_pair(metrics: dict[str, Any]) -> tuple[float | None, float | None]:
    map_value = metrics.get("coco/bbox_mAP", metrics.get("bbox_mAP"))
    tiny_value = metrics.get("aerial_coco/APtiny", metrics.get("APtiny"))
    return (
        None if map_value is None else float(map_value),
        None if tiny_value is None else float(tiny_value),
    )


def best_metrics_from_rows(rows: list[dict[str, Any]]) -> tuple[float, float, int]:
    """Select mAP and APtiny from one common best-mAP history row."""

    selected: tuple[float, float, int] | None = None
    for row in rows:
        map_value, tiny_value = validation_metric_pair(row)
        if map_value is None or tiny_value is None:
            continue
        epoch = int(row.get("epoch", row.get("step", 0)))
        candidate = (map_value, tiny_value, epoch)
        if selected is None or candidate[:2] > selected[:2]:
            selected = candidate
    if selected is None:
        raise RuntimeError(
            "metric history contains no same-epoch validation mAP/APtiny pair"
        )
    return selected


def restore_selection_state(
    checkpoint_state: dict[str, Any], selection_state: dict[str, Any]
) -> None:
    """Restore the authoritative checkpoint-selection pair for a resumed run."""

    metadata = checkpoint_state.get("best_checkpoint_metadata", {})
    selection_state["best_metric"] = float(
        metadata.get("metric_value", float("-inf"))
    )
    selection_state["best_aptiny"] = float(metadata.get("aptiny", float("-inf")))
    selection_state["best_epoch"] = int(metadata.get("epoch", 0))
    selection_state["best_model_state_dict"] = checkpoint_state.get(
        "best_model_state_dict"
    )


def _write_history(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    write_text_atomic(
        run_dir / "epoch_metrics.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    fields = sorted({key for row in rows for key in row})
    write_csv(run_dir / "metrics_history.csv", rows, fields)



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


def configure_scheduler_horizon(
    schedulers: Any, original_max_epochs: int, scheduler_horizon: int
) -> Any:
    """Retarget epoch-based schedules once while preserving their relative policy."""
    if original_max_epochs <= 0 or scheduler_horizon <= 0:
        raise ValueError("scheduler horizons must be positive")
    items = schedulers if isinstance(schedulers, list) else [schedulers]
    for scheduler in items:
        if not isinstance(scheduler, dict):
            continue
        if bool(scheduler.get("by_epoch", True)):
            if "begin" in scheduler:
                scheduler["begin"] = round(
                    int(scheduler["begin"]) * scheduler_horizon / original_max_epochs
                )
            if "end" in scheduler:
                scheduler["end"] = round(
                    int(scheduler["end"]) * scheduler_horizon / original_max_epochs
                )
            if "milestones" in scheduler:
                scheduler["milestones"] = [
                    max(
                        1,
                        round(
                            int(milestone)
                            * scheduler_horizon
                            / original_max_epochs
                        ),
                    )
                    for milestone in scheduler["milestones"]
                ]
            if "T_max" in scheduler:
                scheduler["T_max"] = scheduler_horizon
    return schedulers


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
        try:
            cfg.model.test_cfg.rcnn.max_per_img = int(overrides["max_detections"]); applied["max_detections"] = overrides["max_detections"]
        except Exception:
            unsupported["max_detections"] = overrides["max_detections"]
    if "anchor_sizes" in overrides:
        count = _set_recursive(cfg.model, "base_sizes", list(overrides["anchor_sizes"])); (applied if count else unsupported)["anchor_sizes"] = overrides["anchor_sizes"]
    if "anchor_ratios" in overrides:
        count = _set_recursive(cfg.model, "ratios", list(overrides["anchor_ratios"])); (applied if count else unsupported)["anchor_ratios"] = overrides["anchor_ratios"]
    if "rpn_nms_threshold" in overrides:
        count = _set_recursive(cfg.model, "iou_threshold", float(overrides["rpn_nms_threshold"])); (applied if count else unsupported)["rpn_nms_threshold"] = overrides["rpn_nms_threshold"]
    if "rpn_proposals" in overrides:
        count = _set_recursive(cfg.model, "nms_pre", int(overrides["rpn_proposals"])); (applied if count else unsupported)["rpn_proposals"] = overrides["rpn_proposals"]
    if "roi_score_threshold" in overrides:
        try:
            cfg.model.test_cfg.rcnn.score_thr = float(overrides["roi_score_threshold"]); applied["roi_score_threshold"] = overrides["roi_score_threshold"]
        except Exception:
            unsupported["roi_score_threshold"] = overrides["roi_score_threshold"]
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
    parser.add_argument("--model-id", required=True)
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
    parser.add_argument("--scheduler-horizon", type=int, required=True)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--lr-range-test-steps", type=int, default=0)
    parser.add_argument("--lr-range-output")
    parser.add_argument("--lr-range-start-multiplier", type=float, default=0.01)
    parser.add_argument("--lr-range-end-multiplier", type=float, default=20.0)
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
    if args.scheduler_horizon < args.epochs:
        raise ValueError("--scheduler-horizon must be >= --epochs")
    if args.validation_interval < 0:
        raise ValueError("--validation-interval must be non-negative")
    try:
        import torch
        from mmengine.config import Config
        from mmengine.hooks import Hook
        from mmengine.runner import Runner
    except ImportError as exc:
        raise RuntimeError(
            "Install mmengine, mmdet 3.3.0, and a CUDA-matched mmcv wheel."
        ) from exc

    base_config = Path(args.base_config).resolve()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    training_config = read_yaml(run_dir / "training_config.yaml")
    configuration_hash = str(training_config["configuration_hash"])
    write_runtime_environment_manifest(run_dir, args.model_id, Path.cwd())
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
    if "max_detections" in overrides and int(overrides["max_detections"]) != MAX_DETECTIONS_PER_IMAGE:
        raise ValueError(
            f"max_detections is fixed at {MAX_DETECTIONS_PER_IMAGE} by the benchmark policy"
        )
    if "roi_score_threshold" in overrides and not math.isclose(
        float(overrides["roi_score_threshold"]), DETECTION_SCORE_THRESHOLD
    ):
        raise ValueError(
            f"roi_score_threshold is fixed at {DETECTION_SCORE_THRESHOLD} by the benchmark policy"
        )
    override_report = apply_overrides(cfg, overrides)
    if args.strip_mask_branch:
        override_report["mask_conversion"] = configure_faster_rcnn_from_mask(cfg)
    override_report["evaluation_policy"] = apply_detection_policy(cfg.model)
    (run_dir / "applied_overrides.json").write_text(
        json.dumps(override_report, indent=2), encoding="utf-8"
    )
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
    if not configure_evaluator_annotation(cfg.val_evaluator, args.val_ann):
        raise ValueError("validation config contains no COCO evaluator")
    if not configure_evaluator_annotation(cfg.test_evaluator, args.val_ann):
        raise ValueError("test config contains no COCO evaluator")
    cfg.custom_imports = {
        "imports": ["src.evaluation.mmdet_aerial_metric"],
        "allow_failed_imports": False,
    }
    cfg.val_evaluator = append_aerial_evaluator(cfg.val_evaluator, args.val_ann)
    cfg.test_evaluator = append_aerial_evaluator(cfg.test_evaluator, args.val_ann)
    if hasattr(cfg, "train_cfg") and cfg.train_cfg:
        original_max_epochs = int(cfg.train_cfg.max_epochs)
        cfg.param_scheduler = configure_scheduler_horizon(
            cfg.get("param_scheduler", []),
            original_max_epochs,
            args.scheduler_horizon,
        )
        cfg.train_cfg.max_epochs = args.epochs
        cfg.train_cfg.val_interval = (
            args.validation_interval
            if args.validation_interval > 0
            else args.epochs + 1
        )
    if args.lr_range_test_steps:
        if args.resume:
            raise ValueError("the LR range test must start from pretrained weights")
        cfg.train_cfg = {
            "type": "IterBasedTrainLoop",
            "max_iters": args.lr_range_test_steps * args.accumulation,
            "val_interval": args.lr_range_test_steps * args.accumulation + 1,
        }
        cfg.param_scheduler = []
    cfg.work_dir = str(run_dir)
    cfg.randomness = {"seed": args.seed, "deterministic": True}
    cfg.resume = bool(args.resume)
    if args.resume and (run_dir / "last.pth").exists():
        cfg.load_from = str(run_dir / "last.pth")

    if args.amp:
        base_optimizer = dict(cfg.optim_wrapper)
        base_optimizer["type"] = "AmpOptimWrapper"
        base_optimizer["loss_scale"] = "dynamic"
        cfg.optim_wrapper = base_optimizer
    cfg.optim_wrapper["accumulative_counts"] = args.accumulation
    cfg.default_hooks.pop("checkpoint", None)
    write_json(
        run_dir / "checkpoint_policy.json",
        {
            "implementation": "AtomicRollingCheckpointHook",
            "max_keep_ckpts": 1,
            "save_last": True,
            "save_best": None,
            "filename": "last.pth",
            "selection_metric": "validation_mAP",
        },
    )
    cfg.default_hooks.logger.interval = 20
    cfg.visualizer.vis_backends = [{"type": "LocalVisBackend"}]
    cfg.visualizer.save_dir = str(run_dir)
    tensorboard, _ = run_optional_output(
        "configure_tensorboard",
        run_dir,
        lambda: __import__("tensorboard"),
    )
    if tensorboard is not None:
        cfg.visualizer.vis_backends.append(
            {
                "type": "TensorboardVisBackend",
                "save_dir": str(run_dir / "tensorboard"),
            }
        )
    cfg.dump(str(run_dir / "runtime_config.py"))
    scheduler_contract_path = run_dir / "scheduler_contract.json"
    if args.resume and scheduler_contract_path.exists():
        prior_contract = json.loads(
            scheduler_contract_path.read_text(encoding="utf-8")
        )
        prior_horizon = int(prior_contract["scheduler_horizon"])
        if prior_horizon != args.scheduler_horizon:
            raise ValueError(
                "scheduler horizon changed across resume: "
                f"checkpoint={prior_horizon}, requested={args.scheduler_horizon}"
            )
    scheduler_contract_path.write_text(
        json.dumps(
            {
                "scheduler_horizon": args.scheduler_horizon,
                "target_epochs": args.epochs,
                "resume": bool(args.resume),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    from src.reproducibility import capture_rng_state, restore_rng_state

    class FullResumeStateHook(Hook):
        """Persist RNG and sampler state not covered by backend-specific weights."""

        priority = "VERY_HIGH"

        def before_save_checkpoint(
            self, runner: Any, checkpoint: dict[str, Any]
        ) -> None:
            checkpoint["benchmark_rng_state"] = capture_rng_state()
            sampler = getattr(runner.train_dataloader, "sampler", None)
            sampler_state: dict[str, Any] = {"epoch": int(runner.epoch)}
            if sampler is not None and hasattr(sampler, "state_dict"):
                sampler_state["state_dict"] = sampler.state_dict()
            checkpoint["benchmark_sampler_state"] = sampler_state
            checkpoint["benchmark_scheduler_horizon"] = args.scheduler_horizon

        def after_load_checkpoint(
            self, runner: Any, checkpoint: dict[str, Any]
        ) -> None:
            checkpoint_horizon = int(
                checkpoint.get(
                    "benchmark_scheduler_horizon", args.scheduler_horizon
                )
            )
            if checkpoint_horizon != args.scheduler_horizon:
                raise ValueError(
                    "scheduler horizon changed across resume checkpoint: "
                    f"{checkpoint_horizon} != {args.scheduler_horizon}"
                )
            if checkpoint.get("benchmark_rng_state"):
                restore_rng_state(checkpoint["benchmark_rng_state"])
            sampler_state = checkpoint.get("benchmark_sampler_state", {})
            sampler = getattr(runner.train_dataloader, "sampler", None)
            if (
                sampler is not None
                and sampler_state.get("state_dict") is not None
                and hasattr(sampler, "load_state_dict")
            ):
                sampler.load_state_dict(sampler_state["state_dict"])
            elif sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(int(sampler_state.get("epoch", runner.epoch)))

    started = time.perf_counter()
    runner = Runner.from_cfg(cfg)
    runner.register_hook(FullResumeStateHook(), priority="VERY_HIGH")
    if args.lr_range_test_steps:
        import torch
        from src.training.lr_range import (
            exponential_lr_schedule,
            save_lr_range_artifacts,
            should_stop_range_test,
        )

        baseline_lr = float(cfg.optim_wrapper.optimizer.lr)
        schedule = exponential_lr_schedule(
            baseline_lr,
            args.lr_range_test_steps,
            args.lr_range_start_multiplier,
            args.lr_range_end_multiplier,
        )
        range_history: list[dict[str, Any]] = []
        original_step = runner.optim_wrapper.step
        range_state: dict[str, Any] = {
            "optimizer_step": 0,
            "pending_losses": [],
        }

        def tracked_step(**kwargs: Any) -> Any:
            squared = torch.zeros((), device=runner.model.data_preprocessor.device)
            for parameter in runner.model.parameters():
                if parameter.grad is not None:
                    squared += parameter.grad.detach().float().norm(2).square()
            runner.optim_wrapper._lr_range_gradient_norm = float(
                squared.sqrt().cpu().item()
            )
            result = original_step(**kwargs)
            range_state["optimizer_step"] += 1
            return result

        runner.optim_wrapper.step = tracked_step

        class LRRangeStop(RuntimeError):
            pass

        class LRRangeHook(Hook):
            priority = "VERY_HIGH"

            def before_train_iter(
                self, runner: Any, batch_idx: int, data_batch: Any = None
            ) -> None:
                learning_rate = schedule[
                    min(range_state["optimizer_step"], len(schedule) - 1)
                ]
                for group in runner.optim_wrapper.optimizer.param_groups:
                    group["lr"] = learning_rate

            def after_train_iter(
                self,
                runner: Any,
                batch_idx: int,
                data_batch: Any = None,
                outputs: Any = None,
            ) -> None:
                loss: float | None = None
                if isinstance(outputs, dict) and outputs.get("loss") is not None:
                    value = outputs["loss"]
                    loss = float(value.detach().item() if hasattr(value, "detach") else value)
                if loss is None:
                    scalar = runner.message_hub.get_scalar("train/loss")
                    loss = float(scalar.current())
                range_state["pending_losses"].append(loss)
                if range_state["optimizer_step"] <= len(range_history):
                    return
                learning_rate = schedule[range_state["optimizer_step"] - 1]
                range_history.append(
                    {
                        "optimizer_step": range_state["optimizer_step"],
                        "learning_rate": learning_rate,
                        "raw_loss": float(
                            sum(range_state["pending_losses"])
                            / len(range_state["pending_losses"])
                        ),
                        "gradient_norm": float(
                            getattr(
                                runner.optim_wrapper,
                                "_lr_range_gradient_norm",
                                math.nan,
                            )
                        ),
                    }
                )
                range_state["pending_losses"] = []
                stop, reason = should_stop_range_test(
                    [row["raw_loss"] for row in range_history]
                )
                if stop:
                    raise LRRangeStop(str(reason))

        runner.register_hook(LRRangeHook(), priority="VERY_HIGH")
        stopped_reason = None
        try:
            runner.train()
        except LRRangeStop as error:
            stopped_reason = str(error)
        output = Path(args.lr_range_output or (run_dir / "lr_range_test"))
        summary = save_lr_range_artifacts(
            output,
            range_history,
            baseline_learning_rate=baseline_lr,
            stopped_reason=stopped_reason,
        )
        print(json.dumps(summary, indent=2))
        return
    selection_state: dict[str, Any] = {
        "best_metric": float("-inf"),
        "best_aptiny": float("-inf"),
        "best_epoch": 0,
        "best_model_state_dict": None,
    }

    class AtomicRollingCheckpointHook(Hook):
        """Atomically overwrite one full-state resume checkpoint each epoch."""

        def after_train_epoch(self, runner: Any) -> None:
            temporary_name = ".last.pth.tmp"
            temporary = run_dir / temporary_name
            runner.save_checkpoint(
                str(run_dir),
                filename=temporary_name,
                save_optimizer=True,
                save_param_scheduler=True,
                meta={"epoch": int(runner.epoch + 1), "iter": int(runner.iter)},
                by_epoch=True,
            )
            state = torch.load(temporary, map_location="cpu", weights_only=False)
            if selection_state["best_model_state_dict"] is not None:
                state["best_model_state_dict"] = selection_state[
                    "best_model_state_dict"
                ]
                state["best_checkpoint_metadata"] = {
                    "metric": "validation_mAP",
                    "metric_value": selection_state["best_metric"],
                    "aptiny": selection_state["best_aptiny"],
                    "epoch": selection_state["best_epoch"],
                }
            state["configuration_hash"] = configuration_hash
            atomic_torch_save(state, run_dir / "last.pth")
            temporary.unlink()

    class SingleBestStateHook(Hook):
        """Embed one mAP-selected weight state inside the rolling resume file."""

        def before_train(self, runner: Any) -> None:
            checkpoint = run_dir / "last.pth"
            if not args.resume or not checkpoint.is_file():
                return
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            restore_selection_state(state, selection_state)

        def after_val_epoch(
            self, runner: Any, metrics: dict[str, Any] | None = None
        ) -> None:
            values = metrics or {}
            metric, aptiny = validation_metric_pair(values)
            if metric is None:
                raise RuntimeError("validation completed without coco/bbox_mAP")
            if aptiny is None:
                raise RuntimeError("validation completed without aerial_coco/APtiny")
            if not math.isfinite(metric) or not math.isfinite(aptiny):
                raise RuntimeError(
                    f"validation produced non-finite metrics: mAP={metric}, APtiny={aptiny}"
                )
            if metric <= selection_state["best_metric"]:
                return
            checkpoint = run_dir / "last.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = runner.model.module if hasattr(runner.model, "module") else runner.model
            selection_state["best_model_state_dict"] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            state["best_model_state_dict"] = selection_state[
                "best_model_state_dict"
            ]
            selection_state["best_metric"] = metric
            selection_state["best_aptiny"] = aptiny
            selection_state["best_epoch"] = int(runner.epoch + 1)
            state["best_checkpoint_metadata"] = {
                "metric": "validation_mAP",
                "metric_value": selection_state["best_metric"],
                "aptiny": selection_state["best_aptiny"],
                "epoch": selection_state["best_epoch"],
            }
            state["configuration_hash"] = configuration_hash
            atomic_torch_save(state, checkpoint)

    runner.register_hook(AtomicRollingCheckpointHook(), priority="VERY_LOW")
    runner.register_hook(SingleBestStateHook(), priority="LOWEST")
    runner.train()
    elapsed = time.perf_counter() - started

    last_checkpoint = run_dir / "last.pth"
    if not last_checkpoint.is_file():
        raise FileNotFoundError(
            f"MMDetection completed without required rolling checkpoint {last_checkpoint}"
        )

    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameters_by_module = {
        name: int(sum(parameter.numel() for parameter in module.parameters()))
        for name, module in model.named_children()
    }

    rows = _read_scalar_rows(run_dir, getattr(runner, "log_dir", None))
    _write_history(run_dir, rows)
    validation_required = args.validation_interval > 0
    if validation_required:
        best_map_value = float(selection_state["best_metric"])
        best_tiny_value = float(selection_state["best_aptiny"])
        best_epoch = int(selection_state["best_epoch"])
        if (
            selection_state["best_model_state_dict"] is None
            or not math.isfinite(best_map_value)
            or not math.isfinite(best_tiny_value)
            or best_epoch <= 0
        ):
            raise RuntimeError(
                "validation was enabled but no valid mAP/APtiny checkpoint was selected"
            )
    else:
        best_map_value, best_tiny_value, best_epoch = best_metrics_from_rows(rows)

    checkpoint_identity = {
        "run_id": run_dir.name,
        "model_id": args.model_id,
        "seed": args.seed,
        "configuration_hash": configuration_hash,
        "epoch": best_epoch or args.epochs,
        "selection_metric": "validation_mAP",
        "selection_metric_value": float(best_map_value),
        "weight_variant": "raw",
    }
    checkpoint_best, checkpoint_sha256 = materialize_canonical_best(
        last_checkpoint,
        run_dir / "best.pth",
        checkpoint_identity,
    )

    summary = {
        "checkpoint_best": str(checkpoint_best),
        "checkpoint_resume": str(last_checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_selection_metric": "validation_mAP",
        "weight_variant": "raw",
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_load_verified": True,
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
    write_json(run_dir / "final_metrics.json", summary)
    safe_save_training_curves(
        rows,
        run_dir / "training_curves.png",
        warning_root=run_dir,
    )


if __name__ == "__main__":
    main()
