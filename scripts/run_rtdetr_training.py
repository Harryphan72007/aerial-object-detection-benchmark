#!/usr/bin/env python
"""Native RT-DETRv2 fine-tuning loop with standardized checkpoints."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.training.callbacks import (
    BestMetricState,
    EpochHistoryWriter,
    save_training_curves,
)
from src.runtime_manifest import write_runtime_environment_manifest
from src.training.checkpointing import atomic_torch_save, materialize_checkpoint_alias
from src.training.recipes import (
    RTDETR_BASELINE_LR,
    RTDETR_GRADIENT_CLIP,
    RTDETR_MAX_DETECTIONS,
    RTDETR_WARMUP_EPOCHS,
    RTDETR_WEIGHT_DECAY,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--train-ann", required=True)
    parser.add_argument("--val-ann", required=True)
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--scheduler-horizon", type=int, required=True)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--lr-range-test-steps", type=int, default=0)
    parser.add_argument("--lr-range-output")
    parser.add_argument("--lr-range-start-multiplier", type=float, default=0.01)
    parser.add_argument("--lr-range-end-multiplier", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--accumulation", type=int, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=RTDETR_BASELINE_LR)
    parser.add_argument("--weight-decay", type=float, default=RTDETR_WEIGHT_DECAY)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overrides", default="{}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import psutil
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        RTDetrImageProcessor,
        RTDetrV2Config,
        RTDetrV2ForObjectDetection,
    )

    from src.evaluation.coco_evaluator import evaluate_coco
    from src.data.dataset import CocoDetectionDataset
    from src.reproducibility import (
        capture_rng_state,
        restore_rng_state,
        seed_everything,
        worker_seed,
    )

    seed_everything(args.seed)
    if args.scheduler_horizon < args.epochs:
        raise ValueError("--scheduler-horizon must be >= --epochs")
    if args.validation_interval < 0:
        raise ValueError("--validation-interval must be non-negative")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_runtime_environment_manifest(run_dir, args.model_id, Path.cwd())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data = json.loads(Path(args.train_ann).read_text(encoding="utf-8"))
    classes = [category["name"] for category in train_data["categories"]]
    id2label = {index: name for index, name in enumerate(classes)}
    label2id = {name: index for index, name in id2label.items()}
    overrides = json.loads(args.overrides)
    processor = RTDetrImageProcessor.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        size={"height": args.image_size, "width": args.image_size},
    )
    model_configuration = RTDetrV2Config.from_pretrained(
        args.model_name, revision=args.model_revision
    )
    supported_configuration_keys = {
        "num_queries",
        "decoder_layers",
        "num_denoising",
        "matcher_class_cost",
        "matcher_bbox_cost",
        "matcher_giou_cost",
        "weight_loss_vfl",
        "weight_loss_bbox",
        "weight_loss_giou",
        "dropout",
        "attention_dropout",
    }
    applied_overrides: dict[str, Any] = {}
    unsupported_overrides: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in supported_configuration_keys and hasattr(model_configuration, key):
            setattr(model_configuration, key, value)
            applied_overrides[key] = value
        elif key in {
            "learning_rate",
            "weight_decay",
            "gradient_clip",
            "warmup_epochs",
            "max_detections",
        }:
            applied_overrides[key] = value
        else:
            unsupported_overrides[key] = value
    model_configuration.id2label = id2label
    model_configuration.label2id = label2id
    model = RTDetrV2ForObjectDetection.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        config=model_configuration,
        ignore_mismatched_sizes=True,
    ).to(device)
    override_report = {
        "applied": applied_overrides,
        "unsupported": unsupported_overrides,
    }
    (run_dir / "applied_overrides.json").write_text(
        json.dumps(override_report, indent=2), encoding="utf-8"
    )

    def encode_record(record: dict[str, Any]) -> dict[str, Any]:
        annotations = [
            {**annotation, "category_id": int(annotation["category_id"]) - 1}
            for annotation in record["annotations"]
        ]
        encoded = processor(
            images=record["image"],
            annotations={"image_id": record["image_id"], "annotations": annotations},
            return_tensors="pt",
        )
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"][0],
            "image_id": record["image_id"],
            "image": record["image"],
        }

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        # The processor resizes every image to the configured square resolution.
        return {
            "pixel_values": torch.stack(
                [item["pixel_values"] for item in batch]
            ),
            "labels": [item["labels"] for item in batch],
        }

    train_records = CocoDetectionDataset(
        args.train_images, args.train_ann, transform=encode_record
    )
    validation_records = CocoDetectionDataset(
        args.val_images, args.val_ann, transform=encode_record
    )
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_records,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        generator=sampler_generator,
        worker_init_fn=worker_seed,
    )
    learning_rate = float(overrides.get("learning_rate", args.learning_rate))
    weight_decay = float(overrides.get("weight_decay", args.weight_decay))
    gradient_clip = float(overrides.get("gradient_clip", RTDETR_GRADIENT_CLIP))
    warmup_epochs = int(overrides.get("warmup_epochs", RTDETR_WARMUP_EPOCHS))
    maximum_detections = int(
        overrides.get("max_detections", RTDETR_MAX_DETECTIONS)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.scheduler_horizon - warmup_epochs)
    )
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine_scheduler
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and torch.cuda.is_available()
    )
    if args.lr_range_test_steps:
        if args.resume:
            raise ValueError("the LR range test must start from pretrained weights")
        from src.training.lr_range import (
            exponential_lr_schedule,
            save_lr_range_artifacts,
            should_stop_range_test,
        )

        schedule = exponential_lr_schedule(
            learning_rate,
            args.lr_range_test_steps,
            args.lr_range_start_multiplier,
            args.lr_range_end_multiplier,
        )
        range_history: list[dict[str, Any]] = []
        stopped_reason: str | None = None
        loader_iterator = iter(train_loader)
        model.train()
        for optimizer_step, step_learning_rate in enumerate(schedule, start=1):
            for group in optimizer.param_groups:
                group["lr"] = step_learning_rate
            optimizer.zero_grad(set_to_none=True)
            step_losses: list[float] = []
            for _ in range(args.accumulation):
                try:
                    batch = next(loader_iterator)
                except StopIteration:
                    loader_iterator = iter(train_loader)
                    batch = next(loader_iterator)
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                labels = [
                    {
                        key: value.to(device) if hasattr(value, "to") else value
                        for key, value in label.items()
                    }
                    for label in batch["labels"]
                ]
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=args.amp and torch.cuda.is_available(),
                ):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    raw_loss = outputs.loss
                    scaled_loss = raw_loss / args.accumulation
                step_losses.append(float(raw_loss.detach().item()))
                scaler.scale(scaled_loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            )
            scaler.step(optimizer)
            scaler.update()
            mean_loss = float(np.mean(step_losses))
            range_history.append(
                {
                    "optimizer_step": optimizer_step,
                    "learning_rate": step_learning_rate,
                    "raw_loss": mean_loss,
                    "gradient_norm": gradient_norm,
                }
            )
            stop, stopped_reason = should_stop_range_test(
                [row["raw_loss"] for row in range_history]
            )
            if stop:
                break
        output = Path(args.lr_range_output or (run_dir / "lr_range_test"))
        summary = save_lr_range_artifacts(
            output,
            range_history,
            baseline_learning_rate=learning_rate,
            stopped_reason=stopped_reason,
        )
        print(json.dumps(summary, indent=2))
        return
    history = EpochHistoryWriter(run_dir)
    best = BestMetricState()
    start_epoch = 1
    if args.resume and (run_dir / "last.pth").exists():
        state = torch.load(
            run_dir / "last.pth", map_location="cpu", weights_only=False
        )
        model_state = state["model_state_dict"] if "model_state_dict" in state else state["model"]
        optimizer_state = state["optimizer_state_dict"] if "optimizer_state_dict" in state else state["optimizer"]
        scheduler_state = state["scheduler_state_dict"] if "scheduler_state_dict" in state else state["scheduler"]
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        scaler_state = state.get("scaler_state_dict", state.get("scaler"))
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        start_epoch = int(state["epoch"]) + 1
        best.best_map = float(state.get("best_map", float("-inf")))
        best.best_aptiny = float(state.get("best_aptiny", float("-inf")))
        best.best_map_epoch = int(state.get("best_map_epoch", 0))
        best.best_aptiny_epoch = int(state.get("best_aptiny_epoch", 0))
        checkpoint_horizon = int(state.get("scheduler_horizon", args.scheduler_horizon))
        if checkpoint_horizon != args.scheduler_horizon:
            raise ValueError(
                "scheduler horizon changed across resume: "
                f"checkpoint={checkpoint_horizon}, requested={args.scheduler_horizon}"
            )
        if state.get("rng_state"):
            restore_rng_state(state["rng_state"])
        if state.get("sampler_generator_state") is not None:
            sampler_generator.set_state(state["sampler_generator_state"])

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.perf_counter()
        data_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        optimizer_time = 0.0
        losses: list[float] = []
        component_values: dict[str, list[float]] = defaultdict(list)
        gradient_norm = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        last_batch_end = time.perf_counter()
        for step, batch in enumerate(train_loader, 1):
            data_time += time.perf_counter() - last_batch_end
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = [
                {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in label.items()
                }
                for label in batch["labels"]
            ]
            forward_start = time.perf_counter()
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.amp and torch.cuda.is_available(),
            ):
                outputs = model(pixel_values=pixel_values, labels=labels)
                full_loss = outputs.loss
                loss = full_loss / args.accumulation
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_time += time.perf_counter() - forward_start
            losses.append(float(full_loss.detach().item()))
            for key, value in (outputs.loss_dict or {}).items():
                component_values[str(key)].append(float(value.detach().item()))

            backward_start = time.perf_counter()
            scaler.scale(loss).backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_time += time.perf_counter() - backward_start
            if step % args.accumulation == 0 or step == len(train_loader):
                optimizer_start = time.perf_counter()
                scaler.unscale_(optimizer)
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip
                    )
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                optimizer_time += time.perf_counter() - optimizer_start
            last_batch_end = time.perf_counter()
        scheduler.step()

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            # Legacy aliases keep older local checkpoints resumable.
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "model_name": args.model_name,
            "id2label": id2label,
            "config": vars(args),
            "run_id": run_dir.name,
            "seed": args.seed,
            "best_map": best.best_map,
            "best_aptiny": best.best_aptiny,
            "best_map_epoch": best.best_map_epoch,
            "best_aptiny_epoch": best.best_aptiny_epoch,
            "scheduler_horizon": args.scheduler_horizon,
            "rng_state": capture_rng_state(),
            "sampler_generator_state": sampler_generator.get_state(),
        }
        atomic_torch_save(checkpoint_state, run_dir / "last.pth")

        should_validate = (
            args.validation_interval > 0
            and epoch % args.validation_interval == 0
        )
        validation_seconds = 0.0
        metrics: dict[str, float] = {}
        flags = {"best_map": False, "best_aptiny": False}
        if should_validate:
            validation_start = time.perf_counter()
            model.eval()
            predictions: list[dict[str, Any]] = []
            with torch.inference_mode():
                for index in range(len(validation_records)):
                    item = validation_records[index]
                    encoded = processor(images=item["image"], return_tensors="pt")
                    encoded = {
                        key: value.to(device) for key, value in encoded.items()
                    }
                    outputs = model(**encoded)
                    target_sizes = torch.tensor(
                        [(item["image"].height, item["image"].width)],
                        device=device,
                    )
                    result = processor.post_process_object_detection(
                        outputs, target_sizes=target_sizes, threshold=0.001
                    )[0]
                    if len(result["scores"]) > maximum_detections:
                        keep = result["scores"].argsort(descending=True)[
                            :maximum_detections
                        ]
                        result = {key: value[keep] for key, value in result.items()}
                    for box, score, label in zip(
                        result["boxes"].cpu(),
                        result["scores"].cpu(),
                        result["labels"].cpu(),
                    ):
                        x1, y1, x2, y2 = box.tolist()
                        predictions.append(
                            {
                                "image_id": item["image_id"],
                                "category_id": int(label) + 1,
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "score": float(score),
                            }
                        )
            validation_seconds = time.perf_counter() - validation_start
            prediction_path = run_dir / f"predictions_epoch_{epoch:03d}.json"
            prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
            metrics = evaluate_coco(args.val_ann, prediction_path)
            flags = best.update(epoch, metrics)
        # Save the updated best-state metadata in the canonical last checkpoint.
        checkpoint_state.update(
            {
                "best_map": best.best_map,
                "best_aptiny": best.best_aptiny,
                "best_map_epoch": best.best_map_epoch,
                "best_aptiny_epoch": best.best_aptiny_epoch,
                "rng_state": capture_rng_state(),
                "sampler_generator_state": sampler_generator.get_state(),
            }
        )
        atomic_torch_save(checkpoint_state, run_dir / "last.pth")
        if flags["best_map"]:
            materialize_checkpoint_alias(run_dir / "last.pth", run_dir / "best_map.pth")
        if flags["best_aptiny"]:
            materialize_checkpoint_alias(run_dir / "last.pth", run_dir / "best_aptiny.pth")

        epoch_seconds = time.perf_counter() - epoch_start
        row: dict[str, Any] = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm": gradient_norm,
            "epoch_seconds": epoch_seconds,
            "data_loading_seconds": data_time,
            "forward_seconds": forward_time,
            "backward_seconds": backward_time,
            "optimizer_step_seconds": optimizer_time,
            "validation_seconds": validation_seconds,
            "images_per_second_training": len(train_records)
            / max(epoch_seconds - validation_seconds, 1e-12),
            "batches_per_second_training": len(train_loader)
            / max(epoch_seconds - validation_seconds, 1e-12),
            "peak_allocated_gpu_memory": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else 0,
            "peak_reserved_gpu_memory": torch.cuda.max_memory_reserved()
            if torch.cuda.is_available()
            else 0,
            "cpu_ram_bytes": psutil.Process().memory_info().rss,
            **{
                f"loss_{key}": float(np.mean(values))
                for key, values in component_values.items()
            },
            **{
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            },
        }
        history.append(row)
        save_training_curves(history.rows, run_dir / "training_curves.png")
        print(json.dumps(row))

    if not (run_dir / "best_map.pth").exists():
        materialize_checkpoint_alias(run_dir / "last.pth", run_dir / "best_map.pth")
    if not (run_dir / "best_aptiny.pth").exists():
        materialize_checkpoint_alias(
            run_dir / "last.pth", run_dir / "best_aptiny.pth"
        )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    cumulative = 0.0
    time_to_best_map = 0.0
    time_to_best_tiny = 0.0
    for row in history.rows:
        cumulative += float(row.get("epoch_seconds", 0.0))
        if int(row.get("epoch", 0)) == best.best_map_epoch:
            time_to_best_map = cumulative
        if int(row.get("epoch", 0)) == best.best_aptiny_epoch:
            time_to_best_tiny = cumulative
    summary = {
        "checkpoint_best_map": str(run_dir / "best_map.pth"),
        "checkpoint_best_aptiny": str(run_dir / "best_aptiny.pth"),
        "checkpoint_last": str(run_dir / "last.pth"),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "estimated_model_size_bytes_fp32": total * 4,
        "best_validation_map": best.best_map if np.isfinite(best.best_map) else 0.0,
        "best_validation_aptiny": (
            best.best_aptiny if np.isfinite(best.best_aptiny) else 0.0
        ),
        "best_epoch": best.best_map_epoch,
        "best_aptiny_epoch": best.best_aptiny_epoch,
        "time_to_best_map_seconds": time_to_best_map,
        "time_to_best_aptiny_seconds": time_to_best_tiny,
        "framework_training_seconds": time.perf_counter() - started,
        "pretrained_model_name_or_path": args.model_name,
        "pretrained_revision": args.model_revision,
        "num_queries": int(model.config.num_queries),
        "hyperparameter_overrides": override_report,
    }
    (run_dir / "final_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
