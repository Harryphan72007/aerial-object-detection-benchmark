#!/usr/bin/env python
"""Registry-driven common accuracy, slice, resolution, and efficiency evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.dataloaders import CocoDetectionRecords
from src.evaluation.calibration import detection_calibration
from src.evaluation.coco_evaluator import evaluate_coco
from src.evaluation.detection_metrics import confidence_curves, detailed_metrics
from src.evaluation.error_analysis import decompose_errors, evaluate_visdrone_slices
from src.models.registry import create_adapter
from src.paths import ProjectPaths
from src.drive_sync import validate_drive_writable
from src.training.checkpointing import RunRegistry
from src.utils.serialization import read_yaml, write_json


def require_successful_evaluation(failures: list[dict[str, Any]]) -> None:
    if failures:
        failed = ", ".join(
            f"{row['run_id']} ({row['exception_type']})" for row in failures
        )
        raise RuntimeError(f"Evaluation failed for selected run(s): {failed}")


def discover_evaluation_dataset(
    paths: ProjectPaths,
    dataset_track: str,
    split: str,
    *,
    image_root: str | Path | None = None,
    annotation_file: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve and instantiate the exact evaluator data view."""
    resolved_annotation = (
        Path(annotation_file)
        if annotation_file
        else paths.coco(dataset_track) / "annotations" / f"instances_{split}.json"
    )
    resolved_images = Path(image_root) if image_root else paths.images(split)
    records = CocoDetectionRecords(resolved_images, resolved_annotation)
    return {
        "annotation_file": resolved_annotation,
        "image_root": resolved_images,
        "record_count": len(records),
        "records": records,
    }


def _update_resize(node: Any, image_size: int) -> None:
    if isinstance(node, dict):
        if node.get("type") in {"Resize", "RandomResize"}:
            node["scale"] = (image_size, image_size)
            node["keep_ratio"] = True
        for value in node.values():
            _update_resize(value, image_size)
    elif isinstance(node, list):
        for value in node:
            _update_resize(value, image_size)


def _mmdet_config_for_resolution(
    source: Path, destination: Path, resolution: int
) -> Path:
    try:
        from mmengine.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "MMEngine is required to create resolution-specific inference configs."
        ) from exc
    cfg = Config.fromfile(str(source))
    for key in ("train_dataloader", "val_dataloader", "test_dataloader"):
        if key in cfg:
            _update_resize(cfg[key], resolution)
    try:
        cfg.model.test_cfg.rcnn.score_thr = 0.001
        cfg.model.test_cfg.rcnn.max_per_img = 500
    except (AttributeError, KeyError, TypeError) as error:
        raise RuntimeError(
            "MMDetection final evaluation requires RCNN score_thr and "
            "max_per_img controls"
        ) from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(destination))
    return destination


def _subset_annotation(
    annotation_file: Path,
    image_ids: set[int],
    destination: Path,
) -> Path:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    subset = dict(data)
    subset["images"] = [
        image for image in data["images"] if int(image["id"]) in image_ids
    ]
    subset["annotations"] = [
        annotation
        for annotation in data["annotations"]
        if int(annotation["image_id"]) in image_ids
    ]
    write_json(destination, subset, atomic=False)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    parser.add_argument(
        "--dataset-track", choices=["2class", "10class"], default="2class"
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-root")
    parser.add_argument("--annotation-file")
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--best-per-model", action="store_true")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--device")
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--resolutions", nargs="*", type=int, default=[640, 1024, 1280]
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--skip-profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_drive_writable(args.drive_root)
    paths = ProjectPaths.from_value(args.drive_root).create()
    registry = RunRegistry(paths)
    runs = registry.list_available_runs(dataset_track=args.dataset_track)
    if args.run_id:
        runs = [run for run in runs if run["run_id"] in set(args.run_id)]
    if args.models:
        runs = [run for run in runs if run["model_id"] in set(args.models)]
    if args.best_per_model:
        selected = []
        for model_id in sorted({run["model_id"] for run in runs}):
            selected.append(
                max(
                    [run for run in runs if run["model_id"] == model_id],
                    key=lambda run: float(run.get("best_validation_map", 0)),
                )
            )
        runs = selected
    if not runs:
        raise RuntimeError("No completed compatible runs found in the registry.")
    mappings = {tuple(run["class_names"]) for run in runs}
    if len(mappings) != 1:
        raise RuntimeError("Selected runs have incompatible class mappings.")

    dataset = discover_evaluation_dataset(
        paths,
        args.dataset_track,
        args.split,
        image_root=args.image_root,
        annotation_file=args.annotation_file,
    )
    full_annotation = dataset["annotation_file"]
    records = dataset["records"]
    count = (
        len(records)
        if args.max_images is None
        else min(len(records), args.max_images)
    )
    selected_records = [records[index] for index in range(count)]
    if count < len(records):
        evaluation_annotation = _subset_annotation(
            full_annotation,
            {record["image_id"] for record in selected_records},
            paths.cache
            / f"instances_{args.dataset_track}_{args.split}_first_{count}.json",
        )
    else:
        evaluation_annotation = full_annotation

    results: list[dict[str, Any]] = []
    failed_models: list[dict[str, Any]] = []
    for run in runs:
        run_dir = Path(
            run.get("run_dir") or paths.run_dir(run["model_id"], run["run_id"])
        )
        base_model_config = read_yaml(run_dir / "model_config.yaml")
        resolutions = sorted(
            set([int(run["input_resolution"]), *args.resolutions])
        )
        checkpoint = registry.load_checkpoint_from_registry(
            run["run_id"], "best_map"
        )
        for resolution in resolutions:
            model_config = dict(base_model_config)
            model_config["confidence_threshold"] = 0.001
            model_config["max_detections"] = 500
            model_config["input_resolution"] = resolution
            if run["framework"] in {"mmdetection", "vmamba_mmdetection"}:
                model_config["resolved_framework_config"] = str(
                    _mmdet_config_for_resolution(
                        run_dir / "runtime_config.py",
                        paths.cache
                        / f"{run['run_id']}__inference_{resolution}.py",
                        resolution,
                    )
                )
            try:
                adapter = create_adapter(run["model_id"], args.device)
                adapter.load_model(checkpoint, model_config)
            except Exception as error:
                failure = {
                    "model_id": run["model_id"],
                    "run_id": run["run_id"],
                    "exception_type": type(error).__name__,
                    "message": str(error),
                }
                failed_models.append(failure)
                write_json(paths.evaluation / "evaluation_failures.json", failed_models)
                print(json.dumps({"evaluation_failure": failure}, indent=2))
                continue
            coco_predictions: list[dict[str, Any]] = []
            for item in selected_records:
                prediction = adapter.predict([item["image"]])[0]
                prediction["image_id"] = item["image_id"]
                coco_predictions.extend(
                    adapter.export_predictions_to_coco([prediction])
                )
            prediction_path = (
                paths.predictions
                / f"{run['run_id']}__{args.split}__res{resolution}.json"
            )
            write_json(prediction_path, coco_predictions, atomic=False)
            metrics = evaluate_coco(evaluation_annotation, prediction_path)
            metrics.update(
                detailed_metrics(evaluation_annotation, prediction_path)
            )
            metrics["calibration"] = detection_calibration(
                evaluation_annotation, prediction_path
            )
            metrics["confidence_curves"] = confidence_curves(
                evaluation_annotation, prediction_path
            )
            metrics.update(
                decompose_errors(evaluation_annotation, prediction_path)
            )
            metrics["visdrone_slices"] = evaluate_visdrone_slices(
                evaluation_annotation, prediction_path
            )
            if not args.skip_profile and selected_records:
                try:
                    profile = adapter.profile(
                        [selected_records[0]["image"]],
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                    metrics.update(profile)
                    metrics["profile_status"] = "completed"
                except Exception as error:
                    metrics["profile_status"] = "failed"
                    metrics["profile_error"] = repr(error)
            metrics.update(
                {
                    "run_id": run["run_id"],
                    "model_id": run["model_id"],
                    "architecture_family": run["architecture_family"],
                    "dataset_track": args.dataset_track,
                    "training_resolution": run["input_resolution"],
                    "evaluation_resolution": resolution,
                    "seed": run["seed"],
                    "evaluation_image_count": count,
                    "prediction_file": str(prediction_path),
                    "evaluation_hardware": run.get("gpu_name"),
                    "total_parameters": run.get("total_parameters"),
                    "trainable_parameters": run.get("trainable_parameters"),
                    "total_training_seconds": run.get(
                        "total_training_seconds"
                    ),
                }
            )
            output = (
                paths.evaluation
                / f"{run['run_id']}__res{resolution}__metrics.json"
            )
            write_json(output, metrics)
            results.append(metrics)
            print(
                json.dumps(
                    {
                        "model_id": run["model_id"],
                        "resolution": resolution,
                        "mAP": metrics["mAP"],
                        "APtiny": metrics["APtiny"],
                        "latency_ms": metrics.get("mean_latency_ms"),
                    },
                    indent=2,
                )
            )
    write_json(
        paths.evaluation
        / f"comparison_{args.dataset_track}_{args.split}.json",
        results,
    )
    write_json(paths.evaluation / "evaluation_failures.json", failed_models)
    require_successful_evaluation(failed_models)


if __name__ == "__main__":
    main()
