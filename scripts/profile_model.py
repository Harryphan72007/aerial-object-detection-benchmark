#!/usr/bin/env python
"""Profile registered models at batch 1/4/8 under one runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from src.models.registry import create_adapter
from src.paths import ProjectPaths
from src.training.checkpointing import RunRegistry
from src.utils.serialization import read_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--device")
    args = parser.parse_args()

    paths = ProjectPaths.from_value(args.drive_root).create()
    registry = RunRegistry(paths)
    run = next(
        (
            item
            for item in registry.list_available_runs(status=None)
            if item["run_id"] == args.run_id
        ),
        None,
    )
    if not run:
        raise KeyError(args.run_id)
    run_dir = paths.run_dir(run["model_id"], run["run_id"])
    config = read_yaml(run_dir / "model_config.yaml")
    config["input_resolution"] = run["input_resolution"]
    if run["framework"] in {"mmdetection", "vmamba_mmdetection"}:
        config["resolved_framework_config"] = str(run_dir / "runtime_config.py")
    image_path = (
        Path(args.image)
        if args.image
        else next((paths.coco(run["dataset_track"]) / "val").glob("*"))
    )
    image = Image.open(image_path).convert("RGB")
    adapter = create_adapter(run["model_id"], args.device)
    adapter.load_model(
        registry.load_checkpoint_from_registry(args.run_id), config
    )
    profiles = []
    for batch_size in args.batch_sizes:
        try:
            metrics = adapter.profile(
                [image.copy() for _ in range(batch_size)],
                warmup=args.warmup,
                iterations=args.iterations,
            )
            metrics.update(
                {
                    "batch_size": batch_size,
                    "status": "completed",
                    "timing_scope": "adapter end-to-end inference",
                }
            )
        except Exception as error:
            metrics = {
                "batch_size": batch_size,
                "status": "failed",
                "error": repr(error),
            }
        profiles.append(metrics)
    output = {
        "run_id": args.run_id,
        "model_id": run["model_id"],
        "image": str(image_path),
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "profiles": profiles,
    }
    write_json(paths.evaluation / f"{args.run_id}__profile.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
