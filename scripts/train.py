#!/usr/bin/env python
"""Start or resume one standardized training run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.trainer import TrainingOrchestrator
from src.paths import configured_drive_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument("--drive-root")
    parser.add_argument("--config", default="project_config.yaml")
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--dataset-track", choices=["2class", "10class"], default="2class"
    )
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume-run-id")
    parser.add_argument(
        "--overrides-json",
        default="{}",
        help="JSON object of backend-supported hyperparameter overrides.",
    )
    args = parser.parse_args()
    overrides = json.loads(args.overrides_json)
    if not isinstance(overrides, dict):
        raise ValueError("--overrides-json must decode to an object")
    drive_root = configured_drive_root(args.config, args.drive_root)
    manifest = TrainingOrchestrator(args.repo_root, drive_root).run(
        args.model_id,
        args.dataset_track,
        args.image_size,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.epochs,
        args.seed,
        not args.no_amp,
        args.resume_run_id,
        overrides,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
