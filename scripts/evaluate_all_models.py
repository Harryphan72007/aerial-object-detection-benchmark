#!/usr/bin/env python
"""Evaluate compatible HPO final runs and aggregate three-seed results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.paths import configured_drive_root
from src.workflows.evaluation_runner import evaluate_pending_runs
from src.workflows.hpo_comparison import aggregate_hpo_results

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-track", choices=("2class", "10class"), default="2class"
    )
    parser.add_argument("--drive-root")
    parser.add_argument(
        "--config", default=str(REPOSITORY_ROOT / "project_config.yaml")
    )
    parser.add_argument("--evaluate-missing", action="store_true")
    parser.add_argument("--model-id", action="append", help="limit to these models")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--skip-profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drive_root = configured_drive_root(args.config, args.drive_root)
    if args.evaluate_missing:
        evaluate_pending_runs(
            REPOSITORY_ROOT,
            drive_root,
            args.dataset_track,
            model_ids=args.model_id,
            max_images=args.max_images,
            skip_profile=args.skip_profile,
        )
    print(
        json.dumps(
            aggregate_hpo_results(drive_root, args.dataset_track),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
