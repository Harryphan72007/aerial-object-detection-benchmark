#!/usr/bin/env python
"""Evaluate compatible HPO final runs and aggregate three-seed results."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.hpo.final_workflow import IMAGE_SIZE
from src.paths import configured_drive_root
from src.subprocess_utils import python_module_command
from src.workflows.environment import ensure_model_environment
from src.workflows.hpo_comparison import (
    aggregate_hpo_results,
    compatible_final_runs,
)

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
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--skip-profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drive_root = configured_drive_root(args.config, args.drive_root)
    if args.evaluate_missing:
        for run in compatible_final_runs(drive_root, args.dataset_track):
            output = (
                Path(drive_root)
                / "evaluation"
                / f"{run['run_id']}__res{IMAGE_SIZE}__metrics.json"
            )
            if output.is_file():
                continue
            ensure_model_environment(
                str(run["model_id"]), REPOSITORY_ROOT, drive_root
            )
            command = python_module_command(
                "scripts.evaluate",
                "--drive-root",
                str(drive_root),
                "--dataset-track",
                args.dataset_track,
                "--split",
                "val",
                "--run-id",
                str(run["run_id"]),
                "--resolutions",
                str(IMAGE_SIZE),
            )
            if args.max_images:
                command.extend(["--max-images", str(args.max_images)])
            if args.skip_profile:
                command.append("--skip-profile")
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    print(
        json.dumps(
            aggregate_hpo_results(drive_root, args.dataset_track),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
