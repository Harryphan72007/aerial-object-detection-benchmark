#!/usr/bin/env python
"""Run or inspect automatic baseline and tuned final experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.hpo.final_workflow import FinalExperimentWorkflow
from src.paths import configured_drive_root
from src.workflows.contract import PRIMARY_MODELS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=PRIMARY_MODELS, required=True)
    parser.add_argument(
        "--dataset-track", choices=("2class", "10class"), default="2class"
    )
    parser.add_argument("--drive-root")
    parser.add_argument(
        "--config", default=str(REPOSITORY_ROOT / "project_config.yaml")
    )
    parser.add_argument("--start-finetuning", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workflow = FinalExperimentWorkflow(
        REPOSITORY_ROOT,
        configured_drive_root(args.config, args.drive_root),
        args.model_id,
        args.dataset_track,
    )
    print(
        json.dumps(
            workflow.run(
                start_expensive_stage=args.start_finetuning,
                batch_size=args.batch_size,
                accumulation=args.accumulation,
            ),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
