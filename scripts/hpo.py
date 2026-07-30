#!/usr/bin/env python
"""Run or inspect the persistent two-stage random HPO protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.hpo.workflow import TwoStageRandomHPO
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
    parser.add_argument("--start-hpo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workflow = TwoStageRandomHPO(
        REPOSITORY_ROOT,
        configured_drive_root(args.config, args.drive_root),
        args.model_id,
        args.dataset_track,
    )
    print(
        json.dumps(
            workflow.run(start_expensive_stage=args.start_hpo),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
