#!/usr/bin/env python
"""One user-facing CLI for the notebook-first benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.benchmark_status import (
    discover_all_statuses,
    discover_model_status,
    format_status_table,
)
from src.paths import configured_drive_root
from src.workflows.comparison import compare_completed_models
from src.workflows.contract import PRIMARY_MODELS
from src.workflows.model_day import ModelDayOptions, inspect_model_day, run_model_day
from src.workflows.publishing import publish_results

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--drive-root")
    parser.add_argument("--config", default=str(REPOSITORY_ROOT / "project_config.yaml"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="Show persistent progress")
    _shared(status)
    status.add_argument("--model-id", choices=PRIMARY_MODELS)
    status.add_argument("--json", action="store_true")

    next_step = commands.add_parser("next", help="Show the next incomplete stage")
    _shared(next_step)
    next_step.add_argument("--model-id", choices=PRIMARY_MODELS, required=True)

    run = commands.add_parser("run-model-day", help="Run/resume one selected model")
    _shared(run)
    run.add_argument("--model-id", choices=PRIMARY_MODELS, required=True)
    run.add_argument("--run-mode", default="auto")
    run.add_argument(
        "--run-lr-range-test", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument("--run-boundary-extension", action="store_true")
    run.add_argument("--start-expensive-stage", action="store_true")
    run.add_argument("--allow-over-budget-run", action="store_true")
    run.add_argument("--smoke-test", action="store_true")
    run.add_argument(
        "--data-access-mode",
        choices=["local_cache", "drive_direct"],
        default="drive_direct",
    )
    run.add_argument("--local-cache-root", default="/content/visdrone_cache")

    publish = commands.add_parser("publish", help="Validate/preview or publish results")
    _shared(publish)
    publish.add_argument("--model-id", choices=PRIMARY_MODELS, required=True)
    publish.add_argument(
        "--dataset-track", choices=("2class", "10class"), default="2class"
    )
    publish.add_argument("--publish-results", action="store_true")
    publish.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)

    compare = commands.add_parser("compare", help="Compare compatible completed models")
    _shared(compare)
    compare.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drive_root = configured_drive_root(args.config, args.drive_root)
    if args.command == "status":
        rows = (
            [discover_model_status(drive_root, args.model_id, REPOSITORY_ROOT)]
            if args.model_id
            else discover_all_statuses(drive_root, repo_root=REPOSITORY_ROOT)
        )
        _json(rows) if args.json else print(format_status_table(rows))
        return
    if args.command == "next":
        _json(inspect_model_day(drive_root, args.model_id, REPOSITORY_ROOT))
        return
    if args.command == "run-model-day":
        _json(
            run_model_day(
                REPOSITORY_ROOT,
                drive_root,
                ModelDayOptions(
                    model_id=args.model_id,
                    run_mode=args.run_mode,
                    run_lr_range_test=args.run_lr_range_test,
                    run_boundary_extension=args.run_boundary_extension,
                    start_expensive_stage=args.start_expensive_stage,
                    allow_over_budget_run=args.allow_over_budget_run,
                    smoke_test=args.smoke_test,
                    data_access_mode=args.data_access_mode,
                    local_cache_root=args.local_cache_root,
                ),
            )
        )
        return
    if args.command == "publish":
        _json(
            publish_results(
                REPOSITORY_ROOT,
                drive_root,
                args.model_id,
                dataset_track=args.dataset_track,
                publish_results=args.publish_results,
                dry_run=args.dry_run,
            )
        )
        return
    _json(compare_completed_models(drive_root, args.output_dir))


if __name__ == "__main__":
    main()
