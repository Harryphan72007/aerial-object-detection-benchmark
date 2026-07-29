#!/usr/bin/env python
"""Restart from pretrained weights and fine-tune on complete official train."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.paths import configured_drive_root
from src.training.lr_workflow import LRControlledBenchmark
from src.utils.serialization import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", default=str(REPOSITORY_ROOT)
    )
    parser.add_argument("--drive-root")
    parser.add_argument("--config", default="project_config.yaml")
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--selected-config",
        help="Optional; automatically discovered from Drive or configs/lr_search.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--start-expensive-stage", action="store_true")
    parser.add_argument("--allow-over-budget-run", action="store_true")
    parser.add_argument("--skip-common-evaluation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    from src.benchmark_status import find_selected_config

    selected_path = (
        Path(args.selected_config)
        if args.selected_config
        else find_selected_config(
            configured_drive_root(args.config, args.drive_root),
            args.model_id,
            repo_root,
        )
    )
    if selected_path is None:
        raise FileNotFoundError(
            f"No selected LR config found for {args.model_id}. Run notebook 12."
        )
    if not selected_path.is_absolute():
        selected_path = repo_root / selected_path
    selected = read_yaml(selected_path)
    preview = {
        "model_id": args.model_id,
        "selected_config": str(selected_path),
        "selected_learning_rate": selected.get("search", {}).get(
            "selected_learning_rate"
        ),
        "restart_from_pretrained": selected.get("final_training", {}).get(
            "restart_from_pretrained"
        ),
        "dataset": selected.get("final_training", {}).get("dataset"),
        "epochs": selected.get("final_training", {}).get("epochs"),
        "expensive_stage_started": args.start_expensive_stage,
    }
    print(json.dumps(preview, indent=2))
    if not args.start_expensive_stage:
        print(
            "Preview complete. Pass --start-expensive-stage only after verifying "
            "the selected configuration and full-train identity."
        )
        return
    drive_root = configured_drive_root(args.config, args.drive_root)
    workflow = LRControlledBenchmark(repo_root, drive_root)
    manifest = workflow.run_final_training(
        args.model_id,
        selected_path,
        batch_size=args.batch_size,
        accumulation=args.accumulation,
        allow_over_budget_run=args.allow_over_budget_run,
        run_common_evaluation=not args.skip_common_evaluation,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
