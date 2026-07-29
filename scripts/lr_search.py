#!/usr/bin/env python
"""Run or preview the deterministic learning-rate-only search."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.paths import configured_drive_root
from src.training.lr_search import (
    PROMOTION_RUNGS,
    export_candidate_yaml,
    generate_lr_candidates,
)
from src.training.lr_workflow import LRControlledBenchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", default=str(REPOSITORY_ROOT)
    )
    parser.add_argument("--drive-root")
    parser.add_argument("--config", default="project_config.yaml")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument(
        "--run-lr-range-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--run-boundary-extension", action="store_true")
    parser.add_argument("--start-expensive-stage", action="store_true")
    parser.add_argument("--allow-over-budget-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drive_root = configured_drive_root(args.config, args.drive_root)
    workflow = LRControlledBenchmark(args.repo_root, drive_root)
    split_summary = workflow.prepare_manifests()
    baseline = workflow.resolve_baseline(args.model_id)
    candidates = generate_lr_candidates(baseline.learning_rate)
    candidate_yaml = (
        workflow.repository_config_dir
        / f"{args.model_id}_2class_candidates.yaml"
    )
    export_candidate_yaml(
        candidate_yaml,
        model_id=args.model_id,
        baseline=baseline,
        candidates=candidates,
    )
    preview = {
        "experiment": "VisDrone Learning-Rate-Controlled Architecture Benchmark",
        "model_id": args.model_id,
        "baseline_learning_rate": baseline.learning_rate,
        "baseline_config_path": baseline.baseline_config_path,
        "default_candidates": candidates,
        "rungs": list(PROMOTION_RUNGS),
        "search_train_images": split_summary["statistics"][
            "search_train_seed42.json"
        ]["images"],
        "search_validation_images": split_summary["statistics"][
            "search_validation_seed42.json"
        ]["images"],
        "effective_batch_size": args.batch_size * args.accumulation,
        "candidate_config": str(candidate_yaml),
        "expensive_stage_started": args.start_expensive_stage,
    }
    print(json.dumps(preview, indent=2))
    if not args.start_expensive_stage:
        print(
            "Preview complete. Set START_EXPENSIVE_STAGE=True (or pass "
            "--start-expensive-stage) to calibrate runtime and run the search."
        )
        return
    result = workflow.run_search(
        args.model_id,
        batch_size=args.batch_size,
        accumulation=args.accumulation,
        run_lr_range_test=args.run_lr_range_test,
        run_boundary_extension=args.run_boundary_extension,
        allow_over_budget_run=args.allow_over_budget_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
