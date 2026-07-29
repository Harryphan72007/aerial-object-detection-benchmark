#!/usr/bin/env python
"""Resumable Optuna search with epoch-level pruning through run resumption."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.hyperparameter_search import (
    create_study,
    suggest_common,
    suggest_faster_rcnn,
    suggest_hierarchical_backbone,
    suggest_rtdetr,
)
from src.training.trainer import TrainingOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-track", default="2class")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs-per-trial", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repo-root", default=str(Path(__file__).resolve().parents[1])
    )
    args = parser.parse_args()
    study = create_study(
        Path(args.drive_root) / "cache" / "optuna",
        f"{args.model_id}_{args.dataset_track}",
    )
    orchestrator = TrainingOrchestrator(args.repo_root, args.drive_root)

    def objective(trial):
        import optuna

        parameters = suggest_common(trial)
        if args.model_id.startswith("faster_rcnn"):
            parameters.update(suggest_faster_rcnn(trial))
        if args.model_id in {
            "faster_rcnn_swin_t",
            "faster_rcnn_vmamba_t",
        }:
            parameters.update(suggest_hierarchical_backbone(trial))
        if args.model_id == "rtdetrv2_l":
            parameters.update(suggest_rtdetr(trial))
        resolution = int(parameters.pop("input_resolution"))
        accumulation = int(parameters.pop("gradient_accumulation_steps"))
        run_id = None
        manifest = None
        for target_epoch in range(1, args.epochs_per_trial + 1):
            manifest = orchestrator.run(
                args.model_id,
                args.dataset_track,
                resolution,
                2,
                accumulation,
                target_epoch,
                args.seed,
                True,
                run_id,
                parameters,
            )
            run_id = manifest["run_id"]
            value = float(manifest.get("best_validation_aptiny", 0.0))
            trial.report(value, step=target_epoch)
            trial.set_user_attr("run_id", run_id)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"pruned at epoch {target_epoch} with APtiny={value:.6f}"
                )
        assert manifest is not None
        return float(manifest["best_validation_aptiny"])

    study.optimize(objective, n_trials=args.trials)
    trial_frame = study.trials_dataframe()
    output_csv = (
        Path(args.drive_root)
        / "cache"
        / "optuna"
        / f"{args.model_id}_{args.dataset_track}_trials.csv"
    )
    trial_frame.to_csv(output_csv, index=False)
    print(
        json.dumps(
            {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "trials_csv": str(output_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
