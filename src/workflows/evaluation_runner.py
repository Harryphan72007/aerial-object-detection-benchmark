"""Evaluate completed final runs that do not yet have a metrics artifact.

Evaluation runs in each model's own runtime, so it is launched as a module
rather than imported. It is idempotent: a run whose metrics file already exists
is skipped, which is what makes it safe to re-enter after an interrupted
session and what lets the per-model pipeline and the report notebook call the
same function without duplicating work.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.hpo.final_workflow import IMAGE_SIZE
from src.paths import ProjectPaths
from src.subprocess_utils import run_module_in_model_runtime
from src.workflows.environment import ensure_model_environment
from src.workflows.hpo_comparison import compatible_final_runs


def pending_evaluations(
    drive_root: str | Path,
    dataset_track: str,
    *,
    model_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Compatible completed runs whose metrics artifact does not exist yet."""
    paths = ProjectPaths.from_value(drive_root)
    selected = set(model_ids) if model_ids is not None else None
    pending: list[dict[str, Any]] = []
    for run in compatible_final_runs(paths.root, dataset_track):
        if selected is not None and str(run["model_id"]) not in selected:
            continue
        metrics = paths.evaluation / f"{run['run_id']}__res{IMAGE_SIZE}__metrics.json"
        if metrics.is_file():
            continue
        pending.append({**run, "metrics_path": str(metrics)})
    return pending


def evaluate_pending_runs(
    repo_root: str | Path,
    drive_root: str | Path,
    dataset_track: str,
    *,
    model_ids: Iterable[str] | None = None,
    max_images: int | None = None,
    skip_profile: bool = False,
) -> dict[str, Any]:
    """Evaluate every compatible completed run that has no metrics yet."""
    repo = Path(repo_root).resolve()
    paths = ProjectPaths.from_value(drive_root)
    pending = pending_evaluations(paths.root, dataset_track, model_ids=model_ids)
    evaluated: list[dict[str, Any]] = []
    for run in pending:
        model_id = str(run["model_id"])
        ensure_model_environment(model_id, repo, paths.root)
        arguments: list[str] = [
            "--drive-root",
            str(paths.root),
            "--dataset-track",
            dataset_track,
            "--split",
            "val",
            "--run-id",
            str(run["run_id"]),
            "--resolutions",
            str(IMAGE_SIZE),
        ]
        if max_images:
            arguments.extend(["--max-images", str(max_images)])
        if skip_profile:
            arguments.append("--skip-profile")
        run_module_in_model_runtime(
            repo, "scripts.evaluate", *arguments, environment_name=model_id
        )
        evaluated.append({"model_id": model_id, "run_id": str(run["run_id"])})
    return {
        "dataset_track": dataset_track,
        "evaluated": evaluated,
        "still_missing": [
            {"model_id": str(run["model_id"]), "run_id": str(run["run_id"])}
            for run in pending_evaluations(
                paths.root, dataset_track, model_ids=model_ids
            )
        ],
    }
