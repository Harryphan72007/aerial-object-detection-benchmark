"""Protocol-safe aggregation of baseline and tuned multi-seed results."""
from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

from src.hpo.final_workflow import FINAL_SEEDS, IMAGE_SIZE
from src.hpo.workflow import HPO_PROTOCOL_ID
from src.paths import ProjectPaths
from src.training.checkpointing import RunRegistry
from src.utils.serialization import read_json, write_json
from src.workflows.contract import PRIMARY_MODELS


def _numeric_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _flatten_numeric(
    value: Any, prefix: str = ""
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric(child, name))
    elif (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        flattened[prefix] = float(value)
    return flattened


def compatible_final_runs(
    drive_root: str | Path, dataset_track: str
) -> list[dict[str, Any]]:
    paths = ProjectPaths.from_value(drive_root)
    return [
        run
        for run in RunRegistry(paths).list_available_runs(
            dataset_track=dataset_track, status="completed"
        )
        if run.get("model_id") in PRIMARY_MODELS
        and run.get("protocol_id") == HPO_PROTOCOL_ID
        and run.get("run_kind") == "final_complete_official_train"
        and run.get("baseline_or_tuned") in {"baseline", "tuned"}
        and int(run.get("seed", -1)) in FINAL_SEEDS
        and int(run.get("input_resolution", -1)) == IMAGE_SIZE
    ]


def aggregate_hpo_results(
    drive_root: str | Path,
    dataset_track: str,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if dataset_track not in {"2class", "10class"}:
        raise ValueError(f"unsupported dataset track: {dataset_track}")
    paths = ProjectPaths.from_value(drive_root)
    output = (
        Path(output_dir)
        if output_dir
        else paths.reports
        / "comparison"
        / HPO_PROTOCOL_ID
        / dataset_track
    )
    runs = compatible_final_runs(paths.root, dataset_track)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(
            (str(run["model_id"]), str(run["baseline_or_tuned"])), []
        ).append(run)
    comparisons: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model_id in PRIMARY_MODELS:
        for recipe in ("baseline", "tuned"):
            selected = grouped.get((model_id, recipe), [])
            by_seed = {int(run["seed"]): run for run in selected}
            absent = [seed for seed in FINAL_SEEDS if seed not in by_seed]
            if absent:
                missing.append(
                    {
                        "model_id": model_id,
                        "baseline_or_tuned": recipe,
                        "missing_seeds": absent,
                    }
                )
            metric_rows: list[dict[str, float]] = []
            included: list[dict[str, Any]] = []
            for seed in FINAL_SEEDS:
                run = by_seed.get(seed)
                if run is None:
                    continue
                metric_path = (
                    paths.evaluation
                    / f"{run['run_id']}__res{IMAGE_SIZE}__metrics.json"
                )
                if not metric_path.is_file():
                    missing.append(
                        {
                            "model_id": model_id,
                            "baseline_or_tuned": recipe,
                            "seed": seed,
                            "reason": "evaluation missing",
                        }
                    )
                    continue
                metrics = read_json(metric_path)
                if (
                    metrics.get("dataset_track") != dataset_track
                    or int(metrics.get("seed", -1)) != seed
                    or int(metrics.get("evaluation_resolution", -1))
                    != IMAGE_SIZE
                ):
                    missing.append(
                        {
                            "model_id": model_id,
                            "baseline_or_tuned": recipe,
                            "seed": seed,
                            "reason": "evaluation contract mismatch",
                        }
                    )
                    continue
                metric_rows.append(_flatten_numeric(metrics))
                included.append(
                    {"seed": seed, "run_id": run["run_id"], "metrics": str(metric_path)}
                )
            common_keys = (
                set.intersection(*(set(row) for row in metric_rows))
                if metric_rows
                else set()
            )
            summaries = {
                key: _numeric_summary([row[key] for row in metric_rows])
                for key in sorted(common_keys)
            }
            comparisons.append(
                {
                    "model_id": model_id,
                    "dataset_track": dataset_track,
                    "protocol_id": HPO_PROTOCOL_ID,
                    "baseline_or_tuned": recipe,
                    "status": (
                        "COMPLETE"
                        if len(included) == len(FINAL_SEEDS)
                        else "INCOMPLETE"
                    ),
                    "runs": included,
                    "metrics": summaries,
                }
            )
    payload = {
        "dataset_track": dataset_track,
        "protocol_id": HPO_PROTOCOL_ID,
        "required_seeds": FINAL_SEEDS,
        "groups": comparisons,
        "missing": missing,
        "note": (
            "Track A (2class) and Track B (10class) are separate protocols; "
            "their mAP values must not be compared directly."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", payload)
    return {**payload, "output": str(output / "comparison.json")}
