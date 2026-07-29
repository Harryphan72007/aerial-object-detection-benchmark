from __future__ import annotations

from pathlib import Path

import pytest

from src.paths import ProjectPaths
from src.utils.serialization import write_json, write_yaml
from src.workflows.comparison import compare_completed_models


def _completed_run(
    paths: ProjectPaths,
    model_id: str,
    *,
    image_size: int = 640,
) -> dict:
    run_id = f"{model_id}__2class__{image_size}__20260729_120000__seed42"
    run_dir = paths.final_checkpoints / model_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(
        run_dir / "training_config.yaml",
        {
            "model_id": model_id,
            "dataset_track": "2class",
            "image_size": image_size,
            "seed": 42,
            "epochs": 25,
            "scheduler_horizon": 25,
            "effective_batch_size": 8,
            "use_amp": True,
            "run_kind": "final_complete_official_train",
            "overrides": {"learning_rate": 0.0001},
        },
    )
    write_json(
        paths.evaluation / f"{run_id}__res640__metrics.json",
        {
            "run_id": run_id,
            "model_id": model_id,
            "dataset_track": "2class",
            "evaluation_resolution": 640,
            "seed": 42,
            "mAP": 0.5,
            "APtiny": 0.25,
            "per_class": {
                "person": {"AP": 0.4},
                "vehicle": {"AP": 0.6},
            },
        },
    )
    write_json(
        paths.evaluation / f"{run_id}__profile.json",
        {
            "profiles": [
                {
                    "batch_size": 1,
                    "status": "completed",
                    "median_latency_ms": 10.0,
                    "fps": 100.0,
                }
            ]
        },
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_id": model_id,
        "architecture_family": "test",
        "dataset_track": "2class",
        "status": "completed",
        "created_at": "2026-07-29T12:00:00Z",
    }


def test_comparison_rejects_incompatible_runs(tmp_path):
    paths = ProjectPaths.from_value(tmp_path).create()
    compatible = _completed_run(paths, "rtdetrv2_l")
    incompatible = _completed_run(paths, "faster_rcnn_resnet50", image_size=1024)
    write_json(
        paths.checkpoint_registry,
        {
            "schema_version": 1,
            "runs": {
                compatible["run_id"]: compatible,
                incompatible["run_id"]: incompatible,
            },
        },
    )
    with pytest.raises(RuntimeError, match="At least two compatible"):
        compare_completed_models(paths.root)
