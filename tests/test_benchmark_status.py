from __future__ import annotations

from pathlib import Path

from src.benchmark_status import (
    discover_model_status,
    find_resumable_final_run,
    recommended_next_step,
)
from src.paths import ProjectPaths
from src.utils.serialization import write_json, write_yaml

MODEL_ID = "rtdetrv2_l"
RUN_ID = "rtdetrv2_l__2class__640__20260729_120000__seed42"


def _dataset_ready(paths: ProjectPaths) -> None:
    annotation = paths.coco("2class") / "annotations" / "instances_train.json"
    write_json(annotation, {"images": [], "annotations": [], "categories": []})


def _selected_config(paths: ProjectPaths, learning_rate: float = 0.0001) -> Path:
    selected = paths.root / "lr_search_configs" / f"{MODEL_ID}_2class_selected.yaml"
    write_yaml(
        selected,
        {
            "experiment": {"model_id": MODEL_ID, "dataset_track": "2class"},
            "search": {"selected_learning_rate": learning_rate},
            "final_training": {"learning_rate": learning_rate},
        },
    )
    return selected


def _final_run(
    paths: ProjectPaths,
    *,
    status: str = "completed",
    learning_rate: float = 0.0001,
) -> Path:
    run_dir = paths.final_checkpoints / MODEL_ID / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "last.pth").write_bytes(b"checkpoint")
    write_yaml(
        run_dir / "training_config.yaml",
        {
            "model_id": MODEL_ID,
            "dataset_track": "2class",
            "image_size": 640,
            "seed": 42,
            "epochs": 25,
            "scheduler_horizon": 25,
            "effective_batch_size": 8,
            "use_amp": True,
            "run_kind": "final_complete_official_train",
            "overrides": {"learning_rate": learning_rate},
        },
    )
    manifest = {
        "run_id": RUN_ID,
        "run_dir": str(run_dir),
        "model_id": MODEL_ID,
        "dataset_track": "2class",
        "status": status,
        "created_at": "2026-07-29T12:00:00Z",
    }
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(
        paths.checkpoint_registry,
        {"schema_version": 1, "runs": {RUN_ID: manifest}},
    )
    return run_dir


def test_empty_state_recommends_dataset_setup(tmp_path):
    status = discover_model_status(tmp_path, MODEL_ID, tmp_path)
    assert status["lr_search_status"] == "NOT_STARTED"
    assert status["final_training_status"] == "NOT_STARTED"
    assert recommended_next_step(status, tmp_path).startswith("Open notebook 00")


def test_search_in_progress_and_search_complete_states(tmp_path):
    paths = ProjectPaths.from_value(tmp_path)
    _dataset_ready(paths)
    write_json(
        paths.lr_search_checkpoints / MODEL_ID / "search_state.json",
        {
            "candidates": {
                "candidate": {"learning_rate": 0.0001, "status": "PROMOTED"}
            },
            "rung_decisions": [{"epoch": 2, "promoted_candidate_ids": ["candidate"]}],
        },
    )
    status = discover_model_status(tmp_path, MODEL_ID, tmp_path)
    assert status["lr_search_status"] == "IN_PROGRESS"
    assert status["search_completed_rungs"] == [2]
    assert "resume the LR search" in recommended_next_step(status, tmp_path)

    _selected_config(paths)
    status = discover_model_status(tmp_path, MODEL_ID, tmp_path)
    assert status["lr_search_status"] == "COMPLETE"
    assert status["selected_lr"] == 0.0001
    assert "notebook 01" in recommended_next_step(status, tmp_path)


def test_final_and_evaluated_states(tmp_path):
    paths = ProjectPaths.from_value(tmp_path)
    _dataset_ready(paths)
    _selected_config(paths)
    _final_run(paths)
    status = discover_model_status(tmp_path, MODEL_ID, tmp_path)
    assert status["final_training_status"] == "COMPLETE"
    assert "notebook 01" in recommended_next_step(status, tmp_path)

    write_json(
        paths.evaluation / f"{RUN_ID}__res640__metrics.json",
        {
            "model_id": MODEL_ID,
            "dataset_track": "2class",
            "run_id": RUN_ID,
            "mAP": 0.5,
        },
    )
    status = discover_model_status(tmp_path, MODEL_ID, tmp_path)
    assert status["evaluation_status"] == "COMPLETE"
    assert "notebook 01" in recommended_next_step(status, tmp_path)

    write_json(
        paths.reports / "models" / MODEL_ID / RUN_ID / "final_results.json",
        [{"run_id": RUN_ID}],
    )
    status = discover_model_status(tmp_path, MODEL_ID, tmp_path)
    assert status["report_status"] == "COMPLETE"
    assert "notebook 02" in recommended_next_step(status, tmp_path)


def test_final_resume_requires_compatible_selected_lr(tmp_path):
    paths = ProjectPaths.from_value(tmp_path)
    _final_run(paths, status="failed", learning_rate=0.0001)
    match = find_resumable_final_run(
        tmp_path, MODEL_ID, selected_learning_rate=0.0001
    )
    assert match and match["run_id"] == RUN_ID
    assert (
        find_resumable_final_run(
            tmp_path, MODEL_ID, selected_learning_rate=0.001
        )
        is None
    )
