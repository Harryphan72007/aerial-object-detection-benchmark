"""One model's whole benchmark day, as a single re-entrant call.

Dataset, environment, adapter gate, HPO, final training, and evaluation used to
be six things an operator started by hand, in the right order, from four
different places. They are all idempotent or resumable already, so running them
as one ordered sequence loses nothing and removes the ordering mistakes: a
disconnected session is recovered by running the same notebook again.

``start=False`` previews every stage's contract without touching a GPU or
downloading a dataset. ``start=True`` runs them. The adapter smoke gate keeps
its full force either way — it is executed rather than typed by hand, and
training still refuses to begin without a signed READY record matching this
commit, environment, track, and resolution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config.benchmark_tracks import resolve_controlled_protocol
from src.hpo.final_workflow import FinalExperimentWorkflow
from src.hpo.workflow import TwoStageRandomHPO
from src.models.registry import load_model_config
from src.paths import ProjectPaths
from src.utils.serialization import read_json
from src.workflows.adapter_gate import (
    adapter_fingerprint,
    require_ready_adapter_gate,
    smoke_gate_blockers,
    smoke_record_path,
)
from src.workflows.adapter_smoke import run_adapter_gate
from src.workflows.dataset_setup import (
    DatasetTrackNotPreparedError,
    prepare_visdrone,
    require_prepared_dataset_track,
)
from src.workflows.environment import ensure_model_environment
from src.workflows.evaluation_runner import evaluate_pending_runs, pending_evaluations

PIPELINE_STAGES = (
    "dataset",
    "environment",
    "adapter_gate",
    "hpo",
    "finetune",
    "evaluation",
)


def _stage(name: str, status: str, **detail: Any) -> dict[str, Any]:
    return {"stage": name, "status": status, **detail}


def _dataset_stage(
    repo: Path, paths: ProjectPaths, dataset_track: str, *, start: bool
) -> dict[str, Any]:
    try:
        require_prepared_dataset_track(paths.root, dataset_track)
        return _stage("dataset", "PREPARED", action="reused")
    except DatasetTrackNotPreparedError as error:
        if not start:
            # Preparation downloads and verifies multi-gigabyte archives. A
            # preview reports what is missing; it does not start the download.
            return _stage("dataset", "PENDING", reason=str(error))
    prepare_visdrone(
        repo,
        paths.root,
        prepare_10class_track=dataset_track == "10class",
    )
    require_prepared_dataset_track(paths.root, dataset_track)
    return _stage("dataset", "PREPARED", action="prepared")


def _gate_stage(
    repo: Path,
    paths: ProjectPaths,
    model_id: str,
    dataset_track: str,
    *,
    image_size: int,
    start: bool,
) -> dict[str, Any]:
    record_path = smoke_record_path(paths.root, model_id, dataset_track)
    stored = read_json(record_path) if record_path.is_file() else None
    blockers = smoke_gate_blockers(
        stored,
        adapter_fingerprint(model_id, repo),
        dataset_track=dataset_track,
        image_size=image_size,
    )
    if not blockers:
        return _stage("adapter_gate", "READY", action="reused", record=str(record_path))
    if not start:
        return _stage(
            "adapter_gate", "PENDING", blockers=blockers, record=str(record_path)
        )
    record = run_adapter_gate(
        model_id,
        repo,
        paths.root,
        dataset_track=dataset_track,
        image_size=image_size,
    )
    # Raises AdapterGateError when the fresh record still cannot authorize a
    # run, so nothing downstream can start on an unproven adapter.
    require_ready_adapter_gate(
        repo,
        paths.root,
        model_id,
        dataset_track=dataset_track,
        image_size=image_size,
    )
    return _stage(
        "adapter_gate",
        str(record.get("status", "READY")),
        action="ran",
        record=str(record_path),
    )


def run_model_pipeline(
    repo_root: str | Path,
    drive_root: str | Path,
    model_id: str,
    dataset_track: str = "2class",
    *,
    start: bool = False,
    full_matrix: bool = False,
) -> dict[str, Any]:
    """Run or preview every stage of one model's benchmark day, in order."""
    if dataset_track not in {"2class", "10class"}:
        raise ValueError(f"unsupported dataset track: {dataset_track}")
    repo = Path(repo_root).resolve()
    paths = ProjectPaths.from_value(drive_root).create()
    # Fails immediately on an unknown model id, before any stage does work.
    load_model_config(model_id, repo)
    image_size = int(resolve_controlled_protocol(repo, model_id)["image_size"])

    def result(stages: list[dict[str, Any]]) -> dict[str, Any]:
        reached = [stage["stage"] for stage in stages]
        return {
            "model_id": model_id,
            "dataset_track": dataset_track,
            "image_size": image_size,
            "preview": not start,
            "full_matrix": full_matrix,
            "stages": stages,
            "completed_stages": reached,
            "next_stage": next(
                (name for name in PIPELINE_STAGES if name not in reached), None
            ),
            "message": (
                "Set START = True and run all cells to begin the real run."
                if not start
                else "Rerun this notebook to resume any stage that was interrupted."
            ),
        }

    stages = [_dataset_stage(repo, paths, dataset_track, start=start)]
    if stages[-1]["status"] != "PREPARED":
        return result(stages)

    stages.append(
        _stage("environment", "SKIPPED_PREVIEW")
        if not start
        else _stage(
            "environment",
            "READY",
            detail=ensure_model_environment(model_id, repo, paths.root),
        )
    )

    stages.append(
        _gate_stage(
            repo,
            paths,
            model_id,
            dataset_track,
            image_size=image_size,
            start=start,
        )
    )
    if stages[-1]["status"] not in {"READY", "PENDING"}:
        return result(stages)

    hpo = TwoStageRandomHPO(repo, paths.root, model_id, dataset_track)
    stages.append(_stage("hpo", "PREVIEW" if not start else "COMPLETE",
                         detail=hpo.run(start_expensive_stage=start)))

    final = FinalExperimentWorkflow(repo, paths.root, model_id, dataset_track)
    try:
        stages.append(
            _stage(
                "finetune",
                "PREVIEW" if not start else "COMPLETE",
                detail=final.run(start_expensive_stage=start, full_matrix=full_matrix),
            )
        )
    except FileNotFoundError as error:
        # HPO has not exported a best configuration yet. In a preview that is
        # the expected state on a first pass, not a failure.
        if start:
            raise
        return result(stages + [_stage("finetune", "PENDING", reason=str(error))])

    stages.append(
        _stage(
            "evaluation",
            "PREVIEW" if not start else "COMPLETE",
            detail=(
                {
                    "pending": pending_evaluations(
                        paths.root, dataset_track, model_ids=[model_id]
                    )
                }
                if not start
                else evaluate_pending_runs(
                    repo, paths.root, dataset_track, model_ids=[model_id]
                )
            ),
        )
    )
    return result(stages)
