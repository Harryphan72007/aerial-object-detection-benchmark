"""Automatic, resumable execution of one controlled model day."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.benchmark_status import discover_model_status, find_selected_config
from src.models.registry import create_adapter, load_model_config
from src.paths import ProjectPaths
from src.training.lr_search import resolve_batch_policy
from src.training.lr_workflow import LRControlledBenchmark
from src.utils.serialization import read_json, read_yaml, write_json
from src.workflows.contract import BENCHMARK_CONTRACT, require_primary_model
from src.workflows.environment import ensure_model_environment


class Stage(str, Enum):
    ENVIRONMENT = "ENVIRONMENT"
    DATA = "DATA"
    LR_SEARCH = "LR_SEARCH"
    FINAL_TRAINING = "FINAL_TRAINING"
    EVALUATION = "EVALUATION"
    PROFILING = "PROFILING"
    REPORT = "REPORT"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class ModelDayOptions:
    model_id: str
    run_mode: str = "auto"
    run_lr_range_test: bool = True
    run_boundary_extension: bool = False
    start_expensive_stage: bool = False
    allow_over_budget_run: bool = False
    smoke_test: bool = False

    def validate(self) -> None:
        require_primary_model(self.model_id)
        allowed = {"auto", *(stage.value.lower() for stage in Stage)}
        if self.run_mode.lower() not in allowed:
            raise ValueError(f"RUN_MODE must be auto or a stage name, got {self.run_mode!r}")


def _dataset_ready(paths: ProjectPaths) -> bool:
    return all(
        path.exists()
        for path in (
            paths.coco("2class") / "annotations" / "instances_train.json",
            paths.coco("2class") / "annotations" / "instances_val.json",
            paths.images("train"),
            paths.images("val"),
        )
    )


def _report_contains_run(path: Path, run_id: str | None) -> bool:
    if not run_id or not path.is_file():
        return False
    try:
        rows = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(rows, list) and any(row.get("run_id") == run_id for row in rows)


def inspect_model_day(
    drive_root: str | Path,
    model_id: str,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Derive every path and the next incomplete stage without writing state."""
    require_primary_model(model_id)
    paths = ProjectPaths.from_value(drive_root)
    status = discover_model_status(paths.root, model_id, repo_root)
    run_id = status.get("final_run_id")
    gate_path = paths.lr_search_checkpoints / model_id / "adapter_smoke.json"
    gate = read_json(gate_path) if gate_path.is_file() else {}
    if gate.get("status") in {"FAILED_ENVIRONMENT", "FAILED_ADAPTER", "FAILED_OOM"}:
        status["environment_preflight"] = "BLOCKED"
    profile_path = paths.evaluation / f"{run_id}__profile.json" if run_id else None
    profile_complete = bool(profile_path and profile_path.is_file())
    model_report_dir = (
        paths.reports / "models" / model_id / str(run_id) if run_id else None
    )
    report_complete = bool(
        model_report_dir
        and _report_contains_run(model_report_dir / "final_results.json", run_id)
    )
    status["report_status"] = "COMPLETE" if report_complete else "NOT_STARTED"

    if not _dataset_ready(paths):
        stage = Stage.DATA
    elif (
        gate.get("status") != "READY"
        and status["lr_search_status"] == "NOT_STARTED"
        and status["final_training_status"] == "NOT_STARTED"
    ):
        stage = Stage.ENVIRONMENT
    elif status["lr_search_status"] != "COMPLETE":
        stage = Stage.LR_SEARCH
    elif status["final_training_status"] != "COMPLETE":
        stage = Stage.FINAL_TRAINING
    elif status["evaluation_status"] != "COMPLETE":
        stage = Stage.EVALUATION
    elif not profile_complete:
        stage = Stage.PROFILING
    elif not report_complete:
        stage = Stage.REPORT
    else:
        stage = Stage.COMPLETE

    selected = find_selected_config(paths.root, model_id, repo_root)
    return {
        **status,
        "stage": stage.value,
        "drive_root": str(paths.root),
        "dataset_train": str(paths.coco("2class") / "annotations" / "instances_train.json"),
        "dataset_validation": str(
            paths.coco("2class") / "annotations" / "instances_val.json"
        ),
        "train_images": str(paths.images("train")),
        "validation_images": str(paths.images("val")),
        "search_manifests": str(paths.lr_search_manifests),
        "search_state": str(
            paths.lr_search_checkpoints / model_id / "search_state.json"
        ),
        "selected_config": str(selected) if selected else None,
        "checkpoint_path": status.get("best_checkpoint"),
        "evaluation_paths": status.get("evaluation_files", []),
        "profile_path": str(profile_path) if profile_path else None,
        "report_path": str(model_report_dir / "final_report.md")
        if model_report_dir
        else None,
        "recommended_bundle": (
            f"{model_id}__2class__{run_id}" if run_id else None
        ),
        "adapter_gate": gate,
        "contract": BENCHMARK_CONTRACT,
    }


def _subset_one(source: Path, destination: Path) -> Path:
    data = read_json(source)
    if not data.get("images"):
        raise RuntimeError(f"Cannot smoke-test an empty manifest: {source}")
    image_id = int(data["images"][0]["id"])
    subset = dict(data)
    subset["images"] = [data["images"][0]]
    subset["annotations"] = [
        row for row in data.get("annotations", []) if int(row["image_id"]) == image_id
    ]
    write_json(destination, subset)
    return destination


def _is_oom(error: BaseException) -> bool:
    message = repr(error).lower()
    return "out of memory" in message or "cuda oom" in message


def _adapter_smoke(
    workflow: LRControlledBenchmark,
    model_id: str,
) -> dict[str, Any]:
    """Exercise construction, train/backward/step, validation, save, and reload."""
    gate_path = workflow.paths.lr_search_checkpoints / model_id / "adapter_smoke.json"
    if gate_path.is_file():
        gate = read_json(gate_path)
        if gate.get("status") == "READY":
            return gate
    workflow.prepare_manifests()
    smoke_root = workflow.paths.lr_search_checkpoints / model_id / "_adapter_smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    train_manifest = _subset_one(
        workflow.manifest_dir / "search_train_seed42.json",
        smoke_root / "train.json",
    )
    val_manifest = _subset_one(
        workflow.manifest_dir / "search_validation_seed42.json",
        smoke_root / "val.json",
    )
    attempts = ((2, 4), (1, 8))
    failures: list[dict[str, Any]] = []
    for batch_size, accumulation in attempts:
        policy = resolve_batch_policy(batch_size, accumulation)
        run_dir = smoke_root / f"batch{batch_size}_accum{accumulation}"
        try:
            manifest = workflow.orchestrator.run(
                model_id,
                dataset_track="2class",
                image_size=640,
                batch_size=batch_size,
                gradient_accumulation_steps=accumulation,
                epochs=1,
                seed=42,
                use_amp=True,
                overrides=workflow._benchmark_overrides(
                    workflow.resolve_baseline(model_id).learning_rate
                ),
                train_annotation_override=train_manifest,
                validation_annotation_override=val_manifest,
                train_images_override=workflow.paths.images("train"),
                validation_images_override=workflow.paths.images("train"),
                explicit_run_dir=run_dir,
                explicit_run_id=f"{model_id}__adapter_smoke",
                register_run=False,
                scheduler_horizon=15,
                validation_interval=1,
                run_kind="adapter_smoke_non_promotable",
            )
            checkpoint = run_dir / "last.pth"
            if not checkpoint.is_file():
                raise RuntimeError("Adapter smoke did not save last.pth")
            model_config = load_model_config(model_id, workflow.repo_root)
            model_config["input_resolution"] = 640
            if model_config["framework"] in {"mmdetection", "vmamba_mmdetection"}:
                model_config["resolved_framework_config"] = str(
                    run_dir / "runtime_config.py"
                )
            adapter = create_adapter(model_id)
            adapter.load_model(checkpoint, model_config)
            gate = {
                "status": "READY",
                "model_id": model_id,
                "batch_policy": policy,
                "checks": {
                    "model_construction": True,
                    "training_batch": True,
                    "forward_pass": True,
                    "backward_pass": True,
                    "optimizer_step": True,
                    "validation_prediction": True,
                    "checkpoint_save": True,
                    "checkpoint_reload": True,
                },
                "run_manifest": manifest,
            }
            write_json(gate_path, gate)
            return gate
        except Exception as error:
            failures.append(
                {
                    "batch_size": batch_size,
                    "accumulation": accumulation,
                    "error": repr(error),
                }
            )
            if not _is_oom(error):
                gate = {
                    "status": "FAILED_ADAPTER",
                    "model_id": model_id,
                    "failures": failures,
                }
                write_json(gate_path, gate)
                raise
    gate = {"status": "FAILED_OOM", "model_id": model_id, "failures": failures}
    write_json(gate_path, gate)
    raise RuntimeError("FAILED_OOM: batch 2/accumulation 4 and batch 1/accumulation 8 failed")


def _batch_policy(state: dict[str, Any]) -> tuple[int, int]:
    policy = state.get("adapter_gate", {}).get("batch_policy", {})
    batch = int(policy.get("per_device_batch_size", 2))
    accumulation = int(policy.get("gradient_accumulation_steps", 4))
    resolve_batch_policy(batch, accumulation, int(os.environ.get("WORLD_SIZE", "1")))
    return batch, accumulation


def _run_command(repo: Path, *arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], check=True, cwd=repo)


def run_model_day(
    repo_root: str | Path,
    drive_root: str | Path,
    options: ModelDayOptions,
) -> dict[str, Any]:
    """Run all currently actionable stages, skipping completed persistent state."""
    options.validate()
    repo = Path(repo_root).resolve()
    paths = ProjectPaths.from_value(drive_root)
    initial = inspect_model_day(paths.root, options.model_id, repo)
    if initial["stage"] == Stage.DATA.value:
        return {
            **initial,
            "message": "Dataset is missing. Run notebooks/00_prepare_visdrone.ipynb.",
            "next_notebook": "00_prepare_visdrone.ipynb",
        }
    if not options.start_expensive_stage:
        return {
            **initial,
            "preview": True,
            "message": (
                "Preview only. Set START_EXPENSIVE_STAGE=True after reviewing the "
                "derived paths, contract, and runtime estimate."
            ),
        }

    workflow = LRControlledBenchmark(repo, paths.root)
    history: list[str] = []
    while True:
        state = inspect_model_day(paths.root, options.model_id, repo)
        stage = Stage(state["stage"])
        if options.run_mode.lower() != "auto":
            requested = Stage(options.run_mode.upper())
            if stage != requested:
                return {
                    **state,
                    "history": history,
                    "message": f"Requested {requested.value}; current next stage is {stage.value}.",
                }
        if stage == Stage.ENVIRONMENT:
            gate_path = (
                paths.lr_search_checkpoints / options.model_id / "adapter_smoke.json"
            )
            try:
                environment = ensure_model_environment(
                    options.model_id, repo, paths.root, install_missing=True
                )
            except Exception as error:
                write_json(
                    gate_path,
                    {
                        "status": "FAILED_ENVIRONMENT",
                        "model_id": options.model_id,
                        "error": repr(error),
                    },
                )
                raise
            if environment.get("restart_required"):
                return {
                    **state,
                    "history": history,
                    "environment": environment,
                    "message": (
                        "Core packages changed. Restart the Colab session once, then "
                        "rerun this notebook from the top."
                    ),
                }
            gate = _adapter_smoke(workflow, options.model_id)
            history.append(f"ENVIRONMENT:{gate['status']}")
            continue
        batch_size, accumulation = _batch_policy(state)
        if stage == Stage.LR_SEARCH:
            try:
                summary = workflow.run_search(
                    options.model_id,
                    batch_size=batch_size,
                    accumulation=accumulation,
                    run_lr_range_test=options.run_lr_range_test,
                    run_boundary_extension=options.run_boundary_extension,
                    allow_over_budget_run=options.allow_over_budget_run,
                )
            except RuntimeError:
                search_state_path = (
                    paths.lr_search_checkpoints
                    / options.model_id
                    / "search_state.json"
                )
                search_state = (
                    read_json(search_state_path) if search_state_path.is_file() else {}
                )
                statuses = {
                    row.get("status")
                    for row in search_state.get("candidates", {}).values()
                }
                if batch_size != 2 or not statuses or statuses != {"FAILED_OOM"}:
                    raise
                gate_path = (
                    paths.lr_search_checkpoints
                    / options.model_id
                    / "adapter_smoke.json"
                )
                gate = read_json(gate_path)
                gate["batch_policy"] = resolve_batch_policy(1, 8)
                gate["oom_fallback"] = "batch 2/accumulation 4 failed during search"
                write_json(gate_path, gate)
                summary = workflow.run_search(
                    options.model_id,
                    batch_size=1,
                    accumulation=8,
                    run_lr_range_test=options.run_lr_range_test,
                    run_boundary_extension=options.run_boundary_extension,
                    allow_over_budget_run=options.allow_over_budget_run,
                )
            history.append("LR_SEARCH:COMPLETE")
            print(
                "\n".join(
                    [
                        "LR SEARCH COMPLETE",
                        "",
                        f"Model: {options.model_id}",
                        f"Baseline LR: {summary['baseline']['learning_rate']}",
                        f"Selected LR: {summary['selected']['selected_learning_rate']}",
                        f"Selected configuration: {find_selected_config(paths.root, options.model_id, repo)}",
                        f"Boundary status: {summary['selected']['boundary_status']}",
                        "Next stage: full official-train fine-tuning",
                    ]
                )
            )
            continue
        if stage == Stage.FINAL_TRAINING:
            selected = find_selected_config(paths.root, options.model_id, repo)
            if selected is None:
                raise FileNotFoundError("Selected LR configuration disappeared")
            try:
                workflow.run_final_training(
                    options.model_id,
                    selected,
                    batch_size=batch_size,
                    accumulation=accumulation,
                    allow_over_budget_run=options.allow_over_budget_run,
                    run_common_evaluation=False,
                )
            except Exception as error:
                if batch_size != 2 or not _is_oom(error):
                    raise
                gate_path = (
                    paths.lr_search_checkpoints
                    / options.model_id
                    / "adapter_smoke.json"
                )
                gate = read_json(gate_path)
                gate["batch_policy"] = resolve_batch_policy(1, 8)
                gate["oom_fallback"] = "batch 2/accumulation 4 failed during final training"
                write_json(gate_path, gate)
                workflow.run_final_training(
                    options.model_id,
                    selected,
                    batch_size=1,
                    accumulation=8,
                    allow_over_budget_run=options.allow_over_budget_run,
                    run_common_evaluation=False,
                )
            print("FULL OFFICIAL TRAINING SPLIT VERIFIED: YES")
            history.append("FINAL_TRAINING:COMPLETE")
            continue
        if stage == Stage.EVALUATION:
            _run_command(
                repo,
                str(repo / "scripts" / "evaluate.py"),
                "--drive-root",
                str(paths.root),
                "--dataset-track",
                "2class",
                "--split",
                "val",
                "--run-id",
                str(state["final_run_id"]),
                "--resolutions",
                "640",
                "--skip-profile",
            )
            history.append("EVALUATION:COMPLETE")
            continue
        if stage == Stage.PROFILING:
            profile_args = [
                str(repo / "scripts" / "profile_model.py"),
                "--drive-root",
                str(paths.root),
                "--run-id",
                str(state["final_run_id"]),
                "--batch-sizes",
                "1",
            ]
            if options.smoke_test:
                profile_args.extend(["--warmup", "1", "--iterations", "1"])
            _run_command(repo, *profile_args)
            history.append("PROFILING:COMPLETE")
            continue
        if stage == Stage.REPORT:
            from src.evaluation.report_generator import generate_report

            metric_rows = [
                read_json(path)
                for path in sorted(
                    paths.evaluation.glob(f"{state['final_run_id']}__res*__metrics.json")
                )
            ]
            if not metric_rows:
                raise RuntimeError("Cannot generate a model report without evaluation rows")
            generate_report(
                metric_rows,
                paths.reports
                / "models"
                / options.model_id
                / str(state["final_run_id"]),
            )
            history.append("REPORT:COMPLETE")
            continue
        final = inspect_model_day(paths.root, options.model_id, repo)
        return {**final, "history": history, "options": asdict(options)}
