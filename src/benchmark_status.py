"""Read-only discovery of benchmark progress and recommended next actions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.paths import ProjectPaths
from src.training.checkpointing import resolve_manifest_checkpoint
from src.utils.serialization import read_json, read_yaml

PRIMARY_MODELS = (
    "faster_rcnn_resnet50",
    "faster_rcnn_swin_t",
    "faster_rcnn_vmamba_t",
    "rtdetrv2_l",
)
VALID_STATUSES = {"NOT_STARTED", "IN_PROGRESS", "COMPLETE", "FAILED", "BLOCKED"}


def _selected_checkpoint(manifest: dict[str, Any]) -> str | None:
    if not manifest:
        return None
    try:
        return str(
            resolve_manifest_checkpoint(
                manifest,
                allow_resume=manifest.get("status") != "completed",
                allow_legacy_aliases=True,
            )
        )
    except FileNotFoundError:
        return None


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def find_selected_config(
    drive_root: str | Path,
    model_id: str,
    repo_root: str | Path = ".",
) -> Path | None:
    """Find a selected LR config without requiring a copied manual path."""
    filename = f"{model_id}_2class_selected.yaml"
    candidates = (
        Path(drive_root).expanduser() / "lr_search_configs" / filename,
        Path(repo_root).resolve() / "configs" / "lr_search" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _final_run_records(paths: ProjectPaths, model_id: str) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    registry_path = paths.checkpoint_registry
    if registry_path.is_file():
        registry = _safe_json(registry_path).get("runs", {})
        for run_id, run in registry.items():
            if run.get("model_id") != model_id or run.get("dataset_track") != "2class":
                continue
            run_dir = Path(
                run.get("run_dir")
                or paths.final_checkpoints / model_id / str(run_id)
            )
            if "final" not in {part.lower() for part in run_dir.parts}:
                continue
            records[str(run_id)] = {**run, "run_dir": str(run_dir)}
    model_root = paths.final_checkpoints / model_id
    if model_root.is_dir():
        for manifest_path in model_root.glob("*/run_manifest.json"):
            run = _safe_json(manifest_path)
            run_id = str(run.get("run_id") or manifest_path.parent.name)
            records[run_id] = {
                **records.get(run_id, {}),
                **run,
                "run_id": run_id,
                "run_dir": str(manifest_path.parent),
            }
    return sorted(
        records.values(),
        key=lambda row: (
            str(row.get("created_at", "")),
            Path(str(row.get("run_dir", "."))).stat().st_mtime
            if Path(str(row.get("run_dir", "."))).exists()
            else 0,
        ),
        reverse=True,
    )


def find_resumable_final_run(
    drive_root: str | Path,
    model_id: str,
    *,
    selected_learning_rate: float | None = None,
) -> dict[str, Any] | None:
    """Return the latest compatible incomplete final run with a last checkpoint."""
    paths = ProjectPaths.from_value(drive_root)
    for run in _final_run_records(paths, model_id):
        if run.get("status") == "completed":
            continue
        run_dir = Path(str(run["run_dir"]))
        checkpoint = run_dir / "last.pth"
        config_path = run_dir / "training_config.yaml"
        if not checkpoint.is_file() or not config_path.is_file():
            continue
        config = read_yaml(config_path)
        if (
            config.get("model_id") != model_id
            or config.get("dataset_track") != "2class"
            or int(config.get("image_size", -1)) != 640
            or int(config.get("seed", -1)) != 42
            or int(config.get("scheduler_horizon", -1)) != 25
            or int(config.get("effective_batch_size", -1)) != 8
            or not bool(config.get("use_amp"))
            or config.get("run_kind") != "final_complete_official_train"
        ):
            continue
        configured_lr = config.get("overrides", {}).get("learning_rate")
        if (
            selected_learning_rate is not None
            and configured_lr is not None
            and abs(float(configured_lr) - selected_learning_rate)
            > max(abs(selected_learning_rate), 1e-12) * 1e-12
        ):
            continue
        return run
    return None


def _search_status(
    paths: ProjectPaths, model_id: str, selected_config: Path | None
) -> tuple[str, dict[str, Any]]:
    if selected_config is not None:
        return "COMPLETE", {}
    failure = _safe_json(
        paths.lr_search_checkpoints / model_id / "model_failure.json"
    )
    if failure:
        status = str(failure.get("status", "FAILED"))
        return ("BLOCKED" if status in {"FAILED_ENVIRONMENT", "FAILED_ADAPTER"} else "FAILED"), failure
    state_path = paths.lr_search_checkpoints / model_id / "search_state.json"
    if not state_path.is_file():
        return "NOT_STARTED", {}
    state = _safe_json(state_path)
    decisions = state.get("rung_decisions", [])
    candidates = list(state.get("candidates", {}).values())
    if len(decisions) >= 4:
        return "FAILED", state
    if candidates and all(str(row.get("status", "")).startswith("FAILED") for row in candidates):
        return "FAILED", state
    return "IN_PROGRESS", state


def _final_status(records: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    completed = next(
        (
            run
            for run in records
            if run.get("status") == "completed"
            and (Path(str(run["run_dir"])) / "last.pth").is_file()
        ),
        None,
    )
    if completed:
        return "COMPLETE", completed
    active = records[0] if records else None
    if not active:
        return "NOT_STARTED", {}
    if active.get("status") == "failed":
        return (
            "IN_PROGRESS"
            if (Path(str(active["run_dir"])) / "last.pth").is_file()
            else "FAILED",
            active,
        )
    return "IN_PROGRESS", active


def _report_contains_run(report_json: Path, run_id: str | None) -> bool:
    if not run_id or not report_json.is_file():
        return False
    try:
        rows = read_json(report_json)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(rows, list) and any(row.get("run_id") == run_id for row in rows)


def _bundle_for_model(paths: ProjectPaths, model_id: str) -> tuple[str, Path | None]:
    if not paths.result_bundles.is_dir():
        return "NOT_STARTED", None
    matching: list[Path] = []
    for manifest_path in paths.result_bundles.glob("*/bundle_manifest.json"):
        manifest = _safe_json(manifest_path)
        if model_id in manifest.get("model_ids", []):
            matching.append(manifest_path.parent)
    if not matching:
        return "NOT_STARTED", None
    latest = max(matching, key=lambda path: path.stat().st_mtime)
    try:
        from src.result_export import validate_bundle

        errors = validate_bundle(latest)
    except Exception:
        return "FAILED", latest
    return ("COMPLETE" if not errors else "FAILED"), latest


def discover_model_status(
    drive_root: str | Path,
    model_id: str,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Return all progress fields for one primary model without modifying files."""
    if model_id not in PRIMARY_MODELS:
        raise ValueError(f"unsupported model: {model_id}")
    paths = ProjectPaths.from_value(drive_root)
    selected_path = find_selected_config(paths.root, model_id, repo_root)
    search_status, search_state = _search_status(paths, model_id, selected_path)
    selected_lr = None
    if selected_path:
        try:
            selected_lr = float(
                read_yaml(selected_path)["search"]["selected_learning_rate"]
            )
        except (KeyError, TypeError, ValueError, OSError):
            search_status = "FAILED"
    records = _final_run_records(paths, model_id)
    final_status, final_run = _final_status(records)
    run_id = str(final_run.get("run_id", "")) or None
    evaluation_files = (
        sorted(paths.evaluation.glob(f"{run_id}__res*__metrics.json"))
        if run_id and paths.evaluation.is_dir()
        else []
    )
    evaluation_status = "COMPLETE" if evaluation_files else "NOT_STARTED"
    model_report_dir = (
        paths.reports / "models" / model_id / str(run_id) if run_id else None
    )
    report_json = (
        model_report_dir / "final_results.json"
        if model_report_dir
        else paths.reports / "models" / model_id / "_missing" / "final_results.json"
    )
    report_status = (
        "COMPLETE"
        if _report_contains_run(report_json, run_id)
        else "NOT_STARTED"
    )
    bundle_status, bundle_path = _bundle_for_model(paths, model_id)
    failure = _safe_json(
        paths.lr_search_checkpoints / model_id / "model_failure.json"
    )
    if failure:
        environment_status = (
            "BLOCKED"
            if failure.get("status") in {"FAILED_ENVIRONMENT", "FAILED_ADAPTER"}
            else "FAILED"
        )
    elif search_status != "NOT_STARTED" or final_status != "NOT_STARTED":
        environment_status = "COMPLETE"
    else:
        environment_status = "NOT_STARTED"
    result = {
        "model_id": model_id,
        "environment_preflight": environment_status,
        "lr_search_status": search_status,
        "selected_lr": selected_lr,
        "selected_config": str(selected_path) if selected_path else None,
        "final_training_status": final_status,
        "final_run_id": run_id,
        "final_run_dir": final_run.get("run_dir"),
        "best_checkpoint": _selected_checkpoint(final_run),
        "evaluation_status": evaluation_status,
        "evaluation_files": [str(path) for path in evaluation_files],
        "report_status": report_status,
        "report_path": (
            str(model_report_dir / "final_report.md")
            if report_status == "COMPLETE" and model_report_dir
            else None
        ),
        "bundle_status": bundle_status,
        "bundle_id": bundle_path.name if bundle_path else None,
        "bundle_path": str(bundle_path) if bundle_path else None,
        "search_completed_rungs": [
            int(row["epoch"]) for row in search_state.get("rung_decisions", [])
        ],
    }
    for status_key in (
        "environment_preflight",
        "lr_search_status",
        "final_training_status",
        "evaluation_status",
        "report_status",
        "bundle_status",
    ):
        assert result[status_key] in VALID_STATUSES
    return result


def discover_all_statuses(
    drive_root: str | Path,
    model_ids: Iterable[str] = PRIMARY_MODELS,
    repo_root: str | Path = ".",
) -> list[dict[str, Any]]:
    return [
        discover_model_status(drive_root, model_id, repo_root)
        for model_id in model_ids
    ]


def recommended_next_step(status: dict[str, Any], drive_root: str | Path) -> str:
    paths = ProjectPaths.from_value(drive_root)
    train_annotations = (
        paths.coco("2class") / "annotations" / "instances_train.json"
    )
    if not train_annotations.is_file():
        return "Open notebook 00 and prepare and validate VisDrone."
    if status["environment_preflight"] == "BLOCKED":
        return "Fix the model environment preflight, then rerun notebook 01."
    if status["lr_search_status"] == "NOT_STARTED":
        return "Open notebook 01 and start the LR search."
    if status["lr_search_status"] == "IN_PROGRESS":
        return "Open notebook 01 and resume the LR search."
    if status["lr_search_status"] in {"FAILED", "BLOCKED"}:
        return "Inspect the saved search failure report before rerunning notebook 01."
    if status["final_training_status"] == "NOT_STARTED":
        return "LR search is complete. Rerun notebook 01 for full-dataset fine-tuning."
    if status["final_training_status"] == "IN_PROGRESS":
        return "Rerun notebook 01 to resume the compatible final-training run."
    if status["final_training_status"] in {"FAILED", "BLOCKED"}:
        return "Inspect the final run manifest and correct the failure before rerunning notebook 01."
    if status["evaluation_status"] != "COMPLETE":
        return "Final training is complete. Rerun notebook 01 for evaluation."
    if status["report_status"] != "COMPLETE":
        return "Evaluation is complete. Rerun notebook 01 to generate the report."
    if status["bundle_status"] != "COMPLETE":
        return "Results are ready. Open notebook 02 to create and validate a dry-run bundle."
    return "Validated bundle is ready. Review notebook 02 and publish to experiment-results."


def format_status_table(rows: list[dict[str, Any]]) -> str:
    headers = (
        "Model",
        "Environment preflight",
        "LR search status",
        "Selected LR",
        "Final training status",
        "Final run ID",
        "Evaluation status",
        "Report status",
        "Bundle status",
    )
    keys = (
        "model_id",
        "environment_preflight",
        "lr_search_status",
        "selected_lr",
        "final_training_status",
        "final_run_id",
        "evaluation_status",
        "report_status",
        "bundle_status",
    )
    values = [
        [
            f"{row[key]:.6g}" if key == "selected_lr" and row[key] is not None
            else str(row[key] if row[key] is not None else "-")
            for key in keys
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]
    separator = "-+-".join("-" * width for width in widths)
    lines = [
        " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        separator,
    ]
    lines.extend(
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in values
    )
    return "\n".join(lines)


def format_preflight_summary(values: dict[str, Any]) -> str:
    ordered = (
        ("Model", "model"),
        ("Dataset track", "dataset_track"),
        ("Mode", "mode"),
        ("Train manifest", "train_manifest"),
        ("Validation manifest", "validation_manifest"),
        ("Training images", "training_images"),
        ("Validation images", "validation_images"),
        ("Full official train", "full_official_train"),
        ("Image size", "image_size"),
        ("Batch size", "batch_size"),
        ("Gradient accumulation", "gradient_accumulation"),
        ("Effective batch size", "effective_batch_size"),
        ("Learning rate", "learning_rate"),
        ("Epoch budget", "epoch_budget"),
        ("GPU", "gpu"),
        ("Estimated runtime", "estimated_runtime"),
        ("Output directory", "output_directory"),
        ("Resume status", "resume_status"),
        ("Git commit", "git_commit"),
    )
    return "\n".join(
        ["BENCHMARK PREFLIGHT", "==================="]
        + [f"{label}: {values.get(key, '-')}" for label, key in ordered]
    )
