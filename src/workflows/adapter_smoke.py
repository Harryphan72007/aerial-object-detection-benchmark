"""Run the GPU adapter smoke gate for one model and persist its record.

The checks themselves live in ``scripts.gpu_adapter_smoke`` because they must
execute inside the model's own runtime — the notebook kernel does not have
MMDetection, VMamba, or the pinned Transformers stack installed. This module is
the orchestration around them: provision the runtime, launch the checks there,
and make sure a record exists on disk whatever happens, so a crash before the
first check still leaves evidence rather than silence.

Both the CLI and the model pipeline call ``run_adapter_gate``, so the gate an
operator runs by hand and the gate a notebook runs automatically are the same
code producing the same signed record.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from src.config.benchmark_tracks import resolve_controlled_protocol
from src.subprocess_utils import (
    build_model_subprocess_environment,
    python_module_command,
    run_checked,
)
from src.utils.serialization import read_json, write_json
from src.workflows.adapter_gate import (
    adapter_fingerprint,
    build_smoke_record,
    smoke_record_path,
)

SMOKE_MODULE = "scripts.gpu_adapter_smoke"


def unexpected_failure_record(
    model_id: str,
    repo_root: str | Path,
    *,
    dataset_track: str,
    image_size: int,
    error: BaseException,
    stage: str,
) -> dict[str, Any]:
    """Emit a complete FAILED_ADAPTER record when a run aborts before its checks."""
    try:
        fingerprint = adapter_fingerprint(model_id, repo_root)
    except Exception as fingerprint_error:  # noqa: BLE001 - the record must still exist
        fingerprint = {
            "adapter_schema_version": None,
            "model_id": model_id,
            "gpu": "unknown",
            "fingerprint_error": str(fingerprint_error),
        }
    return build_smoke_record(
        model_id,
        fingerprint,
        [],
        gpu=str(fingerprint.get("gpu", "unknown")),
        dataset_track=dataset_track,
        image_size=image_size,
        failure={
            "check": stage,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )[-4000:],
        },
    )


def run_in_model_runtime(
    model_id: str,
    repo_root: Path,
    record_path: Path,
    *,
    dataset_track: str,
    image_size: int,
) -> None:
    """Launch the smoke checks inside the model's isolated interpreter."""
    command = python_module_command(
        SMOKE_MODULE,
        "--in-runtime",
        "--repo-root",
        str(repo_root),
        "--dataset-track",
        dataset_track,
        "--model-id",
        model_id,
        "--image-size",
        str(image_size),
        "--record-path",
        str(record_path),
    )
    run_checked(
        command,
        cwd=repo_root,
        env=build_model_subprocess_environment(),
        environment_name=f"{model_id} adapter smoke",
        stage="gpu_adapter_smoke",
        python_executable=command[0],
    )


def run_adapter_gate(
    model_id: str,
    repo_root: str | Path,
    drive_root: str | Path,
    *,
    dataset_track: str = "2class",
    image_size: int | None = None,
    skip_provisioning: bool = False,
) -> dict[str, Any]:
    """Run the gate for one model and return the record it wrote.

    The previous record is deleted before the run starts: a stale READY record
    left in place while a new run fails would be indistinguishable from a pass.
    A raised exception is caught and written as a failure record rather than
    propagated, because callers decide what a failure means — the CLI reports
    it, the pipeline refuses to continue.
    """
    repo = Path(repo_root).resolve()
    protocol = resolve_controlled_protocol(repo, model_id)
    resolved_size = int(image_size or protocol["image_size"])
    record_path = smoke_record_path(drive_root, model_id, dataset_track)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if record_path.exists():
        record_path.unlink()
    try:
        if not skip_provisioning:
            from src.workflows.environment import ensure_model_environment

            ensure_model_environment(model_id, repo, drive_root)
        run_in_model_runtime(
            model_id,
            repo,
            record_path,
            dataset_track=dataset_track,
            image_size=resolved_size,
        )
    except Exception as error:  # noqa: BLE001 - never lose the failure record
        if not record_path.is_file():
            write_json(
                record_path,
                unexpected_failure_record(
                    model_id,
                    repo,
                    dataset_track=dataset_track,
                    image_size=resolved_size,
                    error=error,
                    stage="constructs",
                ),
            )
        traceback.print_exc()
    if record_path.is_file():
        return dict(read_json(record_path))
    return {"model_id": model_id, "status": "FAILED_ADAPTER", "record": str(record_path)}
