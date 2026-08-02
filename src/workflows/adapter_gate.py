"""Compatibility fingerprints for persistent adapter smoke gates."""
from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path
from typing import Any

from src.models.registry import load_model_config
from src.reproducibility import git_commit
from src.utils.serialization import read_json
from src.workflows.isolated_environment import RUNTIME_MANIFEST_ENV

ADAPTER_GATE_SCHEMA_VERSION = 2
FAILED_GATE_STATUSES = {"FAILED_ADAPTER", "FAILED_ENVIRONMENT", "FAILED_OOM"}
FINGERPRINT_FIELDS = (
    "adapter_schema_version",
    "git_commit",
    "model_id",
    "framework",
    "python_version",
    "pytorch_version",
    "cuda_version",
    "gpu",
    "dependency_lock_hash",
)


def _dependency_lock_hash(repo_root: Path) -> str:
    paths = sorted(
        {
            repo_root / "pyproject.toml",
            *(repo_root.glob("requirements*.txt")),
        },
        key=lambda path: path.as_posix(),
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(repo_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def adapter_fingerprint(
    model_id: str, repo_root: str | Path
) -> dict[str, Any]:
    """Describe every source/runtime property that can affect an adapter gate."""
    repo = Path(repo_root).resolve()
    model_config = load_model_config(model_id, repo)
    runtime_path = os.environ.get(RUNTIME_MANIFEST_ENV)
    runtime = (
        read_json(runtime_path)
        if runtime_path and Path(runtime_path).is_file()
        else {}
    )
    if runtime.get("model_id") == model_id:
        observed = runtime.get("environment", {})
        python_version = str(observed.get("python", "unknown"))
        pytorch_version = str(
            observed.get("pytorch_version", "not installed")
        )
        cuda_version = observed.get("cuda_version")
        gpu = str(observed.get("gpu_name", "CPU"))
        dependency_hash = str(runtime["dependency_lock_hash"])
    else:
        python_version = platform.python_version()
        try:
            import torch

            pytorch_version = str(torch.__version__)
            cuda_version = str(torch.version.cuda) if torch.version.cuda else None
            gpu = (
                str(torch.cuda.get_device_name(0))
                if torch.cuda.is_available()
                else "CPU"
            )
        except ImportError:
            pytorch_version = "not installed"
            cuda_version = None
            gpu = "CPU"
        dependency_hash = _dependency_lock_hash(repo)
    return {
        "adapter_schema_version": ADAPTER_GATE_SCHEMA_VERSION,
        "git_commit": git_commit(repo),
        "model_id": model_id,
        "framework": str(model_config["framework"]),
        "python_version": python_version,
        "pytorch_version": pytorch_version,
        "cuda_version": cuda_version,
        "gpu": gpu,
        "dependency_lock_hash": dependency_hash,
    }


def fingerprint_differences(
    stored: dict[str, Any] | None,
    current: dict[str, Any],
) -> list[str]:
    if not isinstance(stored, dict):
        return ["legacy gate has no fingerprint"]
    differences = []
    for field in FINGERPRINT_FIELDS:
        if field not in stored:
            differences.append(f"{field}: missing from persisted gate")
        elif stored[field] != current[field]:
            differences.append(
                f"{field}: persisted={stored[field]!r}, current={current[field]!r}"
            )
    return differences


def adapter_gate_decision(
    gate: dict[str, Any],
    current_fingerprint: dict[str, Any],
) -> tuple[str, list[str]]:
    """Return ``run``, ``reuse``, ``invalidate``, ``retry``, or ``blocked``."""
    if not gate:
        return "run", ["no persisted adapter gate"]
    differences = fingerprint_differences(gate.get("fingerprint"), current_fingerprint)
    status = str(gate.get("status", "UNKNOWN"))
    if status == "READY":
        if differences:
            return "invalidate", differences
        return "reuse", ["persisted READY fingerprint is compatible"]
    if status == "FAILED_ENVIRONMENT":
        return "retry", [
            "environment provisioning failures are retried through the transactional runtime state"
        ]
    if status in FAILED_GATE_STATUSES:
        if differences:
            return "retry", differences
        return "blocked", [
            f"persisted {status} fingerprint matches the current source and environment"
        ]
    return "retry", [f"persisted gate has unsupported status {status!r}"]


def print_gate_decision(decision: str, reasons: list[str]) -> None:
    labels = {
        "run": "ADAPTER GATE STARTED",
        "reuse": "ADAPTER GATE REUSED",
        "invalidate": "ADAPTER GATE INVALIDATED",
        "retry": "ADAPTER GATE RETRIED",
        "blocked": "ADAPTER GATE BLOCKED",
    }
    print(f"{labels[decision]}: {'; '.join(reasons)}")
