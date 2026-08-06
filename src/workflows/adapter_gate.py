"""Compatibility fingerprints for persistent adapter smoke gates."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

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


class AdapterGateError(RuntimeError):
    """A GPU adapter smoke check failed; a full run must not start."""


def assert_pretrained_load_complete(
    missing_keys: Sequence[str] | None,
    unexpected_keys: Sequence[str] | None,
) -> None:
    """Reject a partial weight load. A warn-only load hides the historical Swin
    and VMamba failure modes; here any missing/unexpected key is fatal."""
    missing = list(missing_keys or [])
    unexpected = list(unexpected_keys or [])
    if missing or unexpected:
        raise AdapterGateError(
            f"pretrained weights loaded partially: missing={missing}, "
            f"unexpected={unexpected}"
        )


def assert_feature_map_contract(
    feature_maps: Sequence[Mapping[str, Any]],
    expected: Sequence[tuple[int, int]],
    image_size: int,
) -> None:
    """Verify backbone/FPN feature maps: channels, strides, and NCHW spatial
    dims at the *configured* resolution (not 224). ``expected`` is an ordered
    sequence of ``(channels, stride)`` and ``feature_maps`` carry an NCHW
    ``shape`` and a ``stride``."""
    if len(feature_maps) != len(expected):
        raise AdapterGateError(
            f"expected {len(expected)} feature levels, got {len(feature_maps)}"
        )
    for level, (fmap, (channels, stride)) in enumerate(zip(feature_maps, expected)):
        shape = tuple(int(dim) for dim in fmap["shape"])
        if len(shape) != 4:
            raise AdapterGateError(
                f"feature level {level} is not NCHW: shape={shape}"
            )
        _, c, h, w = shape
        if c != channels:
            raise AdapterGateError(
                f"feature level {level} has {c} channels, expected {channels}"
            )
        if int(fmap.get("stride", -1)) != stride:
            raise AdapterGateError(
                f"feature level {level} stride {fmap.get('stride')} != {stride}"
            )
        expected_hw = image_size // stride
        if h != expected_hw or w != expected_hw:
            raise AdapterGateError(
                f"feature level {level} spatial {h}x{w} != "
                f"{expected_hw}x{expected_hw} for stride {stride} at "
                f"resolution {image_size}"
            )


def assert_detection_head_class_count(
    head_class_count: int, expected_classes: int
) -> None:
    """The detection head must be reset to the track's class count with no
    COCO-80 residue."""
    if int(head_class_count) != int(expected_classes):
        raise AdapterGateError(
            f"detection head exposes {head_class_count} classes, expected "
            f"{expected_classes} (COCO-80 residue?)"
        )


def assert_finite_loss(loss: Any) -> float:
    value = float(loss)
    if not math.isfinite(value):
        raise AdapterGateError(f"non-finite training loss: {value}")
    return value


def assert_wellformed_predictions(
    predictions: Sequence[Mapping[str, Any]], *, num_classes: int
) -> None:
    for item in predictions:
        boxes = list(item["boxes"])
        scores = list(item["scores"])
        labels = list(item["labels"])
        if not (len(boxes) == len(scores) == len(labels)):
            raise AdapterGateError("prediction boxes/scores/labels length mismatch")
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = (float(value) for value in box)
            if x2 < x1 or y2 < y1:
                raise AdapterGateError(f"degenerate box: {(x1, y1, x2, y2)}")
            if not 0.0 <= float(score) <= 1.0:
                raise AdapterGateError(f"score outside [0, 1]: {score}")
            if not 0 <= int(label) < num_classes:
                raise AdapterGateError(
                    f"label {label} outside [0, {num_classes})"
                )


def assert_checkpoint_roundtrip(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    """Parameters must be identical after save -> load."""
    if set(before) != set(after):
        raise AdapterGateError("parameter names changed across save/load")
    mismatched = [key for key in before if before[key] != after[key]]
    if mismatched:
        raise AdapterGateError(
            f"parameters changed across save/load: {mismatched[:5]}"
        )


SMOKE_CHECK_ORDER = (
    "constructs",
    "pretrained_weights_complete",
    "feature_map_contract",
    "detection_head_class_count",
    "forward_backward_finite_loss",
    "predict_wellformed",
    "checkpoint_roundtrip",
)


def build_smoke_record(
    model_id: str,
    fingerprint: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    *,
    gpu: str,
) -> dict[str, Any]:
    """Assemble a signed smoke-gate record. Written to the artifact root, never
    to ``results/`` — a smoke pass is not a benchmark result."""
    ordered = [str(check["name"]) for check in checks]
    if ordered != list(SMOKE_CHECK_ORDER):
        raise AdapterGateError(
            f"smoke checks must run in order {SMOKE_CHECK_ORDER}, got {ordered}"
        )
    passed = all(bool(check["passed"]) for check in checks)
    record = {
        "schema_version": ADAPTER_GATE_SCHEMA_VERSION,
        "artifact_kind": "gpu_adapter_smoke",
        "model_id": model_id,
        "gpu": gpu,
        "status": "READY" if passed else "FAILED_ADAPTER",
        "commit": fingerprint.get("git_commit"),
        "fingerprint": dict(fingerprint),
        "checks": [dict(check) for check in checks],
    }
    signature = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**record, "signature": signature}


def print_gate_decision(decision: str, reasons: list[str]) -> None:
    labels = {
        "run": "ADAPTER GATE STARTED",
        "reuse": "ADAPTER GATE REUSED",
        "invalidate": "ADAPTER GATE INVALIDATED",
        "retry": "ADAPTER GATE RETRIED",
        "blocked": "ADAPTER GATE BLOCKED",
    }
    print(f"{labels[decision]}: {'; '.join(reasons)}")
