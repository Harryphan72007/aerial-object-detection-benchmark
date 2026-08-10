"""Compatibility fingerprints for persistent adapter smoke gates."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.models.adapter_validation import (
    CHECKPOINT_INCOMPLETE,
    CHECKPOINT_LOADED,
    CHECKPOINT_MISSING,
    CHECKPOINT_NOT_APPLICABLE,
    CheckpointLoadResult,
)
from src.models.registry import load_model_config
from src.reproducibility import git_commit
from src.utils.serialization import read_json
from src.workflows.isolated_environment import RUNTIME_MANIFEST_ENV

ADAPTER_GATE_SCHEMA_VERSION = 2
# Bumped whenever the meaning of a smoke check changes, so a READY record
# written by an older contract cannot authorize a run under the new one.
SMOKE_CONTRACT_VERSION = 3
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


def assert_pretrained_load_complete(result: CheckpointLoadResult) -> CheckpointLoadResult:
    """Reject anything other than a proven-complete pretrained load.

    The adapter must report one of four explicit states. There is deliberately
    no way to express "no checkpoint information": a warn-only or swallowed load
    is exactly the historical Swin and VMamba failure mode this gate exists to
    catch, and ``None`` must never read as success.
    """
    if not isinstance(result, CheckpointLoadResult):
        raise AdapterGateError(
            "adapter did not report a CheckpointLoadResult; a missing checkpoint "
            f"result cannot be treated as a successful load (got {result!r})"
        )
    if result.state == CHECKPOINT_MISSING:
        raise AdapterGateError(
            f"required pretrained checkpoint is missing: {result.source} "
            f"({result.detail})"
        )
    if result.state == CHECKPOINT_INCOMPLETE:
        raise AdapterGateError(
            f"pretrained weights loaded partially from {result.source}: "
            f"missing={list(result.missing_keys)}, "
            f"unexpected={list(result.unexpected_keys)}, "
            f"value_coverage={result.value_coverage} "
            f"(minimum {result.minimum_value_coverage}); {result.detail}"
        )
    if result.state not in {CHECKPOINT_LOADED, CHECKPOINT_NOT_APPLICABLE}:
        raise AdapterGateError(f"unsupported checkpoint state: {result.state!r}")
    return result


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
    """Adapters export one-based COCO category ids, so a valid label lies in
    ``[1, num_classes]``; a zero-based index leaking through is a defect."""
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
            if not 1 <= int(label) <= num_classes:
                raise AdapterGateError(
                    f"label {label} outside the one-based COCO range "
                    f"[1, {num_classes}]"
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


def smoke_check_result(
    name: str,
    *,
    passed: bool,
    error: BaseException | None = None,
    evidence: Mapping[str, Any] | None = None,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """One structured smoke check outcome; a failure always carries its cause."""
    if name not in SMOKE_CHECK_ORDER:
        raise AdapterGateError(f"unknown smoke check: {name!r}")
    record: dict[str, Any] = {
        "name": name,
        "passed": bool(passed),
        "duration_seconds": duration_seconds,
        "evidence": dict(evidence or {}),
    }
    if error is not None:
        record.update(
            {
                "exception_type": type(error).__name__,
                "error": str(error),
                "traceback": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                )[-4000:],
            }
        )
    return record


def build_smoke_record(
    model_id: str,
    fingerprint: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    *,
    gpu: str,
    dataset_track: str,
    image_size: int,
    checkpoint: Mapping[str, Any] | None = None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a signed smoke-gate record. Written to the artifact root, never
    to ``results/`` — a smoke pass is not a benchmark result.

    A short or out-of-order check list is itself a failure record, never a
    traceback: an aborted run must still leave complete evidence behind.
    """
    ordered = [str(check["name"]) for check in checks]
    expected_prefix = list(SMOKE_CHECK_ORDER[: len(ordered)])
    reasons: list[str] = []
    if ordered != expected_prefix:
        reasons.append(
            f"smoke checks must run in order {SMOKE_CHECK_ORDER}, got {ordered}"
        )
    missing = [name for name in SMOKE_CHECK_ORDER if name not in ordered]
    if missing:
        reasons.append(f"smoke run stopped before completing: {missing}")
    failed = [check["name"] for check in checks if not bool(check["passed"])]
    if failed:
        reasons.append(f"failed checks: {failed}")
    record = {
        "schema_version": ADAPTER_GATE_SCHEMA_VERSION,
        "smoke_contract_version": SMOKE_CONTRACT_VERSION,
        "artifact_kind": "gpu_adapter_smoke",
        "model_id": model_id,
        "dataset_track": dataset_track,
        "image_size": int(image_size),
        "gpu": gpu,
        "status": "READY" if not reasons else "FAILED_ADAPTER",
        "reasons": reasons,
        "commit": fingerprint.get("git_commit"),
        "fingerprint": dict(fingerprint),
        "checkpoint": dict(checkpoint or {}),
        "required_checks": list(SMOKE_CHECK_ORDER),
        "checks": [dict(check) for check in checks],
        "failure": dict(failure) if failure else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    signature = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**record, "signature": signature}


def smoke_record_path(
    drive_root: str | Path, model_id: str, dataset_track: str
) -> Path:
    """Locate the persisted gate record for one model and dataset track."""
    return (
        Path(drive_root).expanduser().resolve()
        / "adapter_smoke"
        / f"{model_id}__{dataset_track}__smoke.json"
    )


def smoke_gate_blockers(
    record: Mapping[str, Any] | None,
    current_fingerprint: Mapping[str, Any],
    *,
    dataset_track: str,
    image_size: int,
) -> list[str]:
    """Return every reason a persisted record cannot authorize a real run."""
    if not record:
        return ["no adapter smoke record exists"]
    blockers: list[str] = []
    status = str(record.get("status", "UNKNOWN"))
    if status != "READY":
        blockers.append(f"persisted adapter gate status is {status}")
    if int(record.get("smoke_contract_version", 0)) != SMOKE_CONTRACT_VERSION:
        blockers.append(
            "record predates the current smoke contract "
            f"(record={record.get('smoke_contract_version')}, "
            f"current={SMOKE_CONTRACT_VERSION})"
        )
    if str(record.get("dataset_track")) != dataset_track:
        blockers.append(
            f"record covers dataset track {record.get('dataset_track')!r}, "
            f"not {dataset_track!r}"
        )
    if int(record.get("image_size", 0)) != int(image_size):
        blockers.append(
            f"record covers image size {record.get('image_size')}, not {image_size}"
        )
    names = [str(check.get("name")) for check in record.get("checks", [])]
    if names != list(SMOKE_CHECK_ORDER):
        blockers.append(f"record is missing required checks: {names}")
    elif not all(bool(check.get("passed")) for check in record.get("checks", [])):
        blockers.append("record contains a failed check")
    signature = record.get("signature")
    unsigned = {key: value for key, value in record.items() if key != "signature"}
    recomputed = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if signature != recomputed:
        blockers.append("record signature does not match its contents")
    blockers.extend(
        fingerprint_differences(record.get("fingerprint"), dict(current_fingerprint))
    )
    return blockers


def require_ready_adapter_gate(
    repo_root: str | Path,
    drive_root: str | Path,
    model_id: str,
    *,
    dataset_track: str,
    image_size: int,
) -> dict[str, Any]:
    """Refuse to start an expensive run without a matching READY gate record."""
    path = smoke_record_path(drive_root, model_id, dataset_track)
    record = read_json(path) if path.is_file() else None
    blockers = smoke_gate_blockers(
        record,
        adapter_fingerprint(model_id, repo_root),
        dataset_track=dataset_track,
        image_size=image_size,
    )
    if blockers:
        raise AdapterGateError(
            f"The GPU adapter smoke gate has not passed for {model_id} on this "
            "commit, environment, and dataset track, so an expensive run must "
            "not start.\n"
            f"Record: {path}\n"
            "Blockers:\n- " + "\n- ".join(blockers) + "\n"
            "Run the gate on the target GPU first:\n"
            f"  python -m scripts.gpu_adapter_smoke --drive-root {Path(drive_root)} "
            f"--dataset-track {dataset_track} --model-id {model_id}"
        )
    assert record is not None  # blockers is non-empty when the record is missing
    return record


def print_gate_decision(decision: str, reasons: list[str]) -> None:
    labels = {
        "run": "ADAPTER GATE STARTED",
        "reuse": "ADAPTER GATE REUSED",
        "invalidate": "ADAPTER GATE INVALIDATED",
        "retry": "ADAPTER GATE RETRIED",
        "blocked": "ADAPTER GATE BLOCKED",
    }
    print(f"{labels[decision]}: {'; '.join(reasons)}")
