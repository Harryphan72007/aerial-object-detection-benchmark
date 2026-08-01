"""Create, validate, finalize, and reload v1 experiment manifests."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.utils.serialization import read_json, write_json

REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "model_id",
        "dataset_track",
        "execution_mode",
        "code",
        "config",
        "dataset",
        "environment",
        "hardware",
        "seed",
        "output_path",
        "status",
        "created_at",
        "completed_at",
        "failure",
        "result",
    }
)
STATUSES = frozenset({"running", "completed", "failed"})
EXECUTION_MODES = frozenset({"legacy", "smoke", "controlled", "performance"})
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")


class ManifestValidationError(ValueError):
    """Raised when an experiment manifest violates the v1 contract."""


def _require_mapping(
    manifest: Mapping[str, Any], field: str, required: set[str]
) -> dict[str, Any]:
    value = manifest.get(field)
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{field} must be an object")
    value = dict(value)
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing or unknown:
        raise ManifestValidationError(
            f"{field} fields mismatch: missing={missing}, unknown={unknown}"
        )
    return value


def _require_nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field} must be a non-empty string")
    return value


def _require_timestamp(value: Any, field: str) -> str:
    text = _require_nonempty(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestValidationError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ManifestValidationError(f"{field} must include a timezone")
    return text


def validate_experiment_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate and copy a v1 experiment manifest."""

    manifest = dict(value)
    missing = sorted(REQUIRED_FIELDS - set(manifest))
    unknown = sorted(set(manifest) - REQUIRED_FIELDS)
    if missing or unknown:
        raise ManifestValidationError(
            f"manifest fields mismatch: missing={missing}, unknown={unknown}"
        )
    if manifest["schema_version"] != 1:
        raise ManifestValidationError("unsupported schema_version")
    for field in ("experiment_id", "model_id", "output_path"):
        _require_nonempty(manifest[field], field)
    if manifest["dataset_track"] not in {"2class", "10class"}:
        raise ManifestValidationError("invalid dataset_track")
    if manifest["execution_mode"] not in EXECUTION_MODES:
        raise ManifestValidationError("invalid execution_mode")
    if manifest["status"] not in STATUSES:
        raise ManifestValidationError("invalid status")
    if isinstance(manifest["seed"], bool) or not isinstance(manifest["seed"], int):
        raise ManifestValidationError("seed must be an integer")
    if manifest["seed"] < 0:
        raise ManifestValidationError("seed must be non-negative")

    code = _require_mapping(manifest, "code", {"repository", "revision"})
    _require_nonempty(code["repository"], "code.repository")
    if not isinstance(code["revision"], str) or not GIT_REVISION.fullmatch(
        code["revision"]
    ):
        raise ManifestValidationError("code.revision must be a 40-character Git SHA")

    config = _require_mapping(manifest, "config", {"path", "sha256"})
    _require_nonempty(config["path"], "config.path")
    if not isinstance(config["sha256"], str) or not SHA256.fullmatch(
        config["sha256"]
    ):
        raise ManifestValidationError("config.sha256 must be a SHA-256 digest")

    dataset = _require_mapping(manifest, "dataset", {"name", "version", "hashes"})
    _require_nonempty(dataset["name"], "dataset.name")
    _require_nonempty(dataset["version"], "dataset.version")
    if not isinstance(dataset["hashes"], Mapping):
        raise ManifestValidationError("dataset.hashes must be an object")
    for name, digest in dataset["hashes"].items():
        _require_nonempty(name, "dataset hash name")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ManifestValidationError(f"dataset hash {name!r} is not SHA-256")

    environment = _require_mapping(
        manifest, "environment", {"python_version", "dependencies_sha256"}
    )
    _require_nonempty(environment["python_version"], "environment.python_version")
    if not isinstance(environment["dependencies_sha256"], str) or not SHA256.fullmatch(
        environment["dependencies_sha256"]
    ):
        raise ManifestValidationError(
            "environment.dependencies_sha256 must be a SHA-256 digest"
        )

    hardware = _require_mapping(
        manifest, "hardware", {"device_type", "device_name"}
    )
    if hardware["device_type"] not in {"cpu", "cuda", "mps", "other"}:
        raise ManifestValidationError("invalid hardware.device_type")
    _require_nonempty(hardware["device_name"], "hardware.device_name")

    _require_timestamp(manifest["created_at"], "created_at")
    if manifest["completed_at"] is not None:
        _require_timestamp(manifest["completed_at"], "completed_at")
    if manifest["result"] is not None and not isinstance(manifest["result"], Mapping):
        raise ManifestValidationError("result must be an object or null")
    if manifest["failure"] is not None:
        _require_nonempty(manifest["failure"], "failure")

    status = manifest["status"]
    if status == "running" and any(
        manifest[field] is not None for field in ("completed_at", "failure", "result")
    ):
        raise ManifestValidationError("running manifest cannot contain terminal fields")
    if status == "completed" and (
        manifest["completed_at"] is None or manifest["failure"] is not None
    ):
        raise ManifestValidationError(
            "completed manifest requires completed_at and no failure"
        )
    if status == "failed" and (
        manifest["completed_at"] is None or manifest["failure"] is None
    ):
        raise ManifestValidationError(
            "failed manifest requires completed_at and failure"
        )
    return manifest


def create_experiment_manifest(
    destination: str | Path,
    *,
    experiment_id: str,
    model_id: str,
    dataset_track: str,
    execution_mode: str,
    code: Mapping[str, str],
    config: Mapping[str, str],
    dataset: Mapping[str, Any],
    environment: Mapping[str, str],
    hardware: Mapping[str, str],
    seed: int,
    output_path: str | Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Create and atomically write a running experiment manifest."""

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "model_id": model_id,
        "dataset_track": dataset_track,
        "execution_mode": execution_mode,
        "code": dict(code),
        "config": dict(config),
        "dataset": dict(dataset),
        "environment": dict(environment),
        "hardware": dict(hardware),
        "seed": seed,
        "output_path": str(output_path),
        "status": "running",
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "failure": None,
        "result": None,
    }
    validated = validate_experiment_manifest(manifest)
    write_json(destination, validated)
    return validated


def finalize_experiment_manifest(
    path: str | Path,
    *,
    status: str,
    result: Mapping[str, Any] | None = None,
    failure: str | None = None,
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Atomically transition a running manifest to completed or failed."""

    if status not in {"completed", "failed"}:
        raise ManifestValidationError("final status must be completed or failed")
    manifest = load_experiment_manifest(path)
    if manifest["status"] != "running":
        raise ManifestValidationError("only a running manifest can be finalized")
    manifest.update(
        {
            "status": status,
            "completed_at": completed_at or datetime.now(timezone.utc).isoformat(),
            "failure": failure,
            "result": dict(result) if result is not None else None,
        }
    )
    validated = validate_experiment_manifest(manifest)
    write_json(path, validated)
    return validated


def load_experiment_manifest(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate a v1 experiment manifest."""

    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ManifestValidationError("manifest must be an object")
    return validate_experiment_manifest(value)
