"""Versioned checkpoint metadata and legacy filename aliases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.artifacts.identity import ArtifactIdentity
from src.compatibility.legacy_writer import write_legacy_checkpoint_alias
from src.training.checkpointing import materialize_checkpoint_alias
from src.utils.serialization import read_json, sha256_file, write_json

CHECKPOINT_ROLES = frozenset({"last", "best_map", "best_aptiny"})


def write_checkpoint_artifact(
    source: str | Path,
    artifact_dir: str | Path,
    identity: ArtifactIdentity,
    *,
    role: str,
    legacy_run_dir: str | Path,
    state_keys: list[str] | None = None,
) -> dict[str, Path]:
    """Copy a checkpoint atomically, describe it, and emit its legacy alias."""

    if role not in CHECKPOINT_ROLES:
        raise ValueError(f"unsupported checkpoint role: {role}")
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output = Path(artifact_dir)
    versioned_checkpoint = output / f"checkpoint_{role}.v1.pth"
    materialize_checkpoint_alias(source_path, versioned_checkpoint)
    metadata = {
        "schema_version": 1,
        "artifact_type": "checkpoint",
        "identity": identity.to_dict(),
        "role": role,
        "checkpoint_file": versioned_checkpoint.name,
        "checkpoint_sha256": sha256_file(versioned_checkpoint),
        "state_keys": sorted(state_keys or []),
    }
    metadata_path = output / f"checkpoint_{role}.v1.json"
    write_json(metadata_path, metadata)
    legacy = write_legacy_checkpoint_alias(
        versioned_checkpoint, legacy_run_dir, role
    )
    return {
        "checkpoint": versioned_checkpoint,
        "metadata": metadata_path,
        "legacy": legacy,
    }


def load_checkpoint_artifact(path: str | Path) -> dict[str, Any]:
    """Validate checkpoint metadata and return it with the resolved file path."""

    metadata_path = Path(path)
    value = read_json(metadata_path)
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint metadata must be an object")
    expected = {
        "schema_version",
        "artifact_type",
        "identity",
        "role",
        "checkpoint_file",
        "checkpoint_sha256",
        "state_keys",
    }
    if set(value) != expected:
        raise ValueError("checkpoint metadata fields do not match schema v1")
    if value["schema_version"] != 1 or value["artifact_type"] != "checkpoint":
        raise ValueError("unsupported checkpoint artifact schema")
    ArtifactIdentity(**value["identity"])
    if value["role"] not in CHECKPOINT_ROLES:
        raise ValueError("invalid checkpoint role")
    checkpoint = metadata_path.parent / str(value["checkpoint_file"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if sha256_file(checkpoint) != value["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA-256 mismatch")
    return {**value, "path": str(checkpoint)}
