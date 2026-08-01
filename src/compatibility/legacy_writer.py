"""Legacy views emitted alongside versioned artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from src.training.checkpointing import materialize_checkpoint_alias
from src.utils.serialization import write_json

LEGACY_CHECKPOINT_NAMES = {
    "last": "last.pth",
    "best_map": "best_map.pth",
    "best_aptiny": "best_aptiny.pth",
}


def write_legacy_metric_view(
    destination: str | Path,
    identity: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    """Write the flat metric dictionary expected by existing consumers."""

    target = Path(destination)
    write_json(target, {**dict(identity), **dict(metrics)})
    return target


def write_legacy_prediction_view(
    destination: str | Path,
    predictions: Sequence[Mapping[str, Any]],
) -> Path:
    """Write the existing flat COCO prediction array."""

    target = Path(destination)
    write_json(target, [dict(item) for item in predictions])
    return target


def write_legacy_checkpoint_alias(
    source: str | Path,
    run_dir: str | Path,
    role: str,
) -> Path:
    """Materialize one frozen checkpoint filename atomically."""

    if role not in LEGACY_CHECKPOINT_NAMES:
        raise ValueError(f"unsupported legacy checkpoint role: {role}")
    target = Path(run_dir) / LEGACY_CHECKPOINT_NAMES[role]
    materialize_checkpoint_alias(source, target)
    return target
