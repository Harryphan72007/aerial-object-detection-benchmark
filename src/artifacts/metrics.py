"""Versioned metric/prediction envelopes with legacy dual-write."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from src.artifacts.identity import ArtifactIdentity
from src.compatibility.legacy_writer import (
    write_legacy_metric_view,
    write_legacy_prediction_view,
)
from src.utils.serialization import read_json, write_json


def write_metric_artifact(
    destination: str | Path,
    identity: ArtifactIdentity,
    metrics: Mapping[str, Any],
    *,
    legacy_destination: str | Path,
) -> dict[str, Path]:
    """Write a v1 metric envelope and the frozen flat metric view."""

    versioned = Path(destination)
    payload = {
        "schema_version": 1,
        "artifact_type": "evaluation_metrics",
        "identity": identity.to_dict(),
        "metrics": dict(metrics),
    }
    _validate_metric_payload(payload)
    write_json(versioned, payload)
    legacy = write_legacy_metric_view(
        legacy_destination, identity.to_dict(), metrics
    )
    return {"versioned": versioned, "legacy": legacy}


def load_metric_artifact(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    _validate_metric_payload(payload)
    return {**payload["identity"], **payload["metrics"]}


def _validate_metric_payload(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("metric artifact must be an object")
    if set(payload) != {"schema_version", "artifact_type", "identity", "metrics"}:
        raise ValueError("metric artifact fields do not match schema v1")
    if payload["schema_version"] != 1 or payload["artifact_type"] != "evaluation_metrics":
        raise ValueError("unsupported metric artifact schema")
    if not isinstance(payload["identity"], Mapping) or not isinstance(
        payload["metrics"], Mapping
    ):
        raise ValueError("metric identity and metrics must be objects")
    ArtifactIdentity(**payload["identity"])


def write_prediction_artifact(
    destination: str | Path,
    identity: ArtifactIdentity,
    predictions: Sequence[Mapping[str, Any]],
    *,
    legacy_destination: str | Path,
) -> dict[str, Path]:
    """Write a v1 prediction envelope and the frozen COCO array view."""

    versioned = Path(destination)
    payload = {
        "schema_version": 1,
        "artifact_type": "coco_predictions",
        "identity": identity.to_dict(),
        "predictions": [dict(item) for item in predictions],
    }
    _validate_prediction_payload(payload)
    write_json(versioned, payload)
    legacy = write_legacy_prediction_view(legacy_destination, predictions)
    return {"versioned": versioned, "legacy": legacy}


def load_prediction_artifact(path: str | Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    _validate_prediction_payload(payload)
    return [dict(item) for item in payload["predictions"]]


def _validate_prediction_payload(payload: Any) -> None:
    expected = {"schema_version", "artifact_type", "identity", "predictions"}
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("prediction artifact fields do not match schema v1")
    if payload["schema_version"] != 1 or payload["artifact_type"] != "coco_predictions":
        raise ValueError("unsupported prediction artifact schema")
    ArtifactIdentity(**payload["identity"])
    if not isinstance(payload["predictions"], list):
        raise ValueError("predictions must be an array")
    required = {"image_id", "category_id", "bbox", "score"}
    for prediction in payload["predictions"]:
        if not isinstance(prediction, Mapping) or not required <= set(prediction):
            raise ValueError("invalid COCO prediction record")
