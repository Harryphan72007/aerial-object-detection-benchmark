"""Strict controlled/performance experiment-track isolation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.utils.serialization import read_yaml

BENCHMARK_TRACKS = frozenset({"controlled", "performance"})
COMMON_OPTIONS = frozenset(
    {"input_resolution", "batch_size", "gradient_accumulation_steps", "use_amp"}
)
PERFORMANCE_ONLY_OPTIONS = frozenset(
    {"ema_enabled", "tiled_training", "sliced_inference", "label_granularity"}
)


def validate_track_config(value: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(value)
    required = {
        "schema_version",
        "benchmark_track",
        "output_namespace",
        "comparison_table",
        "model_ids",
        "allowed_options",
        "defaults",
    }
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError("benchmark track config does not match schema v1")
    track = str(config["benchmark_track"])
    if track not in BENCHMARK_TRACKS:
        raise ValueError(f"unknown benchmark track: {track}")
    if config["output_namespace"] != track:
        raise ValueError("output namespace must equal the benchmark track")
    expected_table = f"{track}_summary"
    if config["comparison_table"] != expected_table:
        raise ValueError(f"comparison table must be {expected_table}")
    allowed = set(config["allowed_options"])
    expected_allowed = set(COMMON_OPTIONS)
    if track == "performance":
        expected_allowed.update(PERFORMANCE_ONLY_OPTIONS)
    if allowed != expected_allowed:
        raise ValueError("track allowed-options contract changed")
    validate_run_options(config, dict(config["defaults"]))
    return config


def load_track_config(repo_root: str | Path, track: str) -> dict[str, Any]:
    if track not in BENCHMARK_TRACKS:
        raise ValueError(f"unknown benchmark track: {track}")
    return validate_track_config(
        read_yaml(Path(repo_root) / "configs" / track / "benchmark.yaml")
    )


def validate_run_options(
    track_config: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    allowed = set(track_config["allowed_options"])
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(
            f"options are not allowed in {track_config['benchmark_track']} mode: {unknown}"
        )
    return dict(options)


def track_output_root(drive_root: str | Path, track: str) -> Path:
    if track not in BENCHMARK_TRACKS:
        raise ValueError(f"unknown benchmark track: {track}")
    return Path(drive_root) / "experiments" / track


def require_comparison_track(artifact: Mapping[str, Any], track: str) -> None:
    artifact_track = artifact.get("benchmark_track", "controlled")
    namespace = artifact.get("output_namespace", artifact_track)
    if artifact_track != track or namespace != track:
        raise ValueError(
            f"{artifact_track}/{namespace} artifact cannot enter {track} summary"
        )
