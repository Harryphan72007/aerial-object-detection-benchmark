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

# The controlled two-stage HPO protocol is defined once, in
# ``configs/controlled/benchmark.yaml`` under a ``protocol:`` block. These are
# the fields the search and final workflows must read from there rather than
# hardcoding.
PROTOCOL_REQUIRED_FIELDS = frozenset(
    {
        "protocol_id",
        "image_size",
        "batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "use_amp",
        "search_seed",
        "phase_trials",
        "phase_a_epochs",
        "phase_b_epochs",
        "final_train_epochs",
        "final_recipes",
        "final_seeds",
        "full_matrix_recipes",
        "full_matrix_seeds",
    }
)
_PROTOCOL_RECIPE_CHOICES = frozenset({"baseline", "tuned"})
# A per-model config must never redeclare any of these; they are frozen by the
# controlled track so every model shares them (learning_rate is the only tuned
# value). ``epochs`` and ``seed`` are included to catch legacy field names.
PROTOCOL_FROZEN_FIELDS = frozenset(
    {
        "image_size",
        "batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "use_amp",
        "search_seed",
        "phase_trials",
        "phase_a_epochs",
        "phase_b_epochs",
        "final_train_epochs",
        "final_recipes",
        "final_seeds",
        "full_matrix_recipes",
        "full_matrix_seeds",
        "epochs",
        "seed",
    }
)


def _validate_protocol_block(
    value: Mapping[str, Any], model_ids: set[str]
) -> dict[str, Any]:
    del model_ids  # every model shares one budget; no per-model epoch overrides
    protocol = dict(value)
    if "model_epoch_overrides" in protocol:
        raise ValueError(
            "controlled track forbids per-model epoch overrides; every model "
            "must share an identical epoch budget"
        )
    missing = sorted(PROTOCOL_REQUIRED_FIELDS - set(protocol))
    unknown = sorted(set(protocol) - PROTOCOL_REQUIRED_FIELDS)
    if missing:
        raise ValueError(f"protocol block missing required fields: {missing}")
    if unknown:
        raise ValueError(f"protocol block has unknown fields: {unknown}")
    effective = int(protocol["batch_size"]) * int(
        protocol["gradient_accumulation_steps"]
    )
    if effective != int(protocol["effective_batch_size"]):
        raise ValueError(
            "protocol effective_batch_size must equal batch_size * "
            f"gradient_accumulation_steps ({effective})"
        )
    for field in ("final_seeds", "full_matrix_seeds"):
        seeds = protocol[field]
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"protocol {field} must be a non-empty list")
    for field in ("final_recipes", "full_matrix_recipes"):
        recipes = protocol[field]
        if not isinstance(recipes, list) or not recipes:
            raise ValueError(f"protocol {field} must be a non-empty list")
        invalid = sorted(set(recipes) - _PROTOCOL_RECIPE_CHOICES)
        if invalid:
            raise ValueError(
                f"protocol {field} may only contain "
                f"{sorted(_PROTOCOL_RECIPE_CHOICES)}, got {invalid}"
            )
    return dict(protocol)


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
    optional = {"protocol"}
    keys = set(config)
    if (
        not required <= keys
        or not keys <= (required | optional)
        or config["schema_version"] != 1
    ):
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
    if "protocol" in config:
        config["protocol"] = _validate_protocol_block(
            config["protocol"], set(config["model_ids"])
        )
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


def load_protocol(repo_root: str | Path, track: str = "controlled") -> dict[str, Any]:
    """Return the validated protocol block for a track's benchmark config."""
    config = load_track_config(repo_root, track)
    protocol = config.get("protocol")
    if not protocol:
        raise ValueError(f"{track} track defines no protocol block")
    return dict(protocol)


def reject_frozen_protocol_overrides(model_config: Mapping[str, Any]) -> None:
    """Raise if a per-model config redeclares any controlled-frozen field."""
    conflicts = sorted(PROTOCOL_FROZEN_FIELDS & set(model_config))
    if conflicts:
        raise ValueError(
            "per-model config must not override controlled-track-frozen protocol "
            f"fields: {conflicts}"
        )


def resolve_controlled_protocol(
    repo_root: str | Path, model_id: str
) -> dict[str, Any]:
    """Resolve the controlled protocol for one model.

    The protocol is identical for every controlled-track model — the whole point
    of the track. There are no per-model epoch deviations; the only per-model
    degree of freedom is the tuned learning rate, which is not part of this
    protocol. The per-model ``model.yaml`` is checked to ensure it does not
    silently redefine a frozen field.
    """
    config = load_track_config(repo_root, "controlled")
    if model_id not in set(config["model_ids"]):
        raise ValueError(f"{model_id!r} is not a controlled-track model")
    protocol = config.get("protocol")
    if not protocol:
        raise ValueError("controlled track defines no protocol block")

    model_config_path = Path(repo_root) / "configs" / model_id / "model.yaml"
    if model_config_path.is_file():
        reject_frozen_protocol_overrides(read_yaml(model_config_path))

    resolved = {field: protocol[field] for field in PROTOCOL_REQUIRED_FIELDS}
    for field in ("final_seeds", "full_matrix_seeds"):
        resolved[field] = tuple(int(seed) for seed in protocol[field])
    for field in ("final_recipes", "full_matrix_recipes"):
        resolved[field] = tuple(str(recipe) for recipe in protocol[field])
    resolved["phase_b_epochs"] = int(protocol["phase_b_epochs"])
    resolved["final_train_epochs"] = int(protocol["final_train_epochs"])
    return resolved


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
