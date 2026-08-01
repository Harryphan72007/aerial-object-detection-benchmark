from pathlib import Path

import pytest

from src.config.benchmark_tracks import (
    load_track_config,
    require_comparison_track,
    track_output_root,
    validate_run_options,
)

ROOT = Path(__file__).resolve().parents[1]


def test_controlled_rejects_every_performance_only_option() -> None:
    controlled = load_track_config(ROOT, "controlled")
    for option in ("ema_enabled", "tiled_training", "sliced_inference", "label_granularity"):
        with pytest.raises(ValueError, match="not allowed"):
            validate_run_options(controlled, {option: False})


def test_track_roots_and_comparison_tables_are_disjoint(tmp_path: Path) -> None:
    controlled = load_track_config(ROOT, "controlled")
    performance = load_track_config(ROOT, "performance")
    assert controlled["comparison_table"] != performance["comparison_table"]
    assert track_output_root(tmp_path, "controlled") != track_output_root(
        tmp_path, "performance"
    )
    require_comparison_track(
        {"benchmark_track": "controlled", "output_namespace": "controlled"},
        "controlled",
    )
    with pytest.raises(ValueError, match="cannot enter"):
        require_comparison_track(
            {"benchmark_track": "performance", "output_namespace": "performance"},
            "controlled",
        )


def test_legacy_untagged_artifact_remains_controlled_compatible() -> None:
    require_comparison_track({}, "controlled")
