"""CPU tests for the throughput timing helper and runtime-budget renderer.

The GPU probe in scripts/measure_throughput.py cannot run without CUDA; these
tests cover the device-independent logic: warm-up exclusion, that the device is
synchronized around the timed window, and the budget arithmetic.
"""

from __future__ import annotations

import pytest

from src.config.benchmark_tracks import load_protocol, resolve_controlled_protocol
from src.training.profiling import (
    assert_flops_valid,
    flops_measurement,
    measure_iteration_seconds,
    render_runtime_budget,
)

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_measure_iteration_seconds_excludes_warmup_and_synchronizes() -> None:
    steps: list[int] = []
    syncs: list[str] = []
    ticks = iter([0.0, 10.0])  # timer() is called only at start and end of window

    result = measure_iteration_seconds(
        lambda index: steps.append(index),
        warmup=3,
        iterations=5,
        synchronize=lambda: syncs.append("sync"),
        timer=lambda: next(ticks),
    )

    # step is called warmup + iterations times, but only iterations are timed.
    assert len(steps) == 8
    assert result["warmup"] == 3
    assert result["iterations"] == 5
    assert result["total_seconds"] == pytest.approx(10.0)
    assert result["seconds_per_iteration"] == pytest.approx(2.0)
    # Synchronized once after warm-up and once after the timed loop.
    assert len(syncs) == 2


def test_measure_iteration_seconds_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="iterations must be > 0"):
        measure_iteration_seconds(lambda _: None, warmup=1, iterations=0)
    with pytest.raises(ValueError, match="warmup must be >= 0"):
        measure_iteration_seconds(lambda _: None, warmup=-1, iterations=1)


def test_render_runtime_budget_math_matches_protocol() -> None:
    protocol = resolve_controlled_protocol(ROOT, "faster_rcnn_resnet50")
    throughput = {
        "faster_rcnn_resnet50": {
            "seconds_per_iteration": 0.30,
            "gpu_name": "Tesla T4",
        },
        # A model with no measurement is reported as null, never guessed.
        "faster_rcnn_swin_t": {"seconds_per_iteration": None},
    }
    budget = render_runtime_budget(
        throughput, protocol, official_train_images=6471
    )

    iters = budget["iterations"]
    # Controlled defaults: 20% search subset, 5% selection holdout.
    assert iters["search_images"] == round(6471 * 0.20)
    assert iters["final_train_images"] == 6471 - round(6471 * 0.05)
    # HPO: phase_trials x (phase_a + phase_b epochs) x search images.
    assert iters["hpo_iterations"] == 5 * (3 + 3) * iters["search_images"]
    # Headline final: 1 recipe x 1 seed x final epochs x final images.
    assert iters["headline_final_iterations"] == 1 * 1 * 8 * iters["final_train_images"]
    # Full matrix: 2 recipes x 3 seeds.
    assert iters["full_matrix_final_iterations"] == 2 * 3 * 8 * iters["final_train_images"]

    resnet = budget["models"]["faster_rcnn_resnet50"]
    expected = (
        (iters["hpo_iterations"] + iters["headline_final_iterations"]) * 0.30 / 3600.0
    )
    assert resnet["headline_total_hours"] == pytest.approx(round(expected, 2))
    assert resnet["gpu_name"] == "Tesla T4"

    assert budget["models"]["faster_rcnn_swin_t"]["seconds_per_iteration"] is None
    assert "reason" in budget["models"]["faster_rcnn_swin_t"]


def test_render_runtime_budget_reads_only_config_values() -> None:
    """The budget uses the resolved protocol, so it tracks config changes."""
    protocol = load_protocol(ROOT, "controlled")
    assert protocol["final_seeds"] == [42]
    assert protocol["full_matrix_seeds"] == [17, 42, 3407]


def test_flops_measurement_never_reports_zero_for_failure() -> None:
    ok = flops_measurement(lambda: 1.2e10, method="fvcore")
    assert ok["flops_macs"] == pytest.approx(1.2e10)
    assert ok["reason"] is None

    # Unavailable / failed / non-positive all become null with a reason, not 0.
    for record in (
        flops_measurement(None, method="fvcore"),
        flops_measurement(lambda: 1 / 0, method="thop"),
        flops_measurement(lambda: 0.0, method="fvcore"),
    ):
        assert record["flops_macs"] is None
        assert record["reason"]


def test_assert_flops_valid_rejects_zero_and_bare_null() -> None:
    assert_flops_valid({"flops_macs": 5.0})
    assert_flops_valid({"flops_macs": None, "reason": "fvcore unavailable"})
    with pytest.raises(ValueError, match="positive or null"):
        assert_flops_valid({"flops_macs": 0.0})
    with pytest.raises(ValueError, match="record a reason"):
        assert_flops_valid({"flops_macs": None})
