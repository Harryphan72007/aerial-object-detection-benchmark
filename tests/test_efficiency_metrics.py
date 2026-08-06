"""PR-13: latency labelling and null-not-zero for missing measurements."""

from __future__ import annotations

import pytest

from src.evaluation.efficiency_metrics import (
    assert_latency_labeling,
    latency_report,
)


def test_latency_report_labels_batch_size() -> None:
    single = latency_report([10.0, 12.0, 11.0], batch_size=1, warmup=5)
    assert single["single_image_latency"] is True
    assert single["latency_label"] == "single-image"
    assert single["p50_latency_ms"] == pytest.approx(11.0)
    assert single["warmup"] == 5

    batched = latency_report([20.0, 22.0], batch_size=8)
    assert batched["single_image_latency"] is False
    assert batched["latency_label"] == "batch-8"


def test_latency_report_missing_measurement_is_null_not_zero() -> None:
    report = latency_report([], batch_size=1)
    for key in ("mean_latency_ms", "p90_latency_ms", "p99_latency_ms", "fps"):
        assert report[key] is None  # never 0, which would read as infinite speed


def test_assert_latency_labeling_rejects_mislabelled_batch() -> None:
    assert_latency_labeling(latency_report([5.0], batch_size=1))
    bad = latency_report([5.0], batch_size=8)
    bad["single_image_latency"] = True  # a caller mislabels batch latency
    with pytest.raises(ValueError, match="single-image"):
        assert_latency_labeling(bad)
    with pytest.raises(ValueError, match="batch_size >= 1"):
        assert_latency_labeling({"batch_size": 0})
