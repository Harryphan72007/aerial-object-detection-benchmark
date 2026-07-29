"""Framework-neutral learning-rate range-test schedules and artifacts."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.training.lr_search import exponential_moving_average
from src.utils.serialization import write_json


def exponential_lr_schedule(
    baseline_learning_rate: float,
    optimizer_steps: int = 300,
    start_lr_multiplier: float = 0.01,
    end_lr_multiplier: float = 20.0,
) -> list[float]:
    if baseline_learning_rate <= 0 or optimizer_steps < 2:
        raise ValueError("baseline LR must be positive and optimizer_steps >= 2")
    if not 0 < start_lr_multiplier < end_lr_multiplier:
        raise ValueError("LR multipliers must be positive and increasing")
    start = baseline_learning_rate * start_lr_multiplier
    end = baseline_learning_rate * end_lr_multiplier
    ratio = (end / start) ** (1 / (optimizer_steps - 1))
    return [start * ratio**step for step in range(optimizer_steps)]


def should_stop_range_test(
    losses: Iterable[float],
    *,
    divergence_multiplier: float = 4.0,
) -> tuple[bool, str | None]:
    values = [float(value) for value in losses]
    if not values:
        return False, None
    if not math.isfinite(values[-1]):
        return True, "non_finite_loss"
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) >= 10 and finite[-1] > divergence_multiplier * min(finite):
        return True, "strong_divergence"
    return False, None


def suggest_safe_lr_interval(
    history: list[Mapping[str, Any]],
    baseline_learning_rate: float,
) -> tuple[float, float] | None:
    """Return a conservative baseline-centered interval from finite observations."""
    finite = [
        row
        for row in history
        if math.isfinite(float(row.get("smoothed_loss", float("nan"))))
        and float(row.get("learning_rate", 0)) > 0
    ]
    if len(finite) < 10:
        return None
    best = min(finite, key=lambda row: float(row["smoothed_loss"]))
    maximum_observed_safe = float(best["learning_rate"])
    if maximum_observed_safe <= baseline_learning_rate:
        return None
    symmetric_factor = min(
        8.0,
        maximum_observed_safe / baseline_learning_rate,
    )
    if symmetric_factor <= 1.0:
        return None
    return (
        baseline_learning_rate / symmetric_factor,
        baseline_learning_rate * symmetric_factor,
    )


def save_lr_range_artifacts(
    output_dir: str | Path,
    history: list[dict[str, Any]],
    *,
    baseline_learning_rate: float,
    stopped_reason: str | None = None,
    smoothing_beta: float = 0.98,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    losses = [float(row["raw_loss"]) for row in history]
    smoothed = exponential_moving_average(losses, beta=smoothing_beta)
    normalized: list[dict[str, Any]] = []
    for row, smooth in zip(history, smoothed, strict=True):
        normalized.append({**row, "smoothed_loss": smooth})
    fields = [
        "optimizer_step",
        "learning_rate",
        "raw_loss",
        "smoothed_loss",
        "gradient_norm",
    ]
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized)
    safe_interval = suggest_safe_lr_interval(normalized, baseline_learning_rate)
    summary = {
        "baseline_learning_rate": baseline_learning_rate,
        "optimizer_steps_completed": len(normalized),
        "stopped_reason": stopped_reason,
        "safe_lower_learning_rate": safe_interval[0] if safe_interval else None,
        "safe_upper_learning_rate": safe_interval[1] if safe_interval else None,
        "valid_safe_interval": safe_interval is not None,
        "model_state_promotable": False,
    }
    write_json(output / "summary.json", summary)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(
            [float(row["learning_rate"]) for row in normalized],
            [float(row["smoothed_loss"]) for row in normalized],
            label="smoothed loss",
        )
        axis.scatter(
            [float(row["learning_rate"]) for row in normalized],
            [float(row["raw_loss"]) for row in normalized],
            s=5,
            alpha=0.2,
            label="raw loss",
        )
        axis.set_xscale("log")
        axis.set_xlabel("learning rate")
        axis.set_ylabel("training loss")
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "loss_vs_lr.png", dpi=180)
        plt.close(figure)
    except ImportError:
        (output / "loss_vs_lr.png.unavailable.txt").write_text(
            "matplotlib is not installed\n", encoding="utf-8"
        )
    return summary
