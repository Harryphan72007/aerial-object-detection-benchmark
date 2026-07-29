"""Framework-neutral history logger and checkpoint selection state."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BestMetricState:
    best_map: float = float("-inf")
    best_aptiny: float = float("-inf")
    best_map_epoch: int = 0
    best_aptiny_epoch: int = 0

    def update(self, epoch: int, metrics: dict[str, float]) -> dict[str, bool]:
        flags = {"best_map": False, "best_aptiny": False}
        map_value = float(metrics.get("mAP", float("-inf")))
        if map_value > self.best_map:
            self.best_map = map_value
            self.best_map_epoch = epoch
            flags["best_map"] = True
        tiny_value = float(metrics.get("APtiny", float("-inf")))
        if tiny_value > self.best_aptiny:
            self.best_aptiny = tiny_value
            self.best_aptiny_epoch = epoch
            flags["best_aptiny"] = True
        return flags


class EpochHistoryWriter:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.rows: list[dict[str, Any]] = []
        history_file = self.run_dir / "metrics_history.csv"
        if history_file.exists():
            with history_file.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    parsed: dict[str, Any] = {}
                    for key, value in row.items():
                        if value is None or value == "":
                            parsed[key] = value
                            continue
                        try:
                            parsed[key] = float(value)
                            if parsed[key].is_integer():
                                parsed[key] = int(parsed[key])
                        except (ValueError, AttributeError):
                            parsed[key] = value
                    self.rows.append(parsed)

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(dict(row))
        with (self.run_dir / "epoch_metrics.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        fields = sorted({key for existing in self.rows for key in existing})
        with (self.run_dir / "metrics_history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(self.rows)


def save_training_curves(
    rows: list[dict[str, Any]], output_path: str | Path
) -> None:
    """Save available loss, mAP, APtiny, learning-rate, and VRAM curves."""
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    x_key = "epoch" if any("epoch" in row for row in rows) else "step"
    x_values = [row.get(x_key, index + 1) for index, row in enumerate(rows)]
    candidate_keys = [
        "training_loss",
        "loss",
        "mAP",
        "coco/bbox_mAP",
        "APtiny",
        "aerial_coco/APtiny",
        "learning_rate",
        "lr",
        "peak_allocated_gpu_memory",
    ]
    available = [
        key
        for key in candidate_keys
        if any(isinstance(row.get(key), (int, float)) for row in rows)
    ]
    figure, axes = plt.subplots(max(1, len(available)), 1, figsize=(9, 3 * max(1, len(available))))
    if len(available) == 1:
        axes = [axes]
    elif not available:
        axes = [axes]
    if not available:
        axes[0].text(0.5, 0.5, "No numeric training curves available", ha="center", va="center")
        axes[0].set_axis_off()
    else:
        for axis, key in zip(axes, available):
            values = [row.get(key, float("nan")) for row in rows]
            axis.plot(x_values, values, marker="o", markersize=2)
            axis.set_xlabel(x_key)
            axis.set_ylabel(key)
            axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
