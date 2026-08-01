"""Isolated comparison tables for v2 metric artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.config.benchmark_tracks import require_comparison_track
from src.utils.serialization import write_json

TABLES = ("controlled", "performance", "full", "sliced", "ensemble")
INFERENCE_MODES = frozenset({"full", "sliced", "ensemble"})


def _row(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if artifact.get("schema_version") != 2:
        raise ValueError("comparison requires metric schema v2")
    track = str(artifact.get("benchmark_track"))
    mode = str(artifact.get("inference_mode"))
    if track not in {"controlled", "performance"} or mode not in INFERENCE_MODES:
        raise ValueError("invalid comparison track or inference mode")
    identity = dict(artifact["identity"])
    metrics = dict(artifact["metrics"])
    return {
        "run_id": identity.get("run_id"),
        "model_id": identity.get("model_id"),
        "benchmark_track": track,
        "inference_mode": mode,
        "weight_variant": artifact.get("weight_variant", "raw"),
        "metrics": metrics,
    }


def build_comparison_tables(
    artifacts: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    rows = [_row(artifact) for artifact in artifacts]
    tables = {
        "controlled": [row for row in rows if row["benchmark_track"] == "controlled"],
        "performance": [row for row in rows if row["benchmark_track"] == "performance"],
        "full": [row for row in rows if row["inference_mode"] == "full"],
        "sliced": [row for row in rows if row["inference_mode"] == "sliced"],
        "ensemble": [row for row in rows if row["inference_mode"] == "ensemble"],
    }
    for row in tables["controlled"]:
        require_comparison_track(row, "controlled")
    for row in tables["performance"]:
        require_comparison_track(row, "performance")
    return tables


def write_comparison_tables(
    output_root: str | Path, tables: Mapping[str, list[dict[str, Any]]]
) -> dict[str, Path]:
    root = Path(output_root)
    outputs: dict[str, Path] = {}
    for name in TABLES:
        rows = list(tables.get(name, []))
        table_root = root / name
        json_path = table_root / "comparison.json"
        write_json(json_path, {"table": name, "rows": rows}, atomic=True)
        csv_path = table_root / "comparison.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["run_id", "model_id", "benchmark_track", "inference_mode", "weight_variant"]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        outputs[f"{name}_json"] = json_path
        outputs[f"{name}_csv"] = csv_path
    return outputs
