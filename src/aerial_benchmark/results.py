from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluation import validate_metrics


def append_result(path: str | Path, result: dict[str, Any]) -> None:
    required = ("model", "seed", "metrics", "provenance", "prediction_sha256")
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"Missing result fields: {', '.join(missing)}")
    validate_metrics(result["metrics"])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
