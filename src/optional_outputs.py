"""Fail-open execution for explicitly non-scientific presentation operations."""
from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from src.utils.serialization import read_json, write_json

T = TypeVar("T")
WARNING_FILE = "optional_output_warnings.json"
WARNING_LOG = "logs/optional_output_warnings.log"


def load_optional_warnings(root: str | Path) -> list[dict[str, Any]]:
    path = Path(root) / WARNING_FILE
    if not path.is_file():
        return []
    try:
        value = read_json(path)
    except (OSError, ValueError):
        return []
    return list(value) if isinstance(value, list) else []


def _record_warning(root: Path, warning: dict[str, Any], traceback_text: str) -> None:
    log_path = root / WARNING_LOG
    warning["traceback_log"] = str(log_path)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"[{warning['timestamp']}] {warning['operation']}: "
                f"{warning['exception_type']}: {warning['message']}\n"
            )
            handle.write(traceback_text)
            if not traceback_text.endswith("\n"):
                handle.write("\n")
        warnings = load_optional_warnings(root)
        warnings.append(warning)
        write_json(root / WARNING_FILE, warnings)
    except Exception as recording_error:
        print(
            "WARNING: optional output failed and its warning artifact could not be "
            f"persisted: operation={warning['operation']!r}, "
            f"error={warning['exception_type']}: {warning['message']}, "
            f"recording_error={recording_error!r}",
            file=sys.stderr,
        )


def run_optional_output(
    operation: str,
    root: str | Path,
    action: Callable[[], T],
) -> tuple[T | None, dict[str, Any] | None]:
    """Run one bounded presentation operation and persist a structured warning."""
    try:
        return action(), None
    except Exception as error:
        warning: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": str(operation),
            "exception_type": type(error).__name__,
            "message": str(error),
            "scientific_artifacts_valid": True,
        }
        traceback_text = traceback.format_exc()
        _record_warning(Path(root), warning, traceback_text)
        print(
            f"WARNING: optional operation {operation!r} failed; scientific "
            f"artifacts remain valid: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return None, warning
