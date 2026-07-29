#!/usr/bin/env python
"""Execute every notebook in guarded CPU smoke mode and write an evidence report."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", default=".notebook-smoke/notebook_smoke_results.json")
    parser.add_argument(
        "--notebook",
        choices=[
            "00_prepare_visdrone.ipynb",
            "01_run_model_day.ipynb",
            "02_publish_results.ipynb",
            "03_compare_all_models.ipynb",
        ],
        help="Execute one notebook so callers can enforce fresh harness processes.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    smoke_root = root / ".notebook-smoke"
    os.environ.update(
        {
            "SMOKE_TEST": "1",
            "SMOKE_TEST_SUBSET_SIZE": "8",
            "BENCHMARK_REPO_ROOT": str(root),
            "VISDRONE_DRIVE_ROOT": str(smoke_root / "dataset"),
        }
    )
    notebook_names = (
        [args.notebook]
        if args.notebook
        else [
            "00_prepare_visdrone.ipynb",
            "01_run_model_day.ipynb",
            "02_publish_results.ipynb",
            "03_compare_all_models.ipynb",
        ]
    )
    notebooks = [
        root / "notebooks" / name
        for name in notebook_names
    ]
    results: list[dict[str, object]] = []
    for path in notebooks:
        started = time.monotonic()
        notebook = nbformat.read(path, as_version=4)
        record: dict[str, object] = {
            "notebook": path.relative_to(root).as_posix(),
            "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
        }
        try:
            NotebookClient(
                notebook,
                timeout=args.timeout,
                kernel_name=args.kernel,
                resources={"metadata": {"path": str(root)}},
            ).execute()
            record["status"] = "passed"
        except CellExecutionError as error:
            record["status"] = "failed"
            record["error"] = str(error)[-4000:]
        except Exception as error:  # Harness failures still need exact evidence.
            record["status"] = "harness_error"
            record["error"] = f"{type(error).__name__}: {error}"
        record["duration_seconds"] = round(time.monotonic() - started, 3)
        results.append(record)
        print(record["status"], record["notebook"], record["duration_seconds"])
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "CPU synthetic VisDrone smoke test; training/tuning/publishing guarded",
        "kernel": args.kernel,
        "results": results,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failures = [result for result in results if result["status"] != "passed"]
    if failures:
        raise SystemExit(f"{len(failures)} notebook smoke test(s) failed; see {output}")


if __name__ == "__main__":
    main()
