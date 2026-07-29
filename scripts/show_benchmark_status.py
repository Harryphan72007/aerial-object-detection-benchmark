#!/usr/bin/env python
"""Print a read-only progress table for the LR-controlled benchmark."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.benchmark_status import (
    PRIMARY_MODELS,
    discover_all_statuses,
    format_status_table,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--repo-root", default=str(REPOSITORY_ROOT))
    parser.add_argument("--model-id", choices=PRIMARY_MODELS)
    args = parser.parse_args()
    models = [args.model_id] if args.model_id else list(PRIMARY_MODELS)
    print(
        format_status_table(
            discover_all_statuses(args.drive_root, models, args.repo_root)
        )
    )


if __name__ == "__main__":
    main()
