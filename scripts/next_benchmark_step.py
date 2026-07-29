#!/usr/bin/env python
"""Print exactly one read-only recommended benchmark action."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.benchmark_status import (
    PRIMARY_MODELS,
    discover_model_status,
    recommended_next_step,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--repo-root", default=str(REPOSITORY_ROOT))
    parser.add_argument("--model-id", required=True, choices=PRIMARY_MODELS)
    args = parser.parse_args()
    status = discover_model_status(
        args.drive_root, args.model_id, args.repo_root
    )
    print(recommended_next_step(status, args.drive_root))


if __name__ == "__main__":
    main()
