#!/usr/bin/env python
"""Run canonical read-only repository and environment diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.diagnostics import inspect_environment, inspect_repository


def build_report(repository_root: str | Path) -> dict:
    return {
        "schema_version": 1,
        "repository": inspect_repository(repository_root).to_dict(),
        "environment": inspect_environment(),
    }


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rendered = json.dumps(build_report(args.repo_root), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.expanduser().resolve().write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
