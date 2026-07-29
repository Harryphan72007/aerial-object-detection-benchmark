#!/usr/bin/env python
"""Generate all final report artifacts from evaluation JSON files."""
from __future__ import annotations

import argparse
import json

from src.evaluation.report_generator import generate_report
from src.paths import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    args = parser.parse_args()
    paths = ProjectPaths.from_value(args.drive_root).create()
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(paths.evaluation.glob("*__metrics.json"))
    ]
    print(json.dumps(generate_report(rows, paths.reports), indent=2))


if __name__ == "__main__":
    main()
