#!/usr/bin/env python
"""Validate checked-in configuration and schema documents for CI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.config.benchmark_tracks import load_track_config
from src.config.experiment import load_experiment_config


def validate_yaml_file(path: Path) -> None:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"{path}: invalid YAML: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")


def validate_json_file(path: Path) -> None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{path}: top-level JSON value must be an object or array")


def validate_repository_contract(repository_root: Path) -> list[str]:
    root = repository_root.resolve()
    errors: list[str] = []
    for path in sorted((root / "configs").rglob("*.yaml")):
        try:
            validate_yaml_file(path)
        except ValueError as error:
            errors.append(str(error))
    for path in sorted((root / "schemas").rglob("*.json")):
        try:
            validate_json_file(path)
        except ValueError as error:
            errors.append(str(error))
    for mode in ("legacy", "smoke"):
        for path in sorted((root / "configs" / mode).glob("*.yaml")):
            try:
                load_experiment_config(path)
            except (OSError, ValueError) as error:
                errors.append(f"{path}: experiment contract: {error}")
    for track in ("controlled", "performance"):
        try:
            load_track_config(root, track)
        except (OSError, ValueError) as error:
            errors.append(f"configs/{track}/benchmark.yaml: track contract: {error}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    errors = validate_repository_contract(args.repo_root)
    if errors:
        raise SystemExit("Configuration/schema validation failed:\n" + "\n".join(errors))
    print("Configuration and schema contracts passed.")


if __name__ == "__main__":
    main()
