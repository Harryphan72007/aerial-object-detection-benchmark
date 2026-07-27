from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Aerial benchmark utilities")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    print(json.dumps(config, indent=2, sort_keys=True))
