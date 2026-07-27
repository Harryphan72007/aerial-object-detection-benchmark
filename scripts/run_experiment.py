from __future__ import annotations

import argparse
import json

from aerial_benchmark.config import load_config, validate_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    if not args.dry_run:
        adapter = config["model"].get("adapter")
        if adapter in (None, "TBD"):
            raise SystemExit(
                "No model adapter is configured yet. "
                "Use --dry-run or implement the adapter contract."
            )
    print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
