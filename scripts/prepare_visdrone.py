from __future__ import annotations

import argparse
import json

from aerial_benchmark.visdrone import convert_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(convert_split(args.split_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
