#!/usr/bin/env python
"""Validate one isolated model package stack without constructing a model."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.notebook_utils import require_gpu, require_model_environment
from src.utils.environment import collect_environment
from src.utils.serialization import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment",
        required=True,
        choices=["rtdetr", "openmmlab"],
    )
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--json-output")
    args = parser.parse_args()
    require_model_environment(args.environment)
    if args.require_gpu:
        require_gpu(args.environment)
    environment = collect_environment()
    if args.json_output:
        write_json(Path(args.json_output), environment)
    print(f"{args.environment} environment: PASS")
    print(f"Python: {environment.get('python')}")
    print(f"PyTorch: {environment.get('pytorch_version')}")
    print(f"CUDA: {environment.get('cuda_version')}")
    print(f"GPU: {environment.get('gpu_name')}")


if __name__ == "__main__":
    main()
