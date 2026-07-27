from __future__ import annotations

import argparse

from aerial_benchmark.evaluation import evaluate_coco
from aerial_benchmark.provenance import collect_provenance
from aerial_benchmark.results import append_result
from aerial_benchmark.visdrone import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metrics = evaluate_coco(args.ground_truth, args.predictions)
    append_result(
        args.output,
        {
            "model": args.model,
            "seed": args.seed,
            "metrics": metrics,
            "provenance": collect_provenance(),
            "prediction_sha256": file_sha256(args.predictions),
            "ground_truth_sha256": file_sha256(args.ground_truth),
        },
    )


if __name__ == "__main__":
    main()
