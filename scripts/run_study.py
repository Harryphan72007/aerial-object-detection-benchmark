from __future__ import annotations

import argparse

from aerial_benchmark.study import run_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--sampler-seed", type=int, default=2026)
    args = parser.parse_args()
    study = run_study(
        name=args.study,
        storage=args.storage,
        objective_path=args.objective,
        trials=args.trials,
        sampler_seed=args.sampler_seed,
    )
    print(f"study={study.study_name} trials={len(study.trials)} best_value={study.best_value}")


if __name__ == "__main__":
    main()
