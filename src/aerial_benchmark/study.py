from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


def load_objective(path: str) -> Callable[[Any], float]:
    module_name, separator, function_name = path.partition(":")
    if not separator:
        raise ValueError("Objective must use 'module:function' syntax")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"{path} is not callable")
    return function


def run_study(
    *,
    name: str,
    storage: str,
    objective_path: str,
    trials: int,
    sampler_seed: int = 2026,
) -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Install the 'optimize' extra to run Optuna studies") from exc
    if trials < 1:
        raise ValueError("trials must be positive")
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        load_if_exists=True,
    )
    study.optimize(load_objective(objective_path), n_trials=trials, gc_after_trial=True)
    return study
