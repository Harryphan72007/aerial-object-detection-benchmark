"""Notebook-first orchestration for the controlled VisDrone benchmark."""

from src.workflows.contract import BENCHMARK_CONTRACT, PRIMARY_MODELS
from src.workflows.model_day import ModelDayOptions, Stage, inspect_model_day, run_model_day

__all__ = [
    "BENCHMARK_CONTRACT",
    "PRIMARY_MODELS",
    "ModelDayOptions",
    "Stage",
    "inspect_model_day",
    "run_model_day",
]
