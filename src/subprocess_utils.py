"""Safe command construction for repository-owned Python entry points."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping

MODEL_PYTHON_ENV = "VISDRONE_MODEL_PYTHON"
HEADLESS_BACKEND = "Agg"
DETERMINISTIC_CUBLAS_WORKSPACE = ":4096:8"

NOTEBOOK_ONLY_ENVIRONMENT_VARIABLES = (
    "IPYTHONDIR",
    "JUPYTER_PATH",
    "PYTHONSTARTUP",
    "DISPLAY",
)

SINGLE_PROCESS_DISTRIBUTED_VARIABLES = (
    "WORLD_SIZE",
    "LOCAL_RANK",
    "RANK",
)

RECORDED_SUBPROCESS_VARIABLES = (
    "MPLBACKEND",
    "MPLCONFIGDIR",
    "PYTHONPATH",
    "IPYTHONDIR",
    "JUPYTER_PATH",
    "DISPLAY",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
    "TORCH_HOME",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "RANK",
)


def model_python_executable() -> str:
    return os.environ.get(MODEL_PYTHON_ENV, sys.executable)


def python_module_command(module: str, *arguments: object) -> list[str]:
    """Return a Python module command and reject filepath-style entry points."""
    if not module or module.endswith(".py") or "/" in module or "\\" in module:
        raise ValueError(
            f"Repository Python entry points must use dotted module names, got {module!r}"
        )
    executable = model_python_executable()
    return [executable, "-m", module, *(str(value) for value in arguments)]


def build_model_subprocess_environment(
    parent: Mapping[str, str] | None = None,
    *,
    matplotlib_config_dir: str | Path | None = None,
) -> dict[str, str]:
    """Build a single-process model environment without notebook-only leakage."""
    environment = dict(os.environ if parent is None else parent)
    for variable in NOTEBOOK_ONLY_ENVIRONMENT_VARIABLES:
        environment.pop(variable, None)
    # The repository launches one backend process directly, never through torchrun.
    for variable in SINGLE_PROCESS_DISTRIBUTED_VARIABLES:
        environment.pop(variable, None)
    # The isolated runtime is editable-installed and launched from the repository;
    # inheriting the notebook kernel's PYTHONPATH can shadow its pinned packages.
    environment.pop("PYTHONPATH", None)
    environment["MPLBACKEND"] = HEADLESS_BACKEND
    environment["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE
    if matplotlib_config_dir is not None:
        config_dir = Path(matplotlib_config_dir).resolve()
        config_dir.mkdir(parents=True, exist_ok=True)
        environment["MPLCONFIGDIR"] = str(config_dir)
    return environment


def configure_headless_matplotlib() -> None:
    """Force a noninteractive backend before pyplot is imported."""
    os.environ["MPLBACKEND"] = HEADLESS_BACKEND
    import matplotlib

    matplotlib.use(HEADLESS_BACKEND, force=True)


def subprocess_environment_record(
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return the non-secret environment policy fields used by model processes."""
    values = os.environ if environment is None else environment
    return {
        "policy": "single_process_headless_v1",
        "variables": {
            variable: values.get(variable)
            for variable in RECORDED_SUBPROCESS_VARIABLES
        },
    }
