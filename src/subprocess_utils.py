"""Safe command construction for repository-owned Python entry points."""
from __future__ import annotations

import os
import sys

MODEL_PYTHON_ENV = "VISDRONE_MODEL_PYTHON"


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
