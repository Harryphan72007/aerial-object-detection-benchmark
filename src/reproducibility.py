"""Determinism and environment capture helpers."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch when available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def worker_seed(worker_id: int) -> None:
    """Seed a data-loader worker from PyTorch's initial seed."""
    try:
        import torch
        seed = torch.initial_seed() % 2**32
    except ImportError:
        seed = worker_id
    random.seed(seed); np.random.seed(seed)


def git_commit(repo_root: str | Path = ".") -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def framework_versions() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        import torch
        result.update({
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        })
    except ImportError:
        result.update({"pytorch_version": "not installed", "cuda_version": None, "cuda_available": False, "gpu_name": "CPU"})
    for package in ("mmengine", "mmdet", "transformers", "optuna"):
        try:
            module = __import__(package)
            result[f"{package}_version"] = getattr(module, "__version__", "unknown")
        except ImportError:
            result[f"{package}_version"] = "not installed"
    return result
