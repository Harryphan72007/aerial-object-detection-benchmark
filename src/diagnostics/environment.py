"""Read-only package and hardware inspection without importing model stacks."""
from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Iterable

import psutil


DEFAULT_PACKAGES = (
    "numpy",
    "pandas",
    "Pillow",
    "PyYAML",
    "pycocotools",
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "mmcv",
    "mmengine",
    "mmdet",
    "timm",
    "optuna",
)


def _package_versions(packages: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _gpu_snapshot() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "nvidia_smi": None, "devices": []}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "nvidia_smi": executable,
            "devices": [],
            "error": type(exc).__name__,
        }
    devices = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) == 3:
                devices.append(
                    {
                        "name": values[0],
                        "memory_total_mb": values[1],
                        "driver_version": values[2],
                    }
                )
    return {
        "available": bool(devices),
        "nvidia_smi": executable,
        "devices": devices,
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
    }


def inspect_environment(
    packages: Iterable[str] = DEFAULT_PACKAGES,
) -> dict[str, Any]:
    """Return metadata only; detector frameworks are never imported."""
    return {
        "schema_version": 1,
        "read_only": True,
        "model_modules_imported": False,
        "model_construction_performed": False,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cwd": os.getcwd(),
            "cpu_count_logical": psutil.cpu_count(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_total_bytes": psutil.virtual_memory().total,
        },
        "packages": _package_versions(packages),
        "gpu": _gpu_snapshot(),
    }
