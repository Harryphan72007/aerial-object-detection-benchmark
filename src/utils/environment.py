"""Hardware and software environment inspection."""
from __future__ import annotations
import os, platform, subprocess, sys
from typing import Any
import psutil
from src.reproducibility import framework_versions

def collect_environment() -> dict[str, Any]:
    data: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_bytes": psutil.virtual_memory().total,
        "cwd": os.getcwd(),
    }
    data.update(framework_versions())
    try:
        data["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip()
    except Exception:
        data["nvidia_smi"] = "unavailable"
    return data
