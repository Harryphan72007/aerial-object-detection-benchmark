from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect_provenance() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "unset"),
    }
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "accelerator": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
                ),
            }
        )
    except ImportError:
        payload.update({"torch": "not-installed", "cuda_available": False, "accelerator": "cpu"})
    return payload
