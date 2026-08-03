"""Deterministic VMamba registration and selective-scan discovery."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

APPROVED_VMAMBA_REVISION = "2ed52ead062a51a64521ed3871d52914bf532876"


def verify_vmamba_revision(
    vmamba_root: str | Path,
    expected: str = APPROVED_VMAMBA_REVISION,
    *,
    revision_reader: Callable[[Path], str] | None = None,
) -> str:
    """Require the exact approved upstream revision before construction."""

    root = Path(vmamba_root).expanduser().resolve()
    if revision_reader is None:
        def revision_reader(path: Path) -> str:
            return subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
    observed = revision_reader(root)
    if observed != expected:
        raise RuntimeError(
            f"VMamba revision mismatch: observed {observed}, expected {expected}"
        )
    return observed


def register_vmamba_detection(vmamba_root: str | Path) -> ModuleType:
    """Load ``detection/model.py`` under its required top-level module name."""

    root = Path(vmamba_root).expanduser().resolve()
    detection_root = root / "detection"
    module_path = detection_root / "model.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    existing = sys.modules.get("model")
    if existing is not None:
        existing_path = Path(str(getattr(existing, "__file__", ""))).resolve()
        if existing_path != module_path:
            raise RuntimeError(
                f"top-level module 'model' is already registered from {existing_path}"
            )
        return existing
    detection_value = str(detection_root)
    if detection_value not in sys.path:
        sys.path.insert(0, detection_value)
    spec = importlib.util.spec_from_file_location("model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["model"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("model", None)
        raise
    return module


def detect_selective_scan_backend(
    finder: Callable[[str], Any] = importlib.util.find_spec,
) -> dict[str, Any]:
    """Report the first available backend, preferring the approved CUDA extension."""

    candidates = (
        ("selective_scan_cuda_oflex", True, "optimized_cuda_oflex"),
        ("selective_scan_cuda_core", False, "legacy_cuda_core"),
        ("selective_scan_cuda", False, "mamba_ssm_cuda"),
        ("mamba_ssm.ops.selective_scan_interface", False, "python_or_package_fallback"),
    )
    for module, approved, kind in candidates:
        try:
            available = finder(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if available:
            return {
                "module": module,
                "kind": kind,
                "optimized": approved,
                "approved": approved,
            }
    return {
        "module": None,
        "kind": "unavailable",
        "optimized": False,
        "approved": False,
    }
