"""Model-family-first environment setup and validation."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from src.notebook_utils import in_colab
from src.subprocess_utils import run_checked
from src.workflows.contract import require_primary_model
from src.workflows.isolated_environment import provision_isolated_environment

MMDET_REVISION = "44ebd17b145c2372c4b700bfb9cb20dbd28ab64a"
VMAMBA_REVISION = "2ed52ead062a51a64521ed3871d52914bf532876"
RTDETR_CHECKPOINT = "PekingU/rtdetr_v2_r101vd"


def model_family(model_id: str) -> str:
    require_primary_model(model_id)
    if model_id == "rtdetrv2_l":
        return "rtdetr"
    if model_id == "faster_rcnn_vmamba_t":
        return "vmamba"
    return "openmmlab"


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    run_checked(command, cwd=cwd, stage="model_environment_setup")


def _clone_pinned(url: str, destination: Path, revision: str) -> None:
    if not destination.exists():
        _run(["git", "clone", url, str(destination)])
    if not (destination / ".git").is_dir():
        raise RuntimeError(f"Refusing to use non-Git upstream directory: {destination}")
    _run(["git", "-C", str(destination), "fetch", "--tags", "origin", revision])
    _run(["git", "-C", str(destination), "checkout", "--detach", revision])


def _require_gpu() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in this model runtime") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required. In Colab choose Runtime > Change runtime type > GPU."
        )
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }


def ensure_model_environment(
    model_id: str,
    repo_root: str | Path,
    drive_root: str | Path,
    *,
    install_missing: bool = True,
) -> dict[str, Any]:
    """Install only the selected family and fail before unsafe binary builds."""
    family = model_family(model_id)
    repo = Path(repo_root).resolve()
    root = Path(drive_root).expanduser().resolve()
    changed = False

    if in_colab():
        runtime = provision_isolated_environment(model_id, repo, root)
        result = {**runtime, "restart_required": False}
        if family == "rtdetr":
            result["checkpoint"] = RTDETR_CHECKPOINT
        if os.environ.get("MMDET_ROOT"):
            result["MMDET_ROOT"] = os.environ["MMDET_ROOT"]
        if family == "vmamba":
            result.update(
                {
                    "VMAMBA_ROOT": os.environ["VMAMBA_ROOT"],
                    "VMAMBA_T_PRETRAINED": os.environ["VMAMBA_T_PRETRAINED"],
                    "selective_scan": "READY",
                }
            )
        return result

    if family == "rtdetr":
        missing = [
            name
            for name, expected in (("transformers", "4.52.4"), ("accelerate", "1.7.0"))
            if _version(name) != expected
        ]
        if missing and install_missing:
            _run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(repo / "requirements-rtdetr-colab.txt"),
                ]
            )
            changed = True
        hardware = _require_gpu()
        try:
            from transformers import AutoConfig

            resolved = AutoConfig.from_pretrained(RTDETR_CHECKPOINT)
        except Exception as exc:
            raise RuntimeError(
                f"Could not resolve the canonical checkpoint {RTDETR_CHECKPOINT}"
            ) from exc
        return {
            "family": family,
            "checkpoint": getattr(resolved, "_name_or_path", RTDETR_CHECKPOINT),
            "packages_changed": changed,
            "restart_required": changed and in_colab(),
            **hardware,
        }

    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "OpenMMLab models require Python 3.10. Use a Colab local/custom runtime "
            f"with Python 3.10; active runtime is {sys.version.split()[0]}. "
            "No MMCV source build was attempted."
        )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Install PyTorch 2.1.0+cu118 and torchvision 0.16.0+cu118 first. "
            "No MMCV source build was attempted."
        ) from exc
    if not str(torch.__version__).startswith("2.1.0") or str(torch.version.cuda) != "11.8":
        raise RuntimeError(
            "OpenMMLab models require PyTorch 2.1.0+cu118 and CUDA 11.8; "
            f"active torch={torch.__version__}, CUDA={torch.version.cuda}. "
            "No MMCV source build was attempted."
        )
    torchvision_version = _version("torchvision")
    if torchvision_version is None or not torchvision_version.startswith("0.16.0"):
        raise RuntimeError(
            "OpenMMLab models require torchvision 0.16.0+cu118; "
            f"active torchvision={torchvision_version or 'not installed'}. "
            "Install the documented Torch/torchvision pair before MMCV."
        )
    expected = {
        "numpy": "1.26.4",
        "mmcv": "2.1.0",
        "mmengine": "0.10.7",
        "mmdet": "3.3.0",
    }
    missing = [name for name, value in expected.items() if _version(name) != value]
    if missing and install_missing:
        if "mmcv" in missing:
            _run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "mmcv==2.1.0",
                    "--find-links",
                    "https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html",
                    "--only-binary",
                    "mmcv",
                ]
            )
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(repo / "requirements-openmmlab-py310-cu118.txt"),
            ]
        )
        changed = True
    unresolved = [name for name, value in expected.items() if _version(name) != value]
    if unresolved:
        raise RuntimeError(
            "OpenMMLab package mismatch after setup: " + ", ".join(unresolved)
        )
    mmdet_root = root / "frameworks" / "mmdetection"
    _clone_pinned(
        "https://github.com/open-mmlab/mmdetection.git",
        mmdet_root,
        MMDET_REVISION,
    )
    os.environ["MMDET_ROOT"] = str(mmdet_root)

    result: dict[str, Any] = {
        "family": family,
        "MMDET_ROOT": str(mmdet_root),
        "packages_changed": changed,
        "restart_required": changed and in_colab(),
        **_require_gpu(),
    }
    if family == "vmamba":
        vmamba_root = root / "frameworks" / "VMamba"
        _clone_pinned(
            "https://github.com/MzeroMiko/VMamba.git",
            vmamba_root,
            VMAMBA_REVISION,
        )
        os.environ["VMAMBA_ROOT"] = str(vmamba_root)
        if importlib.util.find_spec("selective_scan_cuda") is None and install_missing:
            _run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    str(vmamba_root / "kernels" / "selective_scan"),
                    "--no-build-isolation",
                ]
            )
        if importlib.util.find_spec("selective_scan_cuda") is None:
            raise RuntimeError("VMamba selective_scan_cuda could not be imported")
        pretrained = root / "pretrained" / "vmamba_tiny_e292.pth"
        if not pretrained.is_file() or pretrained.stat().st_size == 0:
            raise FileNotFoundError(
                "VMamba will not train from scratch. Place the canonical pretrained "
                f"checkpoint at {pretrained}"
            )
        os.environ["VMAMBA_T_PRETRAINED"] = str(pretrained)
        result.update(
            {
                "VMAMBA_ROOT": str(vmamba_root),
                "VMAMBA_T_PRETRAINED": str(pretrained),
                "selective_scan": "READY",
            }
        )
    return result
