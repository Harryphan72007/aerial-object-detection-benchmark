"""Model-family-first environment setup and validation."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from src.notebook_environment import (
    LOCAL_INSTALL_OPT_IN,
    default_model_runtime_root,
    detect_notebook_platform,
    parse_pinned_requirements,
    restart_required_packages,
)
from src.notebook_utils import in_hosted_notebook
from src.subprocess_utils import run_checked
from src.workflows.contract import require_primary_model
from src.workflows.pretrained_checkpoints import (
    ensure_pretrained_checkpoint,
    family_pretrained_spec,
)
from src.workflows.isolated_environment import (
    _clone_pinned as _clone_framework_pinned,
    _framework_checkout_path,
    _runtime_framework_root,
    provision_isolated_environment,
)

MMDET_REVISION = "44ebd17b145c2372c4b700bfb9cb20dbd28ab64a"
VMAMBA_REVISION = "2ed52ead062a51a64521ed3871d52914bf532876"
RTDETR_CHECKPOINT = "PekingU/rtdetr_v2_r50vd"


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


def _local_install_allowed() -> bool:
    """Installing family pins into the active interpreter needs an opt-in.

    Outside a hosted runtime the active interpreter is the developer's own
    environment; rewriting it from a notebook is destructive.
    """
    return os.environ.get(LOCAL_INSTALL_OPT_IN, "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _restart_state(repo: Path, requirements: str, changed: bool) -> dict[str, Any]:
    """Report whether this kernel now holds stale compiled modules."""
    if not changed:
        return {
            "restart_required": False,
            "restart_reason": "no package in this interpreter changed",
        }
    requirement_path = repo / requirements
    pins = (
        parse_pinned_requirements(requirement_path.read_text(encoding="utf-8"))
        if requirement_path.is_file()
        else {}
    )
    pending = restart_required_packages(pins)
    return {
        "restart_required": bool(pending),
        "restart_reason": (
            "already-imported compiled packages were replaced: "
            + ", ".join(change["package"] for change in pending)
            if pending
            else "no already-imported compiled package changed"
        ),
    }


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    run_checked(command, cwd=cwd, stage="model_environment_setup")


def _clone_pinned(url: str, destination: Path, revision: str) -> None:
    name = "MMDetection" if destination.parent.name == "mmdetection" else "VMamba"
    _clone_framework_pinned(
        {"name": name, "url": url, "revision": revision},
        destination,
        family="openmmlab" if name == "MMDetection" else "vmamba",
        python=Path(sys.executable),
    )


def _require_gpu() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in this model runtime") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required. Enable a GPU accelerator in Colab, Kaggle, "
            "or the local Jupyter host."
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

    if in_hosted_notebook():
        runtime_base = os.environ.get(
            "VISDRONE_MODEL_ENV_ROOT",
            str(default_model_runtime_root(detect_notebook_platform())),
        )
        runtime = provision_isolated_environment(
            model_id,
            repo,
            root,
            runtime_base=runtime_base,
        )
        result = {
            **runtime,
            # The isolated runtime installs into its own interpreter, so this
            # kernel's loaded modules are untouched by construction.
            "restart_required": False,
            "restart_reason": (
                "isolated runtime: packages are installed into "
                f"{runtime.get('python_executable')}, never into this kernel"
            ),
        }
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

    allow_local_install = install_missing and _local_install_allowed()
    if family == "rtdetr":
        missing = [
            name
            for name, expected in (("transformers", "4.52.4"), ("accelerate", "1.7.0"))
            if _version(name) != expected
        ]
        if missing and allow_local_install:
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
        elif missing:
            raise RuntimeError(
                "This is not a hosted runtime, so the RT-DETR pins are not "
                f"installed into {sys.executable}. Missing/mismatched: {missing}. "
                "Create a matching environment yourself (see "
                "requirements-rtdetr-colab.txt), or set "
                f"{LOCAL_INSTALL_OPT_IN}=1 to accept changing this interpreter."
            )
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
            **_restart_state(repo, "requirements-rtdetr-colab.txt", changed),
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
    if missing and not allow_local_install:
        raise RuntimeError(
            "This is not a hosted runtime, so the OpenMMLab pins are not "
            f"installed into {sys.executable}. Missing/mismatched: {missing}. "
            "Create a matching environment yourself (see "
            "requirements-openmmlab-py310-cu118.txt), or set "
            f"{LOCAL_INSTALL_OPT_IN}=1 to accept changing this interpreter."
        )
    if missing and allow_local_install:
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
    local_runtime_base = Path(
        os.environ.get(
            "VISDRONE_MODEL_ENV_ROOT",
            str(default_model_runtime_root("local")),
        )
    )
    framework_root = _runtime_framework_root(local_runtime_base, platform="local")
    mmdet_source = {
        "name": "MMDetection",
        "url": "https://github.com/open-mmlab/mmdetection.git",
        "revision": MMDET_REVISION,
    }
    mmdet_root = _framework_checkout_path(framework_root, mmdet_source)
    _clone_pinned(
        str(mmdet_source["url"]),
        mmdet_root,
        MMDET_REVISION,
    )
    os.environ["MMDET_ROOT"] = str(mmdet_root)

    result: dict[str, Any] = {
        "family": family,
        "MMDET_ROOT": str(mmdet_root),
        "packages_changed": changed,
        **_restart_state(repo, "requirements-openmmlab-py310-cu118.txt", changed),
        **_require_gpu(),
    }
    if family == "vmamba":
        vmamba_source = {
            "name": "VMamba",
            "url": "https://github.com/MzeroMiko/VMamba.git",
            "revision": VMAMBA_REVISION,
        }
        vmamba_root = _framework_checkout_path(framework_root, vmamba_source)
        _clone_pinned(
            str(vmamba_source["url"]),
            vmamba_root,
            VMAMBA_REVISION,
        )
        os.environ["VMAMBA_ROOT"] = str(vmamba_root)
        if (
            importlib.util.find_spec("selective_scan_cuda_oflex") is None
            and allow_local_install
        ):
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
        if importlib.util.find_spec("selective_scan_cuda_oflex") is None:
            raise RuntimeError("VMamba selective_scan_cuda_oflex could not be imported")
        spec = family_pretrained_spec(repo, "vmamba")
        if spec is None:
            raise RuntimeError("configs/runtime_environments.yaml declares no VMamba checkpoint")
        pretrained = root / "pretrained" / spec.filename
        # Verified against the pinned SHA-256; VMamba will not train from scratch.
        verification = ensure_pretrained_checkpoint(spec, pretrained)
        os.environ["VMAMBA_T_PRETRAINED"] = str(pretrained)
        result.update(
            {
                "VMAMBA_ROOT": str(vmamba_root),
                "VMAMBA_T_PRETRAINED": str(pretrained),
                "pretrained_checkpoint": verification,
                "selective_scan": "READY",
            }
        )
    return result
