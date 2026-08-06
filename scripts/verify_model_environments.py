#!/usr/bin/env python
"""Validate one complete isolated model stack and emit a structured probe."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.utils.environment import collect_environment
from src.utils.serialization import write_json
from src.workflows.isolated_environment import resolved_runtime_spec

DEFAULT_MODELS = {
    "openmmlab": "faster_rcnn_resnet50",
    "vmamba": "faster_rcnn_vmamba_t",
    "rtdetr": "rtdetrv2_l",
}
RECORDED_VARIABLES = (
    "PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "TORCH_HOME",
    "HF_HOME",
    "XDG_CACHE_HOME",
    "MPLBACKEND",
    "CUBLAS_WORKSPACE_CONFIG",
    "VISDRONE_MODEL_PYTHON",
    "VISDRONE_RUNTIME_MANIFEST",
    "MMDET_ROOT",
    "VMAMBA_ROOT",
    "VMAMBA_T_PRETRAINED",
)


class ProbeFailure(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(message)


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as error:
        raise ProbeFailure("package_versions", f"{package} is not installed") from error


def _require_exact(package: str, expected: str, versions: dict[str, str]) -> None:
    observed = _version(package)
    versions[package] = observed
    if observed != expected:
        raise ProbeFailure(
            "package_versions",
            f"{package} version mismatch: expected {expected}, observed {observed}",
        )


def _require_import(module: str, stage: str) -> Any:
    try:
        return importlib.import_module(module)
    except Exception as error:
        raise ProbeFailure(
            stage,
            f"could not import {module}: {type(error).__name__}: {error}",
        ) from error


def _source(spec: dict[str, Any], name: str) -> dict[str, Any]:
    for value in spec.get("sources", []):
        if value.get("name") == name:
            return dict(value)
    raise ProbeFailure("source_contract", f"missing configured source {name!r}")


def _git_revision(path: Path, expected: str, stage: str) -> str:
    if not (path / ".git").is_dir():
        raise ProbeFailure(stage, f"pinned source checkout is missing: {path}")
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ProbeFailure(stage, result.stderr.strip() or "git revision probe failed")
    observed = result.stdout.strip()
    if observed != expected:
        raise ProbeFailure(
            stage,
            f"source revision mismatch at {path}: expected {expected}, observed {observed}",
        )
    if (path / ".git" / "index.lock").exists():
        raise ProbeFailure(stage, f"source checkout has an active Git index lock: {path}")
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode:
        raise ProbeFailure(stage, status.stderr.strip() or "git status probe failed")
    if status.stdout.strip():
        raise ProbeFailure(
            stage,
            f"source checkout is dirty at {path}: {status.stdout.strip()}",
        )
    return observed


def _expected_cuda(torch_version: str) -> str | None:
    if "+cu" not in torch_version:
        return None
    digits = torch_version.rsplit("+cu", 1)[1]
    if len(digits) != 3:
        return None
    return f"{digits[:2]}.{digits[2]}"


def _base_probe(
    spec: dict[str, Any],
    *,
    require_gpu: bool,
) -> tuple[dict[str, str], Any]:
    expected_python = str(spec["python"])
    observed_python = platform.python_version()
    if observed_python != expected_python:
        raise ProbeFailure(
            "python_version",
            f"Python mismatch: expected {expected_python}, observed {observed_python}",
        )
    versions: dict[str, str] = {"python": observed_python}
    _require_exact("torch", str(spec["torch"]), versions)
    _require_exact("torchvision", str(spec["torchvision"]), versions)
    torch = _require_import("torch", "torch_import")
    _require_import("torchvision", "torchvision_import")
    expected_cuda = _expected_cuda(str(spec["torch"]))
    observed_cuda = str(torch.version.cuda) if torch.version.cuda else None
    if expected_cuda and observed_cuda != expected_cuda:
        raise ProbeFailure(
            "torch_cuda_build",
            f"CUDA build mismatch: expected {expected_cuda}, observed {observed_cuda}",
        )
    if require_gpu and not torch.cuda.is_available():
        raise ProbeFailure(
            "cuda_visibility",
            "torch.cuda.is_available() is False in the isolated child runtime",
        )
    if require_gpu:
        try:
            versions["gpu"] = str(torch.cuda.get_device_name(0))
        except Exception as error:
            raise ProbeFailure("cuda_device", f"GPU name lookup failed: {error}") from error
    versions["torch_cuda"] = observed_cuda or "none"
    return versions, torch


def _openmmlab_probe(
    spec: dict[str, Any],
    args: argparse.Namespace,
    versions: dict[str, str],
) -> dict[str, Any]:
    for package, expected in (
        ("psutil", "7.0.0"),
        ("numpy", "1.26.4"),
        ("mmcv", str(spec["mmcv"]["version"])),
        ("mmengine", "0.10.7"),
        ("mmdet", "3.3.0"),
    ):
        _require_exact(package, expected, versions)
    for module in ("psutil", "numpy", "mmcv", "mmengine", "mmdet"):
        _require_import(module, f"{module}_import")
    operations = _require_import("mmcv.ops", "mmcv_compiled_operation")
    if not hasattr(operations, "nms"):
        raise ProbeFailure(
            "mmcv_compiled_operation",
            "MMCV compiled module imported but mmcv.ops.nms is unavailable",
        )
    mmdet_root = Path(args.mmdet_root or os.environ.get("MMDET_ROOT", ""))
    expected = str(_source(spec, "MMDetection")["revision"])
    revision = _git_revision(mmdet_root, expected, "mmdetection_revision")
    return {
        "mmdetection_root": str(mmdet_root.resolve()),
        "mmdetection_revision": revision,
        "mmcv_compiled_operation": "mmcv.ops.nms",
    }


def _vmamba_probe(
    spec: dict[str, Any],
    args: argparse.Namespace,
    versions: dict[str, str],
) -> dict[str, Any]:
    for package, expected in (
        ("mmsegmentation", "1.2.2"),
        ("fvcore", "0.1.5.post20221221"),
        ("setuptools", "69.5.1"),
        ("wheel", "0.43.0"),
    ):
        _require_exact(package, expected, versions)
    for module in ("mmseg", "fvcore", "setuptools", "wheel"):
        _require_import(module, f"{module}_import")
    vmamba_root = Path(args.vmamba_root or os.environ.get("VMAMBA_ROOT", ""))
    expected = str(_source(spec, "VMamba")["revision"])
    revision = _git_revision(vmamba_root, expected, "vmamba_revision")
    config = Path(args.vmamba_config or "")
    if not config.is_file():
        raise ProbeFailure(
            "vmamba_detector_configuration",
            f"VMamba detector configuration is missing: {config}",
        )
    try:
        from src.models.vmamba_frcnn.importer import register_vmamba_detection

        registration = register_vmamba_detection(vmamba_root)
    except Exception as error:
        raise ProbeFailure(
            "vmamba_registration",
            f"VMamba top-level registration import failed: {error}",
        ) from error
    extension = _require_import(
        "selective_scan_cuda_oflex", "selective_scan_cuda_oflex_import"
    )
    if not callable(getattr(extension, "fwd", None)) or not callable(
        getattr(extension, "bwd", None)
    ):
        raise ProbeFailure(
            "selective_scan_cuda_oflex_api",
            "selective_scan_cuda_oflex must expose callable fwd and bwd kernels",
        )
    pretrained = Path(
        args.pretrained_checkpoint or os.environ.get("VMAMBA_T_PRETRAINED", "")
    )
    if not pretrained.is_file() or pretrained.stat().st_size == 0:
        raise ProbeFailure(
            "vmamba_pretrained_checkpoint",
            f"required VMamba pretrained checkpoint is missing or empty: {pretrained}",
        )
    construction: dict[str, Any] = {"requested": bool(args.construct_model)}
    if args.construct_model:
        os.environ["VMAMBA_ROOT"] = str(vmamba_root)
        os.environ["VMAMBA_T_PRETRAINED"] = str(pretrained)
        try:
            from src.models.vmamba_frcnn.factory import VMambaFRCNNFactory

            result = VMambaFRCNNFactory(args.repo_root).build(
                num_classes=2,
                device="cuda:0",
            )
            model = result.model
            model.eval()
            torch = _require_import("torch", "vmamba_forward_torch_import")
            with torch.no_grad():
                features = model.extract_feat(
                    torch.zeros((1, 3, 64, 64), device="cuda:0")
                )
            if not features:
                raise RuntimeError("VMamba extract_feat returned no feature maps")
            construction.update(
                {
                    "status": "PASS",
                    "report": result.report,
                    "forward_feature_shapes": [list(value.shape) for value in features],
                }
            )
            del result
        except Exception as error:
            raise ProbeFailure(
                "vmamba_model_construction",
                f"VMamba-T Faster R-CNN construction failed: {error}",
            ) from error
    return {
        "vmamba_root": str(vmamba_root.resolve()),
        "vmamba_revision": revision,
        "detector_config": str(config.resolve()),
        "registration_module": str(registration.__file__),
        "selective_scan": "selective_scan_cuda_oflex",
        "pretrained_checkpoint": str(pretrained.resolve()),
        "pretrained_size": pretrained.stat().st_size,
        "construction": construction,
    }


def _rtdetr_probe(
    spec: dict[str, Any],
    args: argparse.Namespace,
    versions: dict[str, str],
) -> dict[str, Any]:
    for package, expected in (("transformers", "4.52.4"), ("accelerate", "1.7.0")):
        _require_exact(package, expected, versions)
    try:
        from transformers import RTDetrV2Config, RTDetrV2ForObjectDetection
    except Exception as error:
        raise ProbeFailure(
            "rtdetr_import",
            f"RT-DETRv2 class import failed: {error}",
        ) from error
    source = _source(spec, "PekingU/rtdetr_v2_r50vd")
    try:
        config = RTDetrV2Config.from_pretrained(
            str(source["name"]),
            revision=str(source["revision"]),
            local_files_only=bool(args.quick),
        )
    except Exception as error:
        raise ProbeFailure(
            "rtdetr_pretrained_revision",
            f"could not resolve pinned RT-DETRv2 config: {error}",
        ) from error
    return {
        "config_class": RTDetrV2Config.__name__,
        "model_class": RTDetrV2ForObjectDetection.__name__,
        "pretrained_model": str(source["name"]),
        "pretrained_revision": str(source["revision"]),
        "resolved_name": str(getattr(config, "_name_or_path", source["name"])),
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    spec = resolved_runtime_spec(args.repo_root, args.model_id)
    if spec["family"] != args.environment:
        raise ProbeFailure(
            "family_contract",
            f"model {args.model_id} requires {spec['family']}, not {args.environment}",
        )
    versions, _ = _base_probe(spec, require_gpu=args.require_gpu)
    checks: dict[str, Any] = {}
    if args.environment in {"openmmlab", "vmamba"}:
        checks["openmmlab"] = _openmmlab_probe(spec, args, versions)
    if args.environment == "vmamba":
        checks["vmamba"] = _vmamba_probe(spec, args, versions)
    elif args.environment == "rtdetr":
        checks["rtdetr"] = _rtdetr_probe(spec, args, versions)
    return {
        "schema_version": 2,
        "status": "PASS",
        "environment": args.environment,
        "model_id": args.model_id,
        "quick": bool(args.quick),
        "python_executable": sys.executable,
        "versions": versions,
        "checks": checks,
        "environment_variables": {
            name: os.environ.get(name) for name in RECORDED_VARIABLES
        },
        "hardware": collect_environment(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        required=True,
        choices=["rtdetr", "openmmlab", "vmamba"],
    )
    parser.add_argument("--model-id")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--construct-model", action="store_true")
    parser.add_argument("--mmdet-root")
    parser.add_argument("--vmamba-root")
    parser.add_argument("--vmamba-config")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(arguments)
    args.repo_root = args.repo_root.resolve()
    args.model_id = args.model_id or DEFAULT_MODELS[args.environment]
    return args


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        report = verify(args)
    except Exception as error:
        stage = error.stage if isinstance(error, ProbeFailure) else "unexpected_probe_error"
        report = {
            "schema_version": 2,
            "status": "FAILED",
            "environment": args.environment,
            "model_id": args.model_id,
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "python_executable": sys.executable,
            "traceback": traceback.format_exc(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if args.json_output:
            write_json(args.json_output, report)
        print(f"{args.environment} environment: FAILED", file=sys.stderr)
        print(f"Stage: {stage}", file=sys.stderr)
        print(f"Python: {sys.executable}", file=sys.stderr)
        print(f"Cause: {type(error).__name__}: {error}", file=sys.stderr)
        if args.json_output:
            print(f"Probe output: {args.json_output}", file=sys.stderr)
        return 1
    if args.json_output:
        write_json(args.json_output, report)
    print(f"{args.environment} environment: PASS")
    print(f"Python: {report['versions']['python']}")
    print(f"PyTorch: {report['versions']['torch']}")
    print(f"CUDA: {report['versions']['torch_cuda']}")
    print(f"GPU: {report['versions'].get('gpu', 'not required')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
