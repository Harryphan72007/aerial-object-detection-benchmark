"""Transactional, model-family-specific environments for hosted Google Colab."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.subprocess_utils import (
    CheckedSubprocessError,
    build_model_subprocess_environment,
    run_checked,
)
from src.reproducibility import git_commit
from src.utils.serialization import read_json, read_yaml, write_json

MODEL_PYTHON_ENV = "VISDRONE_MODEL_PYTHON"
RUNTIME_MANIFEST_ENV = "VISDRONE_RUNTIME_MANIFEST"
RUNTIME_CONFIG = "configs/runtime_environments.yaml"
RUNTIME_SCHEMA_VERSION = 2
READY_STATE = "READY"
INCOMPLETE_STATES = {"CREATING", "INSTALLING", "VERIFYING"}


class EnvironmentProvisioningError(RuntimeError):
    """Environment setup failed without changing experiment artifacts."""


def runtime_family(model_id: str) -> str:
    if model_id == "rtdetrv2_l":
        return "rtdetr"
    if model_id == "faster_rcnn_vmamba_t":
        return "vmamba"
    return "openmmlab"


def _resolved_spec(repo_root: Path, model_id: str) -> dict[str, Any]:
    config = read_yaml(repo_root / RUNTIME_CONFIG)
    family = runtime_family(model_id)
    if family == "vmamba":
        spec = dict(config["openmmlab"])
        spec["sources"] = [
            *config["openmmlab"]["sources"],
            *config["vmamba"]["sources"],
        ]
        spec["pretrained"] = config["vmamba"]["pretrained"]
        model_config = read_yaml(repo_root / "configs/faster_rcnn_vmamba_t/model.yaml")
        spec["framework_config"] = model_config["framework_config"]
    else:
        spec = dict(config[family])
    spec.update(
        {
            "family": family,
            "model_id": model_id,
            "provisioner": config["provisioner"],
            "schema_version": config["schema_version"],
        }
    )
    return spec


def resolved_runtime_spec(repo_root: str | Path, model_id: str) -> dict[str, Any]:
    return _resolved_spec(Path(repo_root).resolve(), model_id)


def _runtime_hash(repo_root: Path, spec: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    contract_files = [
        "pyproject.toml",
        "requirements-dataset-colab.txt",
        str(spec["requirements"]),
        "scripts/verify_model_environments.py",
        "src/notebook_utils.py",
        "src/subprocess_utils.py",
        "src/workflows/environment.py",
        "src/workflows/isolated_environment.py",
    ]
    if spec["family"] == "vmamba":
        contract_files.extend(
            [
                "configs/faster_rcnn_vmamba_t/model.yaml",
                "src/models/vmamba_frcnn/importer.py",
                "src/models/vmamba_frcnn/factory.py",
            ]
        )
    for relative in contract_files:
        path = repo_root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _python_path(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment_name: str | None = None,
    stage: str | None = None,
    python_executable: Path | None = None,
    probe_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    log_path: Path | None = None,
) -> None:
    run_checked(
        command,
        cwd=cwd,
        env=env,
        environment_name=environment_name,
        stage=stage,
        python_executable=python_executable,
        probe_path=probe_path,
        log_path=log_path,
    )


def _ensure_uv(version: str) -> None:
    try:
        installed = importlib.metadata.version("uv")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed == version:
        return
    _run(
        [sys.executable, "-m", "pip", "install", f"uv=={version}"],
        environment_name="provisioner",
        stage="uv_installation",
        python_executable=Path(sys.executable),
    )


def _uv(*arguments: object) -> list[str]:
    return [sys.executable, "-m", "uv", *(str(value) for value in arguments)]


def _install_runtime(
    repo_root: Path,
    environment: Path,
    python: Path,
    spec: dict[str, Any],
) -> None:
    del environment
    common = {
        "environment_name": str(spec["family"]),
        "python_executable": python,
    }
    _run(
        _uv(
            "pip",
            "install",
            "--python",
            python,
            f"torch=={spec['torch']}",
            f"torchvision=={spec['torchvision']}",
            "--index-url",
            spec["torch_index"],
        ),
        stage="torch_installation",
        **common,
    )
    if spec["family"] in {"openmmlab", "vmamba"}:
        _run(
            _uv(
                "pip",
                "install",
                "--python",
                python,
                f"mmcv=={spec['mmcv']['version']}",
                "--find-links",
                spec["mmcv"]["wheel_index"],
                "--only-binary",
                "mmcv",
            ),
            stage="mmcv_wheel_installation",
            **common,
        )
    _run(
        _uv(
            "pip",
            "install",
            "--python",
            python,
            "-r",
            repo_root / str(spec["requirements"]),
        ),
        stage="requirements_installation",
        **common,
    )
    _run(
        _uv(
            "pip",
            "install",
            "--python",
            python,
            "--editable",
            repo_root,
            "--no-deps",
        ),
        stage="repository_installation",
        **common,
    )


def _source(spec: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for source in spec.get("sources", []):
        if str(source.get("name")) == name:
            return source
    raise KeyError(f"runtime source {name!r} is not configured")


def _clone_pinned(
    source: Mapping[str, Any],
    destination: Path,
    *,
    family: str,
    python: Path,
) -> None:
    url = str(source["url"])
    revision = str(source["revision"])
    if not destination.exists():
        _run(
            ["git", "clone", url, str(destination)],
            environment_name=family,
            stage=f"{source['name']}_clone",
            python_executable=python,
        )
    if not (destination / ".git").is_dir():
        raise RuntimeError(f"Refusing to use non-Git upstream directory: {destination}")
    _run(
        ["git", "-C", str(destination), "fetch", "--tags", "origin", revision],
        environment_name=family,
        stage=f"{source['name']}_fetch",
        python_executable=python,
    )
    _run(
        ["git", "-C", str(destination), "checkout", "--detach", revision],
        environment_name=family,
        stage=f"{source['name']}_checkout",
        python_executable=python,
    )


def _family_paths(drive: Path, spec: Mapping[str, Any]) -> dict[str, Path]:
    values: dict[str, Path] = {}
    if spec["family"] in {"openmmlab", "vmamba"}:
        values["mmdet_root"] = drive / "frameworks" / "mmdetection"
    if spec["family"] == "vmamba":
        vmamba_root = drive / "frameworks" / "VMamba"
        values.update(
            {
                "vmamba_root": vmamba_root,
                "vmamba_config": vmamba_root / str(spec["framework_config"]),
                "pretrained": drive / "pretrained" / str(spec["pretrained"]["filename"]),
            }
        )
    return values


def _prepare_family(
    repo: Path,
    drive: Path,
    python: Path,
    spec: dict[str, Any],
) -> dict[str, Path]:
    del repo
    family = str(spec["family"])
    paths = _family_paths(drive, spec)
    if family in {"openmmlab", "vmamba"}:
        _clone_pinned(
            _source(spec, "MMDetection"),
            paths["mmdet_root"],
            family=family,
            python=python,
        )
    if family == "vmamba":
        _clone_pinned(
            _source(spec, "VMamba"),
            paths["vmamba_root"],
            family=family,
            python=python,
        )
        if not paths["vmamba_config"].is_file():
            raise FileNotFoundError(
                f"VMamba detector configuration is missing: {paths['vmamba_config']}"
            )
        pretrained = paths["pretrained"]
        if not pretrained.is_file() or pretrained.stat().st_size == 0:
            raise FileNotFoundError(
                "VMamba pretrained checkpoint validation failed before HPO: "
                f"expected a non-empty file at {pretrained}. Training from scratch "
                "is intentionally disabled."
            )
        _run(
            _uv(
                "pip",
                "install",
                "--python",
                python,
                paths["vmamba_root"] / "kernels" / "selective_scan",
                "--no-build-isolation",
            ),
            environment_name=family,
            stage="selective_scan_compilation",
            python_executable=python,
        )
    return paths


def _probe_environment(paths: Mapping[str, Path]) -> dict[str, str]:
    environment = build_model_subprocess_environment()
    if "mmdet_root" in paths:
        environment["MMDET_ROOT"] = str(paths["mmdet_root"])
    if "vmamba_root" in paths:
        environment["VMAMBA_ROOT"] = str(paths["vmamba_root"])
        environment["VMAMBA_T_PRETRAINED"] = str(paths["pretrained"])
    return environment


def _verification_command(
    python: Path,
    repo: Path,
    spec: Mapping[str, Any],
    paths: Mapping[str, Path],
    probe_path: Path,
    *,
    quick: bool,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "scripts.verify_model_environments",
        "--environment",
        str(spec["family"]),
        "--model-id",
        str(spec["model_id"]),
        "--repo-root",
        str(repo),
        "--require-gpu",
        "--json-output",
        str(probe_path),
    ]
    if quick:
        command.append("--quick")
    if "mmdet_root" in paths:
        command.extend(["--mmdet-root", str(paths["mmdet_root"])])
    if "vmamba_root" in paths:
        command.extend(
            [
                "--vmamba-root",
                str(paths["vmamba_root"]),
                "--vmamba-config",
                str(paths["vmamba_config"]),
                "--pretrained-checkpoint",
                str(paths["pretrained"]),
            ]
        )
        if not quick:
            command.append("--construct-model")
    return command


def _verify_runtime(
    python: Path,
    repo: Path,
    environment: Path,
    spec: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    quick: bool,
) -> tuple[Path, dict[str, Any]]:
    probe = environment / ("quick_environment_probe.json" if quick else "environment_probe.json")
    stage = "quick_probe" if quick else f"{spec['family']}_complete_probe"
    _run(
        _verification_command(python, repo, spec, paths, probe, quick=quick),
        cwd=repo,
        env=_probe_environment(paths),
        environment_name=str(spec["family"]),
        stage=stage,
        python_executable=python,
        probe_path=probe,
        log_path=environment / "logs" / f"{stage}.log",
    )
    payload = read_json(probe)
    if payload.get("status") != "PASS":
        raise RuntimeError(f"{stage} returned a non-PASS probe: {probe}")
    return probe, payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_state(
    marker: Path,
    *,
    runtime_hash: str,
    model_id: str,
    family: str,
    state: str,
    python: Path,
    started_at: str,
    probe_path: Path,
    completed_at: str | None = None,
    failed_stage: str | None = None,
    failure_type: str | None = None,
    failure_message: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "runtime_hash": runtime_hash,
        "model_id": model_id,
        "family": family,
        "state": state,
        "status": state,
        "started_at": started_at,
        "completed_at": completed_at,
        "python_executable": str(python),
        "failed_stage": failed_stage,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "probe_path": str(probe_path),
    }
    if extra:
        record.update(extra)
    write_json(marker, record)
    return record


def _safe_remove_environment(environment: Path, runtime_base: Path) -> None:
    resolved = environment.resolve()
    boundary = runtime_base.resolve()
    if resolved.parent != boundary:
        raise ValueError(
            f"Refusing to remove runtime outside configured base: {resolved}"
        )
    if resolved.exists():
        shutil.rmtree(resolved)


def _clear_runtime_selection() -> None:
    for variable in (
        MODEL_PYTHON_ENV,
        RUNTIME_MANIFEST_ENV,
        "MMDET_ROOT",
        "VMAMBA_ROOT",
        "VMAMBA_T_PRETRAINED",
    ):
        os.environ.pop(variable, None)


def _select_runtime(
    python: Path,
    manifest_path: Path,
    paths: Mapping[str, Path],
) -> None:
    _clear_runtime_selection()
    os.environ[MODEL_PYTHON_ENV] = str(python)
    os.environ[RUNTIME_MANIFEST_ENV] = str(manifest_path)
    if "mmdet_root" in paths:
        os.environ["MMDET_ROOT"] = str(paths["mmdet_root"])
    if "vmamba_root" in paths:
        os.environ["VMAMBA_ROOT"] = str(paths["vmamba_root"])
        os.environ["VMAMBA_T_PRETRAINED"] = str(paths["pretrained"])


def _failed_stage(error: Exception, probe_path: Path | None = None) -> str:
    if probe_path is not None and probe_path.is_file():
        try:
            probe = read_json(probe_path)
        except (OSError, ValueError, json.JSONDecodeError):
            probe = {}
        if probe.get("stage"):
            return str(probe["stage"])
    if isinstance(error, CheckedSubprocessError) and error.stage:
        return error.stage
    if isinstance(error, FileNotFoundError):
        message = str(error).lower()
        if "pretrained" in message:
            return "vmamba_pretrained_checkpoint"
        if "configuration" in message:
            return "vmamba_detector_configuration"
    return "environment_provisioning"


def provision_isolated_environment(
    model_id: str,
    repo_root: str | Path,
    drive_root: str | Path,
    *,
    runtime_base: str | Path = "/content/visdrone_model_envs",
) -> dict[str, Any]:
    """Provision, fully verify, and select a family runtime transactionally."""
    repo = Path(repo_root).resolve()
    drive = Path(drive_root).expanduser().resolve()
    base = Path(runtime_base).expanduser().resolve()
    spec = _resolved_spec(repo, model_id)
    family = str(spec["family"])
    runtime_hash = _runtime_hash(repo, spec)
    environment = base / f"{family}-{runtime_hash[:12]}"
    python = _python_path(environment)
    marker = environment / "benchmark_runtime.json"
    probe_path = environment / "environment_probe.json"
    manifest_path = drive / "environment_manifests" / model_id / f"{runtime_hash}.json"
    _clear_runtime_selection()
    try:
        existing = read_json(marker) if marker.is_file() else {}
    except (OSError, ValueError, json.JSONDecodeError):
        existing = {}
    reusable = (
        existing.get("state") == READY_STATE
        and existing.get("runtime_hash") == runtime_hash
        and existing.get("model_id") == model_id
        and python.is_file()
    )
    paths = _family_paths(drive, spec)
    if reusable:
        try:
            _, probe = _verify_runtime(
                python, repo, environment, spec, paths, quick=True
            )
        except Exception as error:
            _write_state(
                marker,
                runtime_hash=runtime_hash,
                model_id=model_id,
                family=family,
                state="FAILED",
                python=python,
                started_at=str(existing.get("started_at") or _now()),
                probe_path=environment / "quick_environment_probe.json",
                completed_at=_now(),
                failed_stage=_failed_stage(
                    error, environment / "quick_environment_probe.json"
                ),
                failure_type=type(error).__name__,
                failure_message=str(error),
            )
            _safe_remove_environment(environment, base)
            reusable = False
        else:
            manifest = {
                **existing,
                "state": READY_STATE,
                "status": READY_STATE,
                "completed_at": _now(),
                "packages_changed": False,
                "source_commit": git_commit(repo),
                "log_directory": str(environment / "logs"),
                "environment": probe.get("hardware", probe),
                "verification": probe,
                "manifest_path": str(manifest_path),
            }
            write_json(marker, manifest)
            write_json(manifest_path, manifest)
            _select_runtime(python, manifest_path, paths)
            return manifest

    if not reusable and environment.exists():
        _safe_remove_environment(environment, base)
    started_at = _now()
    packages_changed = True
    try:
        _ensure_uv(str(spec["provisioner"]["version"]))
        environment.parent.mkdir(parents=True, exist_ok=True)
        _run(
            _uv(
                "venv",
                "--python",
                spec["python"],
                "--python-preference",
                "managed",
                environment,
            ),
            environment_name=family,
            stage="virtual_environment_creation",
            python_executable=python,
        )
        _write_state(
            marker,
            runtime_hash=runtime_hash,
            model_id=model_id,
            family=family,
            state="CREATING",
            python=python,
            started_at=started_at,
            probe_path=probe_path,
        )
        if not python.is_file():
            raise FileNotFoundError(f"managed Python executable was not created: {python}")
        _write_state(
            marker,
            runtime_hash=runtime_hash,
            model_id=model_id,
            family=family,
            state="INSTALLING",
            python=python,
            started_at=started_at,
            probe_path=probe_path,
        )
        _install_runtime(repo, environment, python, spec)
        paths = _prepare_family(repo, drive, python, spec)
        _write_state(
            marker,
            runtime_hash=runtime_hash,
            model_id=model_id,
            family=family,
            state="VERIFYING",
            python=python,
            started_at=started_at,
            probe_path=probe_path,
        )
        probe_path, probe = _verify_runtime(
            python, repo, environment, spec, paths, quick=False
        )
        manifest = _write_state(
            marker,
            runtime_hash=runtime_hash,
            model_id=model_id,
            family=family,
            state=READY_STATE,
            python=python,
            started_at=started_at,
            completed_at=_now(),
            probe_path=probe_path,
            extra={
                "dependency_lock_hash": runtime_hash,
                "source_commit": git_commit(repo),
                "log_directory": str(environment / "logs"),
                "packages_changed": packages_changed,
                "environment": probe.get("hardware", probe),
                "verification": probe,
                "sources": spec["sources"],
                "package_contract": {
                    key: spec[key]
                    for key in ("python", "torch", "torchvision", "requirements")
                },
                "compatibility_claim": (
                    "Exact package, CUDA, compiled-operation, source, and family "
                    "probe passed; real training correctness still requires the "
                    "adapter GPU smoke gate."
                ),
                "manifest_path": str(manifest_path),
            },
        )
        write_json(manifest_path, manifest)
        _select_runtime(python, manifest_path, paths)
        return manifest
    except Exception as error:
        environment.mkdir(parents=True, exist_ok=True)
        stage = _failed_stage(error, probe_path)
        _write_state(
            marker,
            runtime_hash=runtime_hash,
            model_id=model_id,
            family=family,
            state="FAILED",
            python=python,
            started_at=started_at,
            completed_at=_now(),
            probe_path=probe_path,
            failed_stage=stage,
            failure_type=type(error).__name__,
            failure_message=str(error),
        )
        _clear_runtime_selection()
        label = "VMamba" if family == "vmamba" else family
        raise EnvironmentProvisioningError(
            f"{label} environment verification failed\n"
            f"Stage: {stage}\n"
            f"Python: {python}\n"
            f"Probe output: {probe_path}\n"
            f"Cause: {error}\n"
            f"Suggested action: rerun the notebook to rebuild only {environment}.\n"
            "Google Drive experiment artifacts were not modified."
        ) from error
