"""Pinned, model-family-specific environments for hosted Google Colab."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.utils.serialization import read_json, read_yaml, write_json

MODEL_PYTHON_ENV = "VISDRONE_MODEL_PYTHON"
RUNTIME_MANIFEST_ENV = "VISDRONE_RUNTIME_MANIFEST"
RUNTIME_CONFIG = "configs/runtime_environments.yaml"


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


def resolved_runtime_spec(
    repo_root: str | Path, model_id: str
) -> dict[str, Any]:
    return _resolved_spec(Path(repo_root).resolve(), model_id)


def _runtime_hash(repo_root: Path, spec: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for relative in (
        "pyproject.toml",
        "requirements-dataset-colab.txt",
        str(spec["requirements"]),
    ):
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


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, check=True, cwd=cwd)


def _ensure_uv(version: str) -> None:
    try:
        installed = importlib.metadata.version("uv")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed == version:
        return
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"uv=={version}",
        ]
    )


def _uv(*arguments: object) -> list[str]:
    return [sys.executable, "-m", "uv", *(str(value) for value in arguments)]


def _install_runtime(
    repo_root: Path,
    environment: Path,
    python: Path,
    spec: dict[str, Any],
) -> None:
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
        )
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
            )
        )
    _run(
        _uv(
            "pip",
            "install",
            "--python",
            python,
            "-r",
            repo_root / str(spec["requirements"]),
        )
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
        )
    )


def provision_isolated_environment(
    model_id: str,
    repo_root: str | Path,
    drive_root: str | Path,
    *,
    runtime_base: str | Path = "/content/visdrone_model_envs",
) -> dict[str, Any]:
    """Provision and verify a selected family without mutating the notebook kernel."""
    repo = Path(repo_root).resolve()
    drive = Path(drive_root).expanduser().resolve()
    spec = _resolved_spec(repo, model_id)
    runtime_hash = _runtime_hash(repo, spec)
    environment = Path(runtime_base) / f"{spec['family']}-{runtime_hash[:12]}"
    python = _python_path(environment)
    marker = environment / "benchmark_runtime.json"
    existing = read_json(marker) if marker.is_file() else {}
    packages_changed = existing.get("runtime_hash") != runtime_hash or not python.is_file()

    _ensure_uv(str(spec["provisioner"]["version"]))
    if not python.is_file():
        environment.parent.mkdir(parents=True, exist_ok=True)
        _run(
            _uv(
                "venv",
                "--python",
                spec["python"],
                "--python-preference",
                "managed",
                environment,
            )
        )
    if packages_changed:
        _install_runtime(repo, environment, python, spec)
        write_json(
            marker,
            {
                "runtime_hash": runtime_hash,
                "model_id": model_id,
                "family": spec["family"],
                "status": "PACKAGES_INSTALLED",
            },
        )

    probe_path = environment / "environment_probe.json"
    _run(
        [
            str(python),
            "-m",
            "scripts.verify_model_environments",
            "--environment",
            "rtdetr" if spec["family"] == "rtdetr" else "openmmlab",
            "--require-gpu",
            "--json-output",
            str(probe_path),
        ],
        cwd=repo,
    )
    probe = read_json(probe_path)
    manifest = {
        "schema_version": 1,
        "status": "PACKAGES_AND_GPU_VERIFIED",
        "model_id": model_id,
        "family": spec["family"],
        "python_executable": str(python),
        "runtime_hash": runtime_hash,
        "dependency_lock_hash": runtime_hash,
        "packages_changed": packages_changed,
        "environment": probe,
        "sources": spec["sources"],
        "package_contract": {
            key: spec[key]
            for key in ("python", "torch", "torchvision", "requirements")
        },
        "compatibility_claim": (
            "Package and GPU probe only; full compatibility requires a READY "
            "adapter smoke gate in this exact runtime."
        ),
    }
    manifest_path = (
        drive
        / "environment_manifests"
        / model_id
        / f"{runtime_hash}.json"
    )
    write_json(manifest_path, manifest)
    os.environ[MODEL_PYTHON_ENV] = str(python)
    os.environ[RUNTIME_MANIFEST_ENV] = str(manifest_path)
    return {**manifest, "manifest_path": str(manifest_path)}
