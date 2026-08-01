#!/usr/bin/env python
"""Report current repository, path, package, and hardware assumptions.

The diagnostic is read-only by default and never imports or constructs a model.
It uses package metadata rather than importing framework modules.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


REQUIREMENT_FILES = (
    "requirements.txt",
    "requirements-colab.txt",
    "requirements-dataset-colab.txt",
    "requirements-hpo-colab.txt",
    "requirements-notebook-test.txt",
    "requirements-openmmlab-py310-cu118.txt",
    "requirements-rtdetr-colab.txt",
    "requirements/legacy-colab.txt",
)
PATH_ENVIRONMENT_VARIABLES = (
    "BENCHMARK_REPOSITORY_BRANCH",
    "BENCHMARK_REPOSITORY_URL",
    "BENCHMARK_REPO_ROOT",
    "MMDET_ROOT",
    "SMOKE_TEST",
    "VISDRONE_DRIVE_ROOT",
    "VISDRONE_MODEL_PYTHON",
    "VISDRONE_RUNTIME_MANIFEST",
    "VMAMBA_ROOT",
    "VMAMBA_T_PRETRAINED",
)
PACKAGE_NAME = re.compile(r"^([A-Za-z0-9_.-]+)")


def _command(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__}
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _git_value(repository_root: Path, *arguments: str) -> str | None:
    result = _command(["git", "-C", str(repository_root), *arguments])
    if not result.get("available") or result.get("returncode") != 0:
        return None
    return str(result.get("stdout") or "")


def _redact_url_credentials(value: str | None) -> str | None:
    if not value or "://" not in value:
        return value
    parsed = urlsplit(value)
    if parsed.username is None and parsed.password is None:
        return value
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, parsed.query, parsed.fragment))


def git_snapshot(repository_root: Path) -> dict[str, Any]:
    is_repository = (repository_root / ".git").exists()
    if not is_repository:
        return {"is_repository": False}
    status = _git_value(repository_root, "status", "--porcelain")
    return {
        "is_repository": True,
        "commit": _git_value(repository_root, "rev-parse", "HEAD"),
        "branch": _git_value(repository_root, "branch", "--show-current"),
        "remote_origin": _redact_url_credentials(
            _git_value(repository_root, "remote", "get-url", "origin")
        ),
        "dirty": bool(status),
        "status_porcelain": status,
    }


def _requirement_entries(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "includes": [], "requirements": []}
    includes: list[str] = []
    requirements: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--requirement ")):
            includes.append(line.split(maxsplit=1)[1])
        else:
            requirements.append(line)
    return {"exists": True, "includes": includes, "requirements": requirements}


def requirement_snapshot(repository_root: Path) -> dict[str, Any]:
    files = {
        relative: _requirement_entries(repository_root / relative)
        for relative in REQUIREMENT_FILES
    }
    package_names: set[str] = set()
    for record in files.values():
        for requirement in record["requirements"]:
            match = PACKAGE_NAME.match(requirement)
            if match:
                package_names.add(match.group(1))
    package_names.update(
        {
            "torch",
            "torchvision",
            "transformers",
            "mmcv",
            "mmengine",
            "mmdet",
            "selective-scan-cuda",
            "uv",
        }
    )
    installed: dict[str, str | None] = {}
    for name in sorted(package_names, key=str.lower):
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    return {"files": files, "installed_versions": installed}


def _load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        import yaml
    except ImportError:
        return None
    if not path.is_file():
        return None
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _path_status(path: Path) -> dict[str, Any]:
    return {
        "value": str(path),
        "exists": path.exists(),
        "is_directory": path.is_dir(),
        "readable": os.access(path, os.R_OK) if path.exists() else False,
        "writable_by_access_check": os.access(path, os.W_OK) if path.exists() else False,
    }


def configured_paths(repository_root: Path) -> dict[str, Any]:
    project_config = _load_yaml(repository_root / "project_config.yaml") or {}
    colab = project_config.get("colab", {})
    configured_repository = Path(
        str(colab.get("repository_path", "/content/aerial-object-detection-benchmark"))
    )
    configured_drive = Path(
        os.environ.get(
            "VISDRONE_DRIVE_ROOT",
            str(
                colab.get(
                    "drive_root",
                    "/content/drive/MyDrive/visdrone_architecture_benchmark",
                )
            ),
        )
    )
    return {
        "active_repository_root": _path_status(repository_root),
        "configured_colab_repository": _path_status(configured_repository),
        "configured_drive_root": _path_status(configured_drive),
        "colab_drive_mount": _path_status(Path("/content/drive")),
        "local_cache_default": _path_status(Path("/content/visdrone_cache")),
        "isolated_environment_default": _path_status(
            Path("/content/visdrone_model_envs")
        ),
    }


def source_pins(repository_root: Path) -> dict[str, Any]:
    config = _load_yaml(repository_root / "configs" / "runtime_environments.yaml")
    if config is None:
        return {"available": False, "reason": "PyYAML or runtime config unavailable"}
    families: dict[str, Any] = {}
    for family in ("rtdetr", "openmmlab", "vmamba"):
        value = config.get(family, {})
        families[family] = {
            "python": value.get("python"),
            "torch": value.get("torch"),
            "torchvision": value.get("torchvision"),
            "requirements": value.get("requirements"),
            "sources": value.get("sources", []),
            "pretrained": value.get("pretrained"),
        }
    return {
        "available": True,
        "schema_version": config.get("schema_version"),
        "provisioner": config.get("provisioner"),
        "families": families,
    }


def notebook_snapshot(repository_root: Path) -> dict[str, Any]:
    snapshot = (
        repository_root
        / "schemas"
        / "legacy"
        / "notebook_artifact_inventory_v1.json"
    )
    if not snapshot.is_file():
        return {"available": False, "path": str(snapshot)}
    value = json.loads(snapshot.read_text(encoding="utf-8"))
    imports = sorted(
        {
            name
            for notebook in value.get("notebooks", [])
            for name in notebook.get("imports", [])
        }
    )
    return {
        "available": True,
        "path": str(snapshot),
        "notebook_count": value.get("notebook_count"),
        "imports": imports,
    }


def hardware_snapshot() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    gpu = (
        _command(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )
        if nvidia_smi
        else {"available": False}
    )
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "nvidia_smi_path": nvidia_smi,
        "gpu_query": gpu,
    }


def collect_report(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    return {
        "schema_version": 1,
        "diagnostic": "current-environment-read-only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_construction_performed": False,
        "model_modules_imported": False,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "colab_detected": bool(
                os.environ.get("COLAB_RELEASE_TAG") or Path("/content").is_dir()
            ),
        },
        "repository": git_snapshot(root),
        "paths": configured_paths(root),
        "environment_variables": {
            name: (
                _redact_url_credentials(os.environ.get(name))
                if name == "BENCHMARK_REPOSITORY_URL"
                else os.environ.get(name)
            )
            for name in PATH_ENVIRONMENT_VARIABLES
        },
        "requirements": requirement_snapshot(root),
        "third_party_sources": source_pins(root),
        "notebooks": notebook_snapshot(root),
        "hardware": hardware_snapshot(),
    }


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = collect_report(args.repo_root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.expanduser().resolve().write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
