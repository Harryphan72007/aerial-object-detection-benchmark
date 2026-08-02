"""Environment provenance captured inside the selected model runtime."""
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path
from typing import Any

from src.reproducibility import git_commit
from src.subprocess_utils import subprocess_environment_record
from src.utils.environment import collect_environment
from src.utils.serialization import write_json
from src.workflows.isolated_environment import resolved_runtime_spec


def write_runtime_environment_manifest(
    run_dir: str | Path,
    model_id: str,
    repo_root: str | Path,
) -> dict[str, Any]:
    destination = Path(run_dir)
    repo = Path(repo_root).resolve()
    packages = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    manifest = {
        "schema_version": 1,
        "model_id": model_id,
        "source_commit": git_commit(repo),
        "python_executable": sys.executable,
        "environment": collect_environment(),
        "subprocess_environment": subprocess_environment_record(),
        "packages": packages,
        "runtime_contract": resolved_runtime_spec(repo, model_id),
    }
    path = destination / "runtime_environment.json"
    write_json(path, manifest)
    (destination / "runtime_packages.txt").write_text(
        "\n".join(packages) + "\n", encoding="utf-8"
    )
    return {**manifest, "path": str(path)}
