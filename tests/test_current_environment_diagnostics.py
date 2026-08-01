from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.diagnostics.report_current_environment import (
    _redact_url_credentials,
    collect_report,
)


ROOT = Path(__file__).resolve().parents[1]


def test_diagnostic_is_read_only_and_does_not_construct_models(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "requirements").mkdir()
    (repo / "requirements.txt").write_text("PyYAML>=6\n", encoding="utf-8")
    (repo / "requirements" / "legacy-colab.txt").write_text(
        "-r ../requirements.txt\n", encoding="utf-8"
    )
    (repo / "project_config.yaml").write_text(
        "colab:\n"
        "  repository_path: /content/example\n"
        "  drive_root: /content/drive/MyDrive/example\n",
        encoding="utf-8",
    )
    (repo / "configs" / "runtime_environments.yaml").write_text(
        "schema_version: 1\n", encoding="utf-8"
    )
    before = {
        path.relative_to(repo): path.read_bytes()
        for path in repo.rglob("*")
        if path.is_file()
    }
    report = collect_report(repo)
    after = {
        path.relative_to(repo): path.read_bytes()
        for path in repo.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert report["model_construction_performed"] is False
    assert report["model_modules_imported"] is False
    assert report["repository"] == {"is_repository": False}
    assert report["requirements"]["files"]["requirements.txt"]["exists"]


def test_diagnostic_cli_reports_current_repository_without_model_setup() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.diagnostics.report_current_environment",
            "--repo-root",
            str(ROOT),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["diagnostic"] == "current-environment-read-only"
    assert report["repository"]["is_repository"] is True
    assert report["notebooks"]["notebook_count"] == 19
    assert report["model_construction_performed"] is False
    assert report["third_party_sources"]["available"] is True


def test_legacy_colab_inventory_keeps_model_families_separate() -> None:
    text = (ROOT / "requirements" / "legacy-colab.txt").read_text(encoding="utf-8")
    assert "requirements-dataset-colab.txt" in text
    assert "requirements-rtdetr-colab.txt" not in text
    assert "requirements-openmmlab-py310-cu118.txt" not in text
    assert "torch==" not in text


def test_diagnostic_redacts_credentials_embedded_in_urls() -> None:
    value = _redact_url_credentials("https://user:secret@example.com/owner/repo.git")
    assert value == "https://example.com/owner/repo.git"
