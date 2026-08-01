from __future__ import annotations

from pathlib import Path

from scripts.validate_notebooks import validate_notebook
from scripts.validation.check_prohibited_files import validate_file
from scripts.validation.validate_ci_contract import (
    validate_repository_contract,
    validate_yaml_file,
)


ROOT = Path(__file__).resolve().parents[1]


def test_repository_configuration_and_schema_contracts_pass() -> None:
    assert not validate_repository_contract(ROOT)


def test_invalid_configuration_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("model_id: [unterminated\n", encoding="utf-8")
    try:
        validate_yaml_file(path)
    except ValueError as error:
        assert "invalid YAML" in str(error)
    else:
        raise AssertionError("malformed configuration was accepted")


def test_prohibited_artifact_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "model.pth"
    artifact.write_bytes(b"checkpoint")
    assert "prohibited artifact type: .pth" in validate_file(
        artifact, "checkpoints/model.pth"
    )


def test_malformed_notebook_is_rejected(tmp_path: Path) -> None:
    notebook = tmp_path / "broken.ipynb"
    notebook.write_text('{"cells": [', encoding="utf-8")
    errors = validate_notebook(notebook)
    assert errors and "unreadable notebook" in errors[0]
