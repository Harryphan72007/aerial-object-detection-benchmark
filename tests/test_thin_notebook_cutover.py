from __future__ import annotations

import ast
from pathlib import Path

import nbformat

from scripts.run_notebook_smoke import PRIMARY_NOTEBOOKS
from scripts.validate_notebooks import CANONICAL_NOTEBOOKS, validate_notebook
from src.evaluation.versioned import read_prediction_artifact


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("resnet50", "swin_t", "vmamba_t", "rtdetrv2")


def test_all_canonical_notebooks_are_thin_and_package_backed() -> None:
    for name in sorted(CANONICAL_NOTEBOOKS):
        path = ROOT / "notebooks" / name
        assert path.is_file(), name
        assert not validate_notebook(path)
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        tree = ast.parse(source)
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in tree.body
        )
        assert "from src" in source or "import src" in source


def test_every_model_has_exactly_one_smoke_entry_point() -> None:
    """One notebook per model, plus one report — the whole operator surface."""
    for model in MODELS:
        matching = [name for name in PRIMARY_NOTEBOOKS if model in name]
        assert len(matching) == 1, f"{model}: {matching}"
    assert len(PRIMARY_NOTEBOOKS) == len(MODELS) + 1


def test_retired_lr_controlled_protocol_is_gone() -> None:
    """Only two_stage_random_hpo_v1 is live; lr_controlled_v1 no longer ships.

    It used to survive as an entry point that raised. A module that cannot be
    imported at all is a stronger guarantee than one that refuses at runtime,
    and it removes the second protocol from the provenance surface entirely.
    """
    import importlib

    import pytest

    for module in ("src.workflows.model_day", "src.training.lr_workflow"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)
    for script in ("benchmark.py", "lr_search.py", "full_dataset_finetune.py"):
        assert not (ROOT / "scripts" / script).exists(), script


def test_canonical_notebooks_do_not_require_browser_uploads() -> None:
    for name in CANONICAL_NOTEBOOKS:
        notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        assert "files.upload" not in source


def test_legacy_prediction_artifact_remains_readable() -> None:
    artifact = read_prediction_artifact(
        ROOT / "tests" / "fixtures" / "legacy_artifacts" / "predictions.json",
        legacy_identity={
            "model_id": "faster_rcnn_resnet50",
            "dataset_track": "2class",
            "run_id": "legacy-fixture",
        },
    )
    assert artifact["schema_version"] == 0
    assert artifact["artifact_type"] == "legacy_coco_predictions"
    assert len(artifact["predictions"]) == 3
