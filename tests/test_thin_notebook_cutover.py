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


def test_every_model_has_hpo_and_final_smoke_entrypoints() -> None:
    for model in MODELS:
        assert any(name.startswith("1") and model in name for name in PRIMARY_NOTEBOOKS)
        assert any(name.startswith("2") and model in name for name in PRIMARY_NOTEBOOKS)


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
