from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.validation.inventory_legacy_artifacts import (
    inventory_notebook,
    inventory_python_source,
)

from src.evaluation.detection_metrics import detailed_metrics
from src.training.checkpointing import MANIFEST_REQUIRED, validate_manifest_dict
from src.utils.serialization import read_json


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "legacy_artifacts"
SCHEMAS = ROOT / "schemas" / "legacy"
INVENTORY = SCHEMAS / "notebook_artifact_inventory_v1.json"


def test_manifest_and_registry_fixtures_match_existing_reader_contract() -> None:
    manifest = read_json(FIXTURES / "run_manifest.json")
    registry = read_json(FIXTURES / "checkpoint_registry.json")
    assert not validate_manifest_dict(manifest)
    assert MANIFEST_REQUIRED <= set(manifest)
    assert registry["schema_version"] == 1
    assert registry["runs"][manifest["run_id"]] == manifest
    with (FIXTURES / "runs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["run_id"] == manifest["run_id"]
    assert set(rows[0]) == set(manifest)


def test_existing_evaluator_reads_representative_fixture() -> None:
    result = detailed_metrics(
        FIXTURES / "ground_truth.json", FIXTURES / "predictions.json"
    )
    assert result["per_class_detailed"]["person"]["true_positives"] == 1
    assert result["per_class_detailed"]["vehicle"]["true_positives"] == 1
    assert result["background_false_positive_count"] == 1


def test_legacy_json_schemas_and_examples_are_parseable() -> None:
    for path in sorted(SCHEMAS.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"].endswith("2020-12/schema")
        assert schema["type"] in {"array", "object"}
    metrics = read_json(FIXTURES / "evaluation_metrics.json")
    predictions = read_json(FIXTURES / "predictions.json")
    assert metrics["dataset_track"] == "2class"
    assert {"image_id", "category_id", "bbox", "score"} <= set(predictions[0])


def test_notebook_inventory_is_complete_and_current() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.validation.inventory_legacy_artifacts",
            "--repo-root",
            str(ROOT),
            "--check",
            str(INVENTORY),
        ],
        cwd=ROOT,
        check=True,
    )
    inventory = read_json(INVENTORY)
    expected = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "notebooks").rglob("*.ipynb")
    }
    observed = {record["path"] for record in inventory["notebooks"]}
    assert observed == expected
    assert inventory["notebook_count"] == len(expected)
    source_paths = {record["path"] for record in inventory["python_sources"]}
    assert {
        "src/paths.py",
        "src/training/checkpointing.py",
        "scripts/evaluate.py",
    } <= source_paths
    assert inventory["python_source_count"] == len(source_paths)


def test_inventory_hashes_are_independent_of_checkout_newlines(tmp_path: Path) -> None:
    notebook_value = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["from src import paths\n", "print('artifact.json')\n"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_lf = tmp_path / "lf.ipynb"
    notebook_crlf = tmp_path / "crlf.ipynb"
    rendered = json.dumps(notebook_value, indent=2)
    notebook_lf.write_bytes((rendered + "\n").encode("utf-8"))
    notebook_crlf.write_bytes((rendered.replace("\n", "\r\n") + "\r\n").encode("utf-8"))
    assert inventory_notebook(notebook_lf, tmp_path)["sha256"] == inventory_notebook(
        notebook_crlf, tmp_path
    )["sha256"]

    source_lf = tmp_path / "lf.py"
    source_crlf = tmp_path / "crlf.py"
    source_lf.write_bytes(b"from pathlib import Path\nPath('artifact.json').read_text()\n")
    source_crlf.write_bytes(
        b"from pathlib import Path\r\nPath('artifact.json').read_text()\r\n"
    )
    assert inventory_python_source(source_lf, tmp_path)["sha256"] == (
        inventory_python_source(source_crlf, tmp_path)["sha256"]
    )
