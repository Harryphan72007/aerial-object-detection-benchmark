from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

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
