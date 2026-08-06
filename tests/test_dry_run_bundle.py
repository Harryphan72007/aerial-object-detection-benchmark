"""PR-09: the dry-run-tiny bundle validates like a real bundle but can never be
mistaken for a canonical result.

The GPU train/eval chain in scripts/dry_run_tiny.py cannot run here; these tests
cover the isolation guarantees: dry-run detection, that the results/ publisher
and validator reject a dry-run bundle, and that a marked bundle still satisfies
the production schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dry_run_tiny import (
    dry_run_bundle_root,
    mark_bundle_as_dry_run,
    validate_dry_run_bundle,
)
from scripts.validate_results import validate_repo_results
from src.result_export import (
    DRY_RUN_NAMESPACE,
    export_bundle,
    is_dry_run_bundle,
    validate_bundle,
)

from test_result_export_workflow import _make_bundle


def test_is_dry_run_bundle_detects_every_marker() -> None:
    assert is_dry_run_bundle({"dry_run_tiny": True})
    assert is_dry_run_bundle({"bundle_kind": "dry_run"})
    assert is_dry_run_bundle({"result_bundle_id": "faster_rcnn_resnet50__dry_run__x"})
    assert not is_dry_run_bundle({"result_bundle_id": "faster_rcnn_resnet50__2class__x"})
    assert not is_dry_run_bundle({})


def test_dry_run_namespace_is_disjoint_from_result_bundles(tmp_path: Path) -> None:
    root = dry_run_bundle_root(tmp_path)
    assert root.parts[-2:] == (DRY_RUN_NAMESPACE, "result_bundles")
    # The canonical publisher copies from <drive>/result_bundles; the dry-run
    # namespace is a sibling the canonical path never walks.
    assert "result_bundles" in root.as_posix()
    assert (tmp_path / "result_bundles") != root


def test_validate_repo_results_rejects_dry_run_latest_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifests" / "latest_result_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "result_bundle_id": "faster_rcnn_resnet50__2class__x",
                "dry_run_tiny": True,
                "dataset_track": "2class",
                "model_id": "faster_rcnn_resnet50",
                "run_id": "r",
            }
        ),
        encoding="utf-8",
    )
    errors = validate_repo_results(tmp_path)
    assert any("dry-run-tiny bundle must not be published" in e for e in errors)


def test_marked_bundle_passes_schema_but_publisher_and_validator_reject_it(
    tmp_path: Path,
) -> None:
    # A real, schema-valid bundle under the Drive result_bundles location.
    drive = tmp_path / "drive"
    bundle = _make_bundle(drive / "result_bundles")
    bundle_id = bundle.name

    # It is a valid canonical bundle until we mark it as a dry run.
    assert validate_bundle(bundle) == []
    mark_bundle_as_dry_run(bundle)

    # Still satisfies the production bundle schema...
    assert validate_bundle(bundle) == []
    # ...and the dry-run validator is happy with the marked bundle.
    assert validate_dry_run_bundle(bundle) == []

    # But the results/ publisher refuses to promote it.
    with pytest.raises(ValueError, match="refusing to publish a dry-run"):
        export_bundle(drive, bundle_id, tmp_path / "repo")


def test_validate_repo_results_rejects_dry_run_bundle_under_results(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    bundle = _make_bundle(results / "bundles")
    mark_bundle_as_dry_run(bundle)
    (results / "manifests").mkdir(parents=True)
    (results / "manifests" / "latest_result_manifest.json").write_text(
        json.dumps(
            {
                "result_bundle_id": bundle.name,
                "bundle_path": f"bundles/{bundle.name}",
                "dataset_track": "2class",
                "model_id": "faster_rcnn_resnet50",
                "run_id": "r",
            }
        ),
        encoding="utf-8",
    )
    errors = validate_repo_results(results)
    assert any("dry-run-tiny bundle present" in e for e in errors)
