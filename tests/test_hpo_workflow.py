from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.hpo.final_workflow import (
    FINAL_SEEDS,
    FinalExperimentWorkflow,
    configuration_hash,
)
from src.hpo.search_spaces import broad_search_space, refined_search_space
from src.hpo.workflow import HPO_PROTOCOL_ID
from src.result_export import HPO_REQUIRED_BUNDLE_FILES, validate_bundle
from src.training.trainer import TrainingOrchestrator
from src.utils.serialization import read_json, write_json, write_yaml
from src.workflows.hpo_comparison import aggregate_hpo_results
from src.workflows.publishing import (
    LATEST_MANIFEST,
    _authentication_environment,
    _prepare_publication_clone,
    _safe_cleanup,
    _validate_source_checkout,
    _validate_staged_paths,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "rtdetrv2_l"


class FakeOrchestrator:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def run(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"model_id": model_id, **kwargs})
        run_dir = Path(kwargs["explicit_run_dir"])
        write_json(
            run_dir / "run_manifest.json",
            {
                "run_id": kwargs["explicit_run_id"],
                "run_dir": str(run_dir),
                "model_id": model_id,
                "status": "completed",
            },
        )
        return read_json(run_dir / "run_manifest.json")


def _best_config(root: Path) -> None:
    write_yaml(
        root
        / "hpo"
        / HPO_PROTOCOL_ID
        / MODEL_ID
        / "2class"
        / "best_config.yaml",
        {
            "model_id": MODEL_ID,
            "dataset_track": "2class",
            "protocol_id": HPO_PROTOCOL_ID,
            "search_seed": 42,
            "parameters": {"learning_rate": 0.00002},
        },
    )


def test_model_specific_spaces_refine_only_observed_values():
    broad = broad_search_space(MODEL_ID)
    strongest = [
        {
            name: (
                definition["choices"][0]
                if definition["kind"] == "categorical"
                else definition["low"]
            )
            for name, definition in broad.items()
        }
    ]
    refined = refined_search_space(broad, strongest)
    assert refined["num_queries"]["choices"] == [200]
    assert set(refined) == set(broad)


def test_final_workflow_automatically_runs_two_recipes_and_three_seeds(tmp_path):
    _best_config(tmp_path)
    fake = FakeOrchestrator()
    workflow = FinalExperimentWorkflow(
        ROOT, tmp_path, MODEL_ID, "2class", orchestrator=fake
    )
    result = workflow.run(start_expensive_stage=True)
    assert len(result["runs"]) == 6
    assert {call["seed"] for call in fake.calls} == set(FINAL_SEEDS)
    assert {call["baseline_or_tuned"] for call in fake.calls} == {
        "baseline",
        "tuned",
    }
    assert all(call["protocol_id"] == HPO_PROTOCOL_ID for call in fake.calls)
    assert all(
        call["train_annotation_override"].name == "instances_train.json"
        for call in fake.calls
    )


def test_resume_contract_rejects_configuration_drift(tmp_path):
    _best_config(tmp_path)
    fake = FakeOrchestrator()
    workflow = FinalExperimentWorkflow(
        ROOT, tmp_path, MODEL_ID, "2class", orchestrator=fake
    )
    contract = workflow._contract(42, "tuned", {"learning_rate": 1e-5}, 1, 8)
    run_dir = tmp_path / "checkpoints" / "final" / MODEL_ID / "old"
    write_json(run_dir / "resume_contract.json", contract)
    (run_dir / "last.pth").write_bytes(b"preserve")
    changed = dict(contract, configuration_hash=configuration_hash({"x": 1}))
    run_id, selected, resume = workflow._resumable(changed)
    assert selected != run_dir
    assert resume is None
    assert run_id != "old"
    assert (run_dir / "last.pth").read_bytes() == b"preserve"


def test_controlled_override_report_must_match_every_requested_value(tmp_path):
    write_json(
        tmp_path / "applied_overrides.json",
        {"applied": {"learning_rate": 0.1}, "unsupported": {}},
    )
    with pytest.raises(RuntimeError, match="ignored or changed"):
        TrainingOrchestrator._validate_controlled_overrides(
            tmp_path,
            {
                "run_kind": "hpo_phase_a_trial",
                "overrides": {"learning_rate": 0.2},
            },
        )


def test_hpo_aggregation_separates_recipe_and_reports_missing_seeds(tmp_path):
    registry: dict[str, Any] = {"schema_version": 1, "runs": {}}
    for recipe in ("baseline", "tuned"):
        for seed in FINAL_SEEDS:
            run_id = f"{MODEL_ID}__2class__640__20260101_000000__seed{seed}_{recipe}"
            registry["runs"][run_id] = {
                "run_id": run_id,
                "model_id": MODEL_ID,
                "dataset_track": "2class",
                "protocol_id": HPO_PROTOCOL_ID,
                "run_kind": "final_complete_official_train",
                "baseline_or_tuned": recipe,
                "seed": seed,
                "input_resolution": 640,
                "status": "completed",
                "created_at": "2026-01-01",
            }
            write_json(
                tmp_path
                / "evaluation"
                / f"{run_id}__res640__metrics.json",
                {
                    "dataset_track": "2class",
                    "seed": seed,
                    "evaluation_resolution": 640,
                    "mAP": seed / 10000,
                    "APtiny": seed / 20000,
                },
            )
    write_json(
        tmp_path / "experiment_registry" / "checkpoint_registry.json",
        registry,
    )
    result = aggregate_hpo_results(tmp_path, "2class")
    selected = [
        group
        for group in result["groups"]
        if group["model_id"] == MODEL_ID
    ]
    assert {group["baseline_or_tuned"] for group in selected} == {
        "baseline",
        "tuned",
    }
    assert all(group["status"] == "COMPLETE" for group in selected)
    assert all(group["metrics"]["mAP"]["count"] == 3 for group in selected)


def test_multi_seed_bundle_validation_rejects_checkpoint_artifacts(tmp_path):
    bundle = tmp_path / f"{MODEL_ID}__2class__hpo__20260101_000000"
    for relative in HPO_REQUIRED_BUNDLE_FILES:
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            payload: dict[str, Any] = {}
            if relative == "metrics/comparison.json":
                payload = {
                    "groups": [
                        {
                            "model_id": MODEL_ID,
                            "dataset_track": "2class",
                            "baseline_or_tuned": recipe,
                            "status": "COMPLETE",
                        }
                        for recipe in ("baseline", "tuned")
                    ]
                }
            write_json(path, payload)
        elif path.suffix in {".yaml", ".yml"}:
            write_yaml(path, {"model_id": MODEL_ID})
        else:
            path.write_text("Measured multi-seed result.\n", encoding="utf-8")
    write_json(
        bundle / "bundle_manifest.json",
        {
            "schema_version": 3,
            "result_bundle_id": bundle.name,
            "created_at": "2026-01-01T00:00:00Z",
            "model_id": MODEL_ID,
            "architecture_family": "End-to-end Transformer",
            "dataset_track": "2class",
            "class_names": ["person", "vehicle"],
            "protocol_id": HPO_PROTOCOL_ID,
            "run_ids": [f"run-{index}" for index in range(6)],
            "seeds": [17, 42, 3407],
            "seed_status": "multi-seed",
            "checkpoint_sha256": ["a" * 64] * 6,
            "annotation_sha256": "b" * 64,
            "official_full_train_verified": True,
            "evaluation_git_commit": "c" * 40,
            "training_git_commits": ["d" * 40],
            "generated_files": sorted(HPO_REQUIRED_BUNDLE_FILES),
            "intentionally_excluded_files": ["checkpoints"],
            "export_status": "created",
        },
    )
    assert validate_bundle(bundle) == []
    (bundle / "checkpoint.pth").write_bytes(b"not allowed")
    assert any("checkpoint.pth" in error for error in validate_bundle(bundle))


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _local_remote(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    source = tmp_path / "source"
    subprocess.run(
        ["git", "init", "--bare", str(remote)], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "init", "-b", "main", str(source)],
        check=True,
        capture_output=True,
    )
    _git(source, "config", "user.name", "Test")
    _git(source, "config", "user.email", "test@example.invalid")
    (source / "README.md").write_text("test\n", encoding="utf-8")
    _git(source, "add", "README.md")
    _git(source, "commit", "-m", "initial")
    _git(source, "remote", "add", "origin", str(remote))
    _git(source, "push", "-u", "origin", "main")
    return source, remote


@pytest.mark.parametrize("branch_exists", [False, True])
def test_publication_clone_handles_missing_and_existing_result_branch(
    tmp_path, branch_exists
):
    source, _ = _local_remote(tmp_path)
    if branch_exists:
        _git(source, "switch", "-c", "experiment-results")
        (source / "result.txt").write_text("old\n", encoding="utf-8")
        _git(source, "add", "result.txt")
        _git(source, "commit", "-m", "old result")
        _git(source, "push", "-u", "origin", "experiment-results")
        _git(source, "switch", "main")
    temporary = Path(
        __import__("tempfile").mkdtemp(prefix="visdrone-results-publish-")
    )
    try:
        clone, existed = _prepare_publication_clone(
            source, temporary, dict(__import__("os").environ)
        )
        assert existed is branch_exists
        assert _git(clone, "branch", "--show-current") == "experiment-results"
        assert _git(source, "branch", "--show-current") == "main"
    finally:
        _safe_cleanup(temporary)


def test_publishing_guards_auth_dirty_checkout_and_staged_paths(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="authentication is missing"):
        _authentication_environment(tmp_path)
    source, _ = _local_remote(tmp_path / "git")
    (source / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty source"):
        _validate_source_checkout(source)
    _validate_staged_paths(
        [
            "results/bundles/bundle/README.md",
            LATEST_MANIFEST,
        ],
        "bundle",
    )
    with pytest.raises(RuntimeError, match="Staging safety"):
        _validate_staged_paths(
            ["results/bundles/bundle/README.md", "credentials.txt"],
            "bundle",
        )
    with pytest.raises(RuntimeError, match="Staging safety"):
        _validate_staged_paths(["results/bundles/bundle/README.md"], "bundle")
