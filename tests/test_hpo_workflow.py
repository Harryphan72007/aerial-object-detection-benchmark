from __future__ import annotations

import json
import copy
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.hpo.final_workflow import (
    FINAL_SEEDS,
    FULL_MATRIX_SEEDS,
    FinalExperimentWorkflow,
    configuration_hash,
)
from src.hpo.search_spaces import broad_search_space, refined_search_space
from src.hpo.workflow import (
    HPO_PROTOCOL_ID,
    OBJECTIVE_CONTRACT_VERSION,
    TwoStageRandomHPO,
    _failure_kind,
    source_tree_fingerprint,
    validated_objective_pair,
)
from src.result_export import HPO_REQUIRED_BUNDLE_FILES, validate_bundle
from src.training.trainer import TrainingOrchestrator
from src.training.checkpointing import resolve_manifest_checkpoint
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


def _coco_dataset(image_count: int, *, validation: bool = False) -> dict[str, Any]:
    images = [
        {
            "id": index + 1,
            "file_name": f"{'val' if validation else 'train'}_{index:04d}.jpg",
            "width": 100,
            "height": 100,
        }
        for index in range(image_count)
    ]
    annotations = []
    annotation_id = 1
    for index, image in enumerate(images):
        if index % 10 == 0:
            continue
        cats = (1, 2) if index % 3 == 0 else (1 if index % 2 else 2,)
        for offset, category_id in enumerate(cats):
            width = 8 if offset == 0 else 20
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image["id"],
                    "category_id": category_id,
                    "bbox": [1, 1, width, width],
                    "area": width * width,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "vehicle"}],
    }


def _prepare_official_dataset(workflow: "FinalExperimentWorkflow") -> None:
    """Write a tiny but valid official train/val so the held-out selection
    manifests (PR-05) can be built by the final workflow."""
    annotation_root = workflow.paths.coco(workflow.dataset_track) / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    write_json(annotation_root / "instances_train.json", _coco_dataset(60))
    write_json(
        annotation_root / "instances_val.json", _coco_dataset(12, validation=True)
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
    assert set(refined) == {"learning_rate"}
    assert set(broad) == {"learning_rate"}
    assert broad["learning_rate"]["low"] == 1e-6
    assert broad["learning_rate"]["high"] == 5e-4


def test_rtdetr_divergence_is_pruned_and_missing_successes_are_replaced(tmp_path):
    optuna = pytest.importorskip("optuna")
    workflow = object.__new__(TwoStageRandomHPO)
    workflow.model_id = MODEL_ID
    workflow.root = tmp_path
    calls = 0

    def runner(phase, trial_number, parameters, run_dir):
        nonlocal calls
        calls += 1
        assert set(parameters) == {"learning_rate"}
        assert run_dir.is_dir()
        if calls == 1:
            raise ValueError("boxes1 must be valid, but got NaN values")
        return 0.1 + calls / 1000, 0.05

    workflow.trial_runner = runner
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=42),
        directions=["maximize", "maximize"],
    )
    workflow._run_phase(study, "phase_a", broad_search_space(MODEL_ID))

    assert calls == 6
    assert sum(trial.state.name == "COMPLETE" for trial in study.trials) == 5
    pruned = next(trial for trial in study.trials if trial.state.name == "PRUNED")
    assert pruned.user_attrs["diverged"] is True
    assert pruned.user_attrs["trial_status"] == "PRUNED"
    assert pruned.user_attrs["failure_type"] == "ValueError"
    assert pruned.user_attrs["divergence_learning_rate"] == pruned.params[
        "learning_rate"
    ]
    assert all(trial.user_attrs["resume"] is False for trial in study.trials)


def test_unexpected_hpo_error_is_not_pruned(tmp_path):
    optuna = pytest.importorskip("optuna")
    workflow = object.__new__(TwoStageRandomHPO)
    workflow.model_id = MODEL_ID
    workflow.root = tmp_path

    def runner(*_args):
        raise RuntimeError("implementation contract is broken")

    workflow.trial_runner = runner
    study = optuna.create_study(directions=["maximize", "maximize"])
    with pytest.raises(RuntimeError, match="implementation contract"):
        workflow._run_phase(study, "phase_a", broad_search_space(MODEL_ID))
    assert study.trials[0].state.name == "FAIL"
    assert study.trials[0].user_attrs["trial_status"] == "FAILED"


def test_failure_classifier_recognizes_cuda_oom_and_numerical_markers():
    assert _failure_kind(RuntimeError("CUDA out of memory")) == "out_of_memory"
    assert (
        _failure_kind(ValueError("boxes2 must be valid; got infinite values"))
        == "numerical_divergence"
    )


def test_smoke_and_full_hpo_storage_are_isolated(tmp_path, monkeypatch):
    monkeypatch.delenv("SMOKE_TEST", raising=False)
    full = TwoStageRandomHPO(ROOT, tmp_path, MODEL_ID, "2class")
    monkeypatch.setenv("SMOKE_TEST", "1")
    smoke = TwoStageRandomHPO(ROOT, tmp_path, MODEL_ID, "2class")

    assert full.study_path.name == "study.db"
    assert smoke.study_path.name == "study_smoke.db"
    assert smoke.study_path.parent == full.root / "smoke_test"
    assert smoke.study_path != full.study_path


def test_hpo_rejects_scratch_storage_inside_drive(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "VISDRONE_HPO_SCRATCH_ROOT", str(tmp_path / "drive" / "scratch")
    )
    with pytest.raises(ValueError, match="must not be inside Google Drive"):
        TwoStageRandomHPO(ROOT, tmp_path / "drive", MODEL_ID, "2class")


def test_legacy_checkpoint_resolver_does_not_materialize_aliases(tmp_path):
    (tmp_path / "best_map.pth").write_bytes(b"best")
    selected = resolve_manifest_checkpoint(
        {"run_dir": str(tmp_path)}, allow_legacy_aliases=True
    )
    assert selected.name == "best_map.pth"
    assert (tmp_path / "best_map.pth").read_bytes() == b"best"
    assert not (tmp_path / "best.pth").exists()
    assert not (tmp_path / "best.pt").exists()


def test_final_workflow_default_matrix_is_one_recipe_one_seed(tmp_path):
    """PR-06: the headline matrix is a single tuned recipe at a single seed."""
    _best_config(tmp_path)
    fake = FakeOrchestrator()
    workflow = FinalExperimentWorkflow(
        ROOT, tmp_path, MODEL_ID, "2class", orchestrator=fake
    )
    _prepare_official_dataset(workflow)
    result = workflow.run(start_expensive_stage=True)
    assert len(result["runs"]) == 1
    assert FINAL_SEEDS == (42,)
    assert {call["seed"] for call in fake.calls} == {42}
    assert {call["baseline_or_tuned"] for call in fake.calls} == {"tuned"}
    assert all(call["protocol_id"] == HPO_PROTOCOL_ID for call in fake.calls)


def test_final_workflow_full_matrix_override_runs_two_recipes_three_seeds(tmp_path):
    """The full baseline+tuned x multi-seed matrix stays reachable via opt-in."""
    _best_config(tmp_path)
    fake = FakeOrchestrator()
    workflow = FinalExperimentWorkflow(
        ROOT, tmp_path, MODEL_ID, "2class", orchestrator=fake
    )
    _prepare_official_dataset(workflow)
    result = workflow.run(start_expensive_stage=True, full_matrix=True)
    assert len(result["runs"]) == 6
    assert {call["seed"] for call in fake.calls} == set(FULL_MATRIX_SEEDS)
    assert {call["baseline_or_tuned"] for call in fake.calls} == {
        "baseline",
        "tuned",
    }
    # PR-05: final training uses the held-out final-train split and selects
    # best.pth on the model-selection split, never on official validation.
    assert all(
        call["train_annotation_override"].name == "final_train_seed42.json"
        for call in fake.calls
    )
    assert all(
        call["validation_annotation_override"].name == "model_selection_seed42.json"
        for call in fake.calls
    )
    # The selection images come from train, not the official validation images.
    assert all(
        call["validation_images_override"] == workflow.paths.images("train")
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


def test_source_tree_fingerprint_captures_uncommitted_implementation_changes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "implementation.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    before = source_tree_fingerprint(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert source_tree_fingerprint(tmp_path) != before


def test_non_rtdetr_hpo_refuses_legacy_or_missing_objective_pairs() -> None:
    with pytest.raises(RuntimeError, match="all-zero"):
        validated_objective_pair(
            {"best_validation_map": 0.0, "best_validation_aptiny": 0.0},
            "faster_rcnn_resnet50",
        )
    with pytest.raises(RuntimeError, match="required HPO objectives"):
        validated_objective_pair(
            {"best_validation_map": 0.2}, "faster_rcnn_resnet50"
        )
    assert validated_objective_pair(
        {"best_validation_map": 0.2, "best_validation_aptiny": 0.1},
        "faster_rcnn_resnet50",
    ) == (0.2, 0.1)


def test_non_rtdetr_study_rejects_stale_source_contract_without_deleting_db(
    tmp_path: Path,
) -> None:
    pytest.importorskip("optuna")
    workflow = TwoStageRandomHPO(
        ROOT, tmp_path, "faster_rcnn_resnet50", "2class"
    )
    space = broad_search_space("faster_rcnn_resnet50")
    metadata = workflow._metadata({"hashes": {"train": "fixture"}}, space)
    workflow._study(metadata)
    changed = copy.deepcopy(metadata)
    changed["source_tree_fingerprint"] = "0" * 64

    with pytest.raises(RuntimeError, match="Archive the old study"):
        workflow._study(changed)

    assert workflow.study_path.is_file()


def test_non_rtdetr_final_contract_binds_source_environment_and_dataset(
    tmp_path: Path,
) -> None:
    model_id = "faster_rcnn_resnet50"
    workflow = FinalExperimentWorkflow(
        ROOT, tmp_path, model_id, "2class", orchestrator=FakeOrchestrator()
    )
    write_yaml(
        workflow.best_config_path,
        {
            "model_id": model_id,
            "dataset_track": "2class",
            "protocol_id": HPO_PROTOCOL_ID,
            "search_seed": 42,
            "objective_contract_version": OBJECTIVE_CONTRACT_VERSION,
            "parameters": {"learning_rate": 0.0001},
        },
    )
    annotation_root = workflow.paths.coco("2class") / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    (annotation_root / "instances_train.json").write_text("{}", encoding="utf-8")
    (annotation_root / "instances_val.json").write_text("{}", encoding="utf-8")
    parameters = workflow._load_tuned_parameters()

    contract = workflow._contract(42, "tuned", parameters, 1, 8)

    assert contract["objective_contract_version"] == OBJECTIVE_CONTRACT_VERSION
    assert len(contract["source_tree_fingerprint"]) == 64
    assert len(contract["dataset_hashes"]["train"]) == 64
    assert contract["evaluation_policy"]["max_detections_per_image"] == 500
    assert contract["environment_fingerprint"]

    old_run = workflow.paths.final_checkpoints / model_id / "old-contract"
    write_json(old_run / "resume_contract.json", contract)
    (old_run / "last.pth").write_bytes(b"resume")
    (annotation_root / "instances_val.json").write_text(
        '{"changed": true}', encoding="utf-8"
    )
    changed_contract = workflow._contract(42, "tuned", parameters, 1, 8)
    _, selected_run, resume = workflow._resumable(changed_contract)
    assert changed_contract["dataset_hashes"] != contract["dataset_hashes"]
    assert selected_run != old_run
    assert resume is None
    assert (old_run / "last.pth").read_bytes() == b"resume"


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
        for seed in FULL_MATRIX_SEEDS:
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
    result = aggregate_hpo_results(
        tmp_path, "2class", required_seeds=FULL_MATRIX_SEEDS
    )
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
