from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import notebook_environment as notebook_env
from src.utils.serialization import read_yaml
from src.workflows import environment as model_environment

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MODELS = (
    "faster_rcnn_resnet50",
    "faster_rcnn_swin_t",
    "faster_rcnn_vmamba_t",
)


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"COLAB_RELEASE_TAG": "release"}, "colab"),
        ({"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, "kaggle"),
        ({}, "local"),
        ({"VISDRONE_NOTEBOOK_PLATFORM": "kaggle"}, "kaggle"),
    ],
)
def test_platform_detection(environment: dict[str, str], expected: str) -> None:
    assert notebook_env.detect_notebook_platform(environment) == expected


def test_platform_defaults_are_writable_host_locations(tmp_path: Path) -> None:
    assert notebook_env.default_repository_root("colab") == Path(
        "/content/aerial-object-detection-benchmark"
    )
    assert notebook_env.default_repository_root("kaggle") == Path(
        "/kaggle/working/aerial-object-detection-benchmark"
    )
    assert notebook_env.default_artifact_root("kaggle", tmp_path) == Path(
        "/kaggle/working/visdrone_architecture_benchmark"
    )
    assert notebook_env.default_local_cache_root("kaggle", tmp_path) == Path(
        "/kaggle/working/visdrone_cache"
    )
    assert notebook_env.default_model_runtime_root("kaggle") == Path(
        "/kaggle/working/visdrone_model_envs"
    )
    assert notebook_env.default_hpo_scratch_root("kaggle") == Path(
        "/kaggle/working/visdrone_hpo_trials"
    )
    assert notebook_env.default_repository_root("local", tmp_path) == (
        tmp_path / "aerial-object-detection-benchmark"
    )


def test_git_worktree_probe_accepts_the_current_checkout() -> None:
    assert notebook_env.is_git_worktree(ROOT) is True


def test_setup_exports_one_path_contract_without_installing(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    (repository / "src").mkdir(parents=True)
    (repository / "src" / "__init__.py").write_text("", encoding="utf-8")
    (repository / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    environment = {
        "VISDRONE_DRIVE_ROOT": str(tmp_path / "artifacts"),
        "VISDRONE_LOCAL_CACHE_ROOT": str(tmp_path / "cache"),
        "VISDRONE_MODEL_ENV_ROOT": str(tmp_path / "model-envs"),
        "VISDRONE_HPO_SCRATCH_ROOT": str(tmp_path / "hpo-scratch"),
    }
    resolved = notebook_env.setup_notebook_environment(
        repository,
        platform="kaggle",
        install_dependencies=False,
        environ=environment,
    )
    assert resolved.platform == "kaggle"
    assert resolved.hosted is True
    assert resolved.artifact_root == (tmp_path / "artifacts").resolve()
    assert resolved.local_cache_root == (tmp_path / "cache").resolve()
    assert environment["BENCHMARK_REPO_ROOT"] == str(repository.resolve())
    assert environment["VISDRONE_NOTEBOOK_PLATFORM"] == "kaggle"
    assert all(
        path.is_dir()
        for path in (
            resolved.artifact_root,
            resolved.local_cache_root,
            resolved.model_runtime_root,
            resolved.hpo_scratch_root,
        )
    )


def test_explicit_requirements_are_installed_for_local_notebooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    (repository / "src").mkdir(parents=True)
    (repository / "src" / "__init__.py").write_text("", encoding="utf-8")
    (repository / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    captured: list[tuple[Path, str | Path | None]] = []
    monkeypatch.setattr(
        notebook_env,
        "_install_shared_dependencies",
        lambda root, requirements: captured.append((root, requirements)),
    )

    notebook_env.setup_notebook_environment(
        repository,
        platform="local",
        requirements_file="requirements-dataset-colab.txt",
        environ={},
    )

    assert captured == [
        (repository.resolve(), "requirements-dataset-colab.txt")
    ]


def test_shared_editable_install_disables_networked_build_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    requirement = repository / "requirements.txt"
    requirement.parent.mkdir(parents=True)
    requirement.write_text("", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(
        notebook_env.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(command),
    )

    notebook_env._install_shared_dependencies(repository, "requirements.txt")

    assert commands[-1][-2:] == ["--no-deps", "--no-build-isolation"]


def test_kaggle_uses_existing_isolated_model_provisioner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "model-envs"
    captured: dict[str, object] = {}

    def fake_provision(model_id, repo_root, drive_root, *, runtime_base):
        captured.update(
            model_id=model_id,
            repo_root=Path(repo_root),
            drive_root=Path(drive_root),
            runtime_base=Path(runtime_base),
        )
        return {"family": "openmmlab", "state": "READY"}

    monkeypatch.setattr(model_environment, "in_hosted_notebook", lambda: True)
    monkeypatch.setattr(
        model_environment, "detect_notebook_platform", lambda: "kaggle"
    )
    monkeypatch.setattr(
        model_environment, "provision_isolated_environment", fake_provision
    )
    monkeypatch.setenv("VISDRONE_MODEL_ENV_ROOT", str(runtime_root))
    result = model_environment.ensure_model_environment(
        "faster_rcnn_resnet50", ROOT, tmp_path / "artifacts"
    )
    assert result["state"] == "READY"
    assert captured["runtime_base"] == runtime_root


def test_all_canonical_notebooks_share_the_cross_platform_setup() -> None:
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "setup_notebook_environment" in source, path.name
        assert "IN_KAGGLE" in source, path.name


def test_active_model_and_epoch_contracts_remain_bounded() -> None:
    workflow = (ROOT / "src" / "hpo" / "workflow.py").read_text(encoding="utf-8")
    final = (ROOT / "src" / "hpo" / "final_workflow.py").read_text(
        encoding="utf-8"
    )
    # The protocol is defined once in configs/controlled/benchmark.yaml; both
    # workflows resolve it rather than hardcoding epoch/batch/resolution constants.
    assert "resolve_controlled_protocol" in workflow
    assert "resolve_controlled_protocol" in final
    protocol = read_yaml(
        ROOT / "configs" / "controlled" / "benchmark.yaml"
    )["protocol"]
    assert protocol["phase_a_epochs"] == 3
    assert protocol["final_train_epochs"] == 8
    for model_id in ACTIVE_MODELS:
        assert model_id in (
            ROOT / "src" / "hpo" / "search_spaces.py"
        ).read_text(encoding="utf-8")
    model_day = json.loads(
        (ROOT / "notebooks" / "01_run_model_day.ipynb").read_text(encoding="utf-8")
    )
    parameters = "".join(model_day["cells"][1]["source"])
    assert 'MODEL_ID = "faster_rcnn_resnet50"' in parameters
    for name, flag in (
        ("13_hpo_rtdetrv2.ipynb", "START_HPO = False"),
        ("23_finetune_rtdetrv2.ipynb", "START_FINETUNING = False"),
    ):
        notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        assert flag in "".join(notebook["cells"][1]["source"])


def test_rtdetrv2_quarantine_is_lifted_in_the_readme() -> None:
    """PR-04: RT-DETRv2 is a first-class four-family model, not quarantined."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Historical results only" not in readme
    assert "do not start new" not in readme
    # The HPO/final notebooks are advertised as the active RT-DETR entry points.
    assert "13_hpo_rtdetrv2.ipynb" in readme
    assert "23_finetune_rtdetrv2.ipynb" in readme


def test_non_rtdetr_canonical_notebooks_preserve_the_selected_git_ref() -> None:
    selected = (
        "00_prepare_visdrone.ipynb",
        "10_hpo_resnet50.ipynb",
        "11_hpo_swin_t.ipynb",
        "12_hpo_vmamba_t.ipynb",
        "20_finetune_resnet50.ipynb",
        "21_finetune_swin_t.ipynb",
        "22_finetune_vmamba_t.ipynb",
    )
    for name in selected:
        source = (ROOT / "notebooks" / name).read_text(encoding="utf-8")
        assert '"pull", "--ff-only"' not in source
        assert "rev-parse" in source
    for name in (
        "20_finetune_resnet50.ipynb",
        "21_finetune_swin_t.ipynb",
        "22_finetune_vmamba_t.ipynb",
    ):
        source = (ROOT / "notebooks" / name).read_text(encoding="utf-8")
        assert "requirements-dataset-colab.txt" in source
