from __future__ import annotations

import json
import sys
import types
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


def _fake_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repo"
    (repository / "src").mkdir(parents=True)
    (repository / "src" / "__init__.py").write_text("", encoding="utf-8")
    (repository / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    return repository


def test_local_notebooks_never_install_hosted_pins_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local run must not silently downgrade the developer's interpreter."""
    repository = _fake_repository(tmp_path)
    captured: list[tuple[Path, str | Path | None]] = []
    monkeypatch.setattr(
        notebook_env,
        "_install_shared_dependencies",
        lambda root, requirements: captured.append((root, requirements)),
    )

    resolved = notebook_env.setup_notebook_environment(
        repository,
        platform="local",
        requirements_file="requirements-dataset-colab.txt",
        environ={},
    )

    assert captured == []
    assert resolved.dependencies_installed is False
    assert "never installs hosted pins" in resolved.dependency_decision


def test_local_installation_requires_an_explicit_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _fake_repository(tmp_path)
    (repository / "requirements-dataset-colab.txt").write_text(
        "numpy==1.26.4\n", encoding="utf-8"
    )
    captured: list[tuple[Path, str | Path | None]] = []
    monkeypatch.setattr(
        notebook_env,
        "_install_shared_dependencies",
        lambda root, requirements: captured.append((root, requirements)),
    )
    monkeypatch.setattr(notebook_env, "restart_required_packages", lambda pins: [])

    notebook_env.setup_notebook_environment(
        repository,
        platform="local",
        requirements_file="requirements-dataset-colab.txt",
        environ={notebook_env.LOCAL_INSTALL_OPT_IN: "1"},
    )

    assert captured == [(repository.resolve(), "requirements-dataset-colab.txt")]


@pytest.mark.parametrize("platform", ["colab", "kaggle"])
def test_hosted_runtimes_still_install_their_pins(
    platform: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _fake_repository(tmp_path)
    captured: list[tuple[Path, str | Path | None]] = []
    monkeypatch.setattr(
        notebook_env,
        "_install_shared_dependencies",
        lambda root, requirements: captured.append((root, requirements)),
    )
    monkeypatch.setattr(notebook_env, "_mount_google_drive", lambda *_a, **_k: None)

    notebook_env.setup_notebook_environment(
        repository,
        platform=platform,
        requirements_file=None,
        environ={
            "VISDRONE_DRIVE_ROOT": str(tmp_path / "artifacts"),
            "VISDRONE_LOCAL_CACHE_ROOT": str(tmp_path / "cache"),
            "VISDRONE_MODEL_ENV_ROOT": str(tmp_path / "envs"),
            "VISDRONE_HPO_SCRATCH_ROOT": str(tmp_path / "scratch"),
        },
    )

    assert captured == [(repository.resolve(), None)]


def test_no_restart_when_versions_already_match() -> None:
    assert (
        notebook_env.restart_required_packages(
            {"numpy": "1.26.4"},
            installed={"numpy": "1.26.4"},
            imported_modules=["numpy"],
        )
        == []
    )


def test_no_restart_when_the_changed_package_was_never_imported() -> None:
    assert (
        notebook_env.restart_required_packages(
            {"numpy": "1.26.4"},
            installed={"numpy": "2.1.0"},
            imported_modules=["json"],
        )
        == []
    )


def test_restart_required_when_an_imported_binary_dependency_is_replaced() -> None:
    changes = notebook_env.restart_required_packages(
        {"numpy": "1.26.4", "requests": "2.32.0"},
        installed={"numpy": "2.1.0", "requests": "2.0.0"},
        imported_modules=["numpy", "requests"],
    )

    assert [change["package"] for change in changes] == ["numpy"]
    assert changes[0]["installed"] == "2.1.0"
    assert changes[0]["required"] == "1.26.4"


def test_setup_stops_with_a_restart_instruction_instead_of_continuing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _fake_repository(tmp_path)
    (repository / "pins.txt").write_text("numpy==1.26.4\n", encoding="utf-8")
    monkeypatch.setattr(notebook_env, "_install_shared_dependencies", lambda *_a: None)
    monkeypatch.setattr(
        notebook_env,
        "restart_required_packages",
        lambda pins: [
            {
                "package": "numpy",
                "installed": "2.1.0",
                "required": "1.26.4",
                "imported_module": "numpy",
            }
        ],
    )
    mounted: list[bool] = []
    monkeypatch.setattr(
        notebook_env, "_mount_google_drive", lambda *_a, **_k: mounted.append(True)
    )

    with pytest.raises(notebook_env.KernelRestartRequired, match="RESTART REQUIRED"):
        notebook_env.setup_notebook_environment(
            repository,
            platform="colab",
            requirements_file="pins.txt",
            environ={},
        )

    assert mounted == []


def test_requirements_pins_are_parsed_without_executing_pip() -> None:
    pins = notebook_env.parse_pinned_requirements(
        "# comment\nnumpy==1.26.4\ntorch==2.1.0+cu118 ; sys_platform == 'linux'\n"
        "--extra-index-url https://example.invalid\nrequests>=2\n"
    )

    assert pins == {"numpy": "1.26.4", "torch": "2.1.0+cu118"}


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
        assert "bootstrap_notebook" in source, path.name
        # The Git bootstrap lives in src/notebook_bootstrap.py, not in notebooks.
        assert "rev-parse" not in source, path.name
        assert '(REPO_PATH / ".git")' not in source, path.name


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


def test_readme_model_table_matches_controlled_model_ids() -> None:
    """PR-12: the README model table is exactly the controlled-track models."""
    import re

    from src.config.benchmark_tracks import load_track_config

    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    header = next(
        i for i, line in enumerate(lines) if line.startswith("| `MODEL_ID`")
    )
    table_ids: set[str] = set()
    for line in lines[header + 2 :]:  # skip the header separator row
        if not line.startswith("|"):
            break
        match = re.match(r"\|\s*`([^`]+)`", line)
        if match:
            table_ids.add(match.group(1))
    configured = set(load_track_config(ROOT, "controlled")["model_ids"])
    assert table_ids == configured


def test_rtdetrv2_quarantine_is_lifted_in_the_readme() -> None:
    """PR-04: RT-DETRv2 is a first-class four-family model, not quarantined."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Historical results only" not in readme
    assert "do not start new" not in readme
    # The HPO/final notebooks are advertised as the active RT-DETR entry points.
    assert "13_hpo_rtdetrv2.ipynb" in readme
    assert "23_finetune_rtdetrv2.ipynb" in readme


def test_canonical_notebooks_preserve_the_selected_git_ref() -> None:
    """No notebook fast-forwards the checkout it was pointed at."""
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        source = path.read_text(encoding="utf-8")
        assert '"pull", "--ff-only"' not in source, path.name
        assert "bootstrap_notebook" in source, path.name
    for name in (
        "20_finetune_resnet50.ipynb",
        "21_finetune_swin_t.ipynb",
        "22_finetune_vmamba_t.ipynb",
    ):
        source = (ROOT / "notebooks" / name).read_text(encoding="utf-8")
        assert "requirements-dataset-colab.txt" in source


def test_final_notebooks_expose_full_matrix_without_editing_code() -> None:
    """PR-06's opt-in matrix must be a parameter, not an internal argument."""
    for name in (
        "20_finetune_resnet50.ipynb",
        "21_finetune_swin_t.ipynb",
        "22_finetune_vmamba_t.ipynb",
        "23_finetune_rtdetrv2.ipynb",
    ):
        notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        parameters = "".join(notebook["cells"][1]["source"])
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "FULL_MATRIX = False" in parameters, name
        assert "full_matrix=FULL_MATRIX" in source, name


def test_model_notebooks_guard_the_selected_dataset_track() -> None:
    for name in (
        "10_hpo_resnet50.ipynb",
        "11_hpo_swin_t.ipynb",
        "12_hpo_vmamba_t.ipynb",
        "13_hpo_rtdetrv2.ipynb",
        "20_finetune_resnet50.ipynb",
        "21_finetune_swin_t.ipynb",
        "22_finetune_vmamba_t.ipynb",
        "23_finetune_rtdetrv2.ipynb",
    ):
        source = (ROOT / "notebooks" / name).read_text(encoding="utf-8")
        assert "require_prepared_dataset_track" in source, name


def test_as_dict_reports_the_recorded_restart_decision() -> None:
    """``restart_required`` must come from state, never be a literal.

    ``setup_notebook_environment`` raises rather than returning while a restart
    is pending, so a returned environment records False. Reporting a hardcoded
    False instead would keep printing "no restart needed" for any future caller
    that constructs the environment differently.
    """
    fields = {
        "platform": "colab",
        "repository_root": Path("/repo"),
        "artifact_root": Path("/artifacts"),
        "local_cache_root": Path("/cache"),
        "model_runtime_root": Path("/runtimes"),
        "hpo_scratch_root": Path("/scratch"),
    }
    assert (
        notebook_env.NotebookEnvironment(**fields).as_dict()["restart_required"]
        is False
    )
    assert (
        notebook_env.NotebookEnvironment(
            **fields, restart_required=True
        ).as_dict()["restart_required"]
        is True
    )


def test_unmountable_drive_names_the_options_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colab draft/restricted sessions raise NotImplementedError from mount().

    Silently continuing would put the dataset, the Optuna study, and every
    checkpoint on storage that vanishes at disconnect, so the mount failure has
    to surface as a choice.
    """

    class _Drive:
        @staticmethod
        def mount(_mountpoint: str) -> None:
            raise NotImplementedError(
                "Mounting drive is unsupported in this environment. Use PyDrive2 instead."
            )

    module = types.ModuleType("google.colab")
    module.drive = _Drive  # type: ignore[attr-defined]
    package = types.ModuleType("google")
    package.colab = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", package)
    monkeypatch.setitem(sys.modules, "google.colab", module)

    with pytest.raises(notebook_env.DriveUnavailableError) as failure:
        notebook_env._mount_google_drive("colab", True)

    message = str(failure.value)
    assert "USE_GOOGLE_DRIVE = False" in message
    assert "VISDRONE_DRIVE_ROOT" in message
    assert "DELETED" in message


def test_drive_is_not_mounted_when_it_was_not_requested() -> None:
    # No google.colab in sys.modules: opting out must not even try to import it.
    notebook_env._mount_google_drive("colab", False)
    notebook_env._mount_google_drive("local", True)


@pytest.mark.parametrize(
    ("platform", "root", "persistent"),
    [
        ("colab", "/content/drive/MyDrive/visdrone_architecture_benchmark", True),
        ("colab", "/content/drive", True),
        ("colab", "/content/aerial-object-detection-benchmark/local_artifacts", False),
        ("colab", "/content/visdrone_artifacts", False),
        ("kaggle", "/kaggle/working/visdrone_architecture_benchmark", True),
        ("local", "/home/user/repo/local_artifacts", True),
    ],
)
def test_artifact_root_persistence_classification(
    platform: str, root: str, persistent: bool
) -> None:
    assert notebook_env.artifact_root_is_persistent(platform, root) is persistent


def test_active_notebooks_expose_the_drive_switch() -> None:
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        if path.name.startswith(("01_", "02_", "03_", "00_bootstrap")):
            continue  # retired protocol notebooks
        source = path.read_text(encoding="utf-8")
        assert "USE_GOOGLE_DRIVE" in source, path.name
        assert "use_google_drive=True," not in source, path.name
