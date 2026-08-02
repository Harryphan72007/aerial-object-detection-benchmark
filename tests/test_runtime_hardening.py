from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import src.hpo.workflow as hpo_workflow
import src.hpo.rtdetr_v2 as rtdetr_hpo
import src.training.callbacks as callbacks
from src.hpo.rtdetr_v2 import RTDetrOptunaV2
from src.optional_outputs import load_optional_warnings, run_optional_output
from src.subprocess_utils import build_model_subprocess_environment
from src.training.trainer import TrainingOrchestrator

ROOT = Path(__file__).resolve().parents[1]
INLINE_BACKEND = "module://matplotlib_inline.backend_inline"


def _critical_artifacts(run_dir: Path) -> dict[str, object]:
    checkpoints: dict[str, str] = {}
    for field, filename in (
        ("checkpoint_best_map", "best_map.pth"),
        ("checkpoint_best_aptiny", "best_aptiny.pth"),
        ("checkpoint_last", "last.pth"),
    ):
        path = run_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"valid-checkpoint")
        checkpoints[field] = str(path)
    summary: dict[str, object] = {
        **checkpoints,
        "best_validation_map": 0.25,
        "best_validation_aptiny": 0.10,
    }
    (run_dir / "final_metrics.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return summary


def test_model_subprocess_environment_replaces_notebook_backend_and_rank_state(
    tmp_path: Path,
) -> None:
    parent = {
        **os.environ,
        "MPLBACKEND": INLINE_BACKEND,
        "IPYTHONDIR": "/notebook/ipython",
        "JUPYTER_PATH": "/notebook/jupyter",
        "DISPLAY": ":99",
        "PYTHONPATH": "/notebook/packages",
        "WORLD_SIZE": "8",
        "LOCAL_RANK": "3",
        "RANK": "3",
        "CUBLAS_WORKSPACE_CONFIG": "invalid",
    }
    child = build_model_subprocess_environment(
        parent, matplotlib_config_dir=tmp_path / "mpl"
    )

    assert child["MPLBACKEND"] == "Agg"
    assert child["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert child["MPLCONFIGDIR"] == str((tmp_path / "mpl").resolve())
    for variable in (
        "IPYTHONDIR",
        "JUPYTER_PATH",
        "DISPLAY",
        "PYTHONPATH",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "RANK",
    ):
        assert variable not in child

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; import matplotlib.pyplot as plt; "
                "assert os.environ['MPLBACKEND'] == 'Agg'; "
                "assert plt.get_backend().lower() == 'agg'"
            ),
        ],
        env=child,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_exact_inline_backend_plot_failure_is_nonfatal_and_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    summary = _critical_artifacts(run_dir)

    def fail_plot(*_args: object, **_kwargs: object) -> None:
        raise ValueError(
            f"Key backend: {INLINE_BACKEND!r} is not a valid value for backend"
        )

    monkeypatch.setattr(callbacks, "save_training_curves", fail_plot)
    warning = callbacks.safe_save_training_curves(
        [{"epoch": 1, "loss": 1.0}],
        run_dir / "training_curves.png",
        warning_root=run_dir,
    )

    assert warning is not None
    assert warning["operation"] == "save_training_curves"
    assert warning["exception_type"] == "ValueError"
    assert warning["scientific_artifacts_valid"] is True
    assert not (run_dir / "training_curves.png").exists()
    assert TrainingOrchestrator._load_backend_summary(run_dir) == summary
    assert load_optional_warnings(run_dir)[0]["operation"] == "save_training_curves"


def test_backend_completion_requires_metrics_and_checkpoints(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="required final metrics"):
        TrainingOrchestrator._load_backend_summary(tmp_path)


def test_direct_backend_rejects_inherited_distributed_world_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "8")
    orchestrator = object.__new__(TrainingOrchestrator)
    orchestrator.repo_root = ROOT
    orchestrator.paths = type(
        "Paths",
        (),
        {"root": tmp_path},
    )()
    with pytest.raises(ValueError, match="single-process"):
        # Reach the check without invoking a backend by replacing preflight helpers.
        monkeypatch.setattr("src.training.trainer.validate_drive_writable", lambda *_: None)
        monkeypatch.setattr("src.training.trainer.validate_dataset", lambda *_: None)
        orchestrator.run(
            "rtdetrv2_l",
            "2class",
            640,
            1,
            8,
            1,
            42,
        )
    (tmp_path / "final_metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="checkpoint_best_map"):
        TrainingOrchestrator._load_backend_summary(tmp_path)


def test_optuna_snapshot_failure_warns_without_invalidating_primary_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = RTDetrOptunaV2(ROOT, tmp_path, "2class")
    workflow.study_path.parent.mkdir(parents=True, exist_ok=True)
    workflow.study_path.write_bytes(b"primary-study-remains")

    def fail_snapshot(*_args: object, **_kwargs: object) -> Path:
        raise OSError("temporary Drive disconnect")

    monkeypatch.setattr(rtdetr_hpo, "snapshot_sqlite_database", fail_snapshot)
    workflow._after_trial(object())

    assert workflow.study_path.read_bytes() == b"primary-study-remains"
    warning = load_optional_warnings(workflow.root)[0]
    assert warning["operation"] == "snapshot_optuna_database"
    assert warning["scientific_artifacts_valid"] is True


def test_completed_hpo_trial_with_plot_warning_is_not_repeated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("optuna")
    monkeypatch.setattr(hpo_workflow, "PHASE_TRIALS", 1)
    calls: list[int] = []

    def trial_runner(
        phase: str,
        trial_number: int,
        parameters: dict[str, object],
        run_dir: Path,
    ) -> tuple[float, float]:
        del phase, parameters
        calls.append(trial_number)
        _critical_artifacts(run_dir)

        def fail_plot() -> None:
            raise ValueError(
                f"Key backend: {INLINE_BACKEND!r} is not a valid value for backend"
            )

        run_optional_output("save_training_curves", run_dir, fail_plot)
        return 0.25, 0.10

    workflow = RTDetrOptunaV2(
        ROOT, tmp_path, "2class", trial_runner=trial_runner
    )
    metadata = workflow._metadata(
        {"hashes": {"train": "a", "validation": "b"}},
        workflow._broad_search_space(),
    )
    study = workflow._study(metadata)
    workflow._run_phase(study, "phase_a", workflow._broad_search_space())
    workflow._run_phase(study, "phase_a", workflow._broad_search_space())

    assert calls == [0]
    assert study.trials[0].state.name == "COMPLETE"
    assert study.trials[0].user_attrs["trial_status"] == "COMPLETE"
    run_dir = Path(study.trials[0].user_attrs["run_dir"])
    assert (run_dir / "final_metrics.json").is_file()
    assert (run_dir / "best_map.pth").is_file()
    assert load_optional_warnings(run_dir)[0]["operation"] == "save_training_curves"
