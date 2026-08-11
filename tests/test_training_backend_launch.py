import subprocess
import sys
import json
import hashlib
from pathlib import Path

import pytest

from src import subprocess_utils
from src.subprocess_utils import (
    MODEL_PYTHON_ENV,
    EnvironmentNotProvisionedError,
    python_module_command,
)
from src.training.trainer import TrainingOrchestrator


def _write_backend_contract(run_dir: Path, model_id: str) -> None:
    best = run_dir / "best.pth"
    resume = run_dir / "last.pth"
    best.write_bytes(b"checkpoint")
    resume.write_bytes(b"resume")
    identity = {
        "run_id": run_dir.name,
        "model_id": model_id,
        "seed": 42,
        "configuration_hash": "config-hash",
        "epoch": 1,
        "selection_metric": "validation_mAP",
        "selection_metric_value": 0.25,
        "weight_variant": "raw",
    }
    (run_dir / "training_config.yaml").write_text(
        f"model_id: {model_id}\nseed: 42\nconfiguration_hash: config-hash\n",
        encoding="utf-8",
    )
    (run_dir / "final_metrics.json").write_text(
        json.dumps(
            {
                "checkpoint_best": str(best),
                "checkpoint_resume": str(resume),
                "checkpoint_sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
                "checkpoint_identity": identity,
                "checkpoint_load_verified": True,
                "best_validation_map": 0.25,
                "best_validation_aptiny": 0.10,
            }
        ),
        encoding="utf-8",
    )


def test_rtdetr_backend_is_launched_as_repository_module(
    tmp_path: Path, monkeypatch
) -> None:
    # The backend always runs in the isolated model runtime; here the host
    # interpreter stands in for one that has been provisioned.
    monkeypatch.setenv(MODEL_PYTHON_ENV, sys.executable)
    orchestrator = object.__new__(TrainingOrchestrator)
    orchestrator.repo_root = tmp_path
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    _write_backend_contract(run_dir, "rtdetrv2_l")

    launched: dict[str, object] = {}

    def capture(command: list[str], cwd: Path, log_path: Path) -> None:
        launched.update(command=command, cwd=cwd, log_path=log_path)

    orchestrator._run_backend_process = capture
    result = orchestrator._run_rtdetr(
        run_dir,
        {
            "model_id": "rtdetrv2_l",
            "pretrained_model_name_or_path": "example/model",
            "pretrained_revision": "abc123",
        },
        {
            "epochs": 1,
            "scheduler_horizon": 1,
            "validation_interval": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "image_size": 640,
            "seed": 42,
            "overrides": {},
            "use_amp": False,
            "run_kind": "adapter_smoke_non_promotable",
        },
        tmp_path / "train.json",
        tmp_path / "val.json",
        tmp_path / "train-images",
        tmp_path / "val-images",
    )

    assert result["best_validation_map"] == 0.25
    assert launched["command"][:3] == [
        sys.executable,
        "-m",
        "scripts.run_rtdetr_training",
    ]
    assert launched["cwd"] == tmp_path
    assert launched["log_path"] == run_dir / "logs" / "backend.log"


def test_mmdetection_backend_is_launched_as_repository_module(
    tmp_path: Path, monkeypatch
) -> None:
    orchestrator = object.__new__(TrainingOrchestrator)
    orchestrator.repo_root = tmp_path
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "dataset_config.yaml").write_text(
        "class_names:\n  - person\n  - vehicle\n", encoding="utf-8"
    )
    upstream_root = tmp_path / "mmdetection"
    upstream_config = upstream_root / "configs" / "model.py"
    upstream_config.parent.mkdir(parents=True)
    upstream_config.write_text("", encoding="utf-8")
    monkeypatch.setenv("TEST_MMDET_ROOT", str(upstream_root))
    monkeypatch.setenv(MODEL_PYTHON_ENV, sys.executable)
    _write_backend_contract(run_dir, "example")

    launched: dict[str, object] = {}

    def capture(command: list[str], cwd: Path, log_path: Path) -> None:
        launched.update(command=command, cwd=cwd, log_path=log_path)

    orchestrator._run_backend_process = capture
    result = orchestrator._run_mmdetection(
        run_dir,
        {
            "model_id": "example",
            "external_root_env": "TEST_MMDET_ROOT",
            "framework_config": "configs/model.py",
            "allow_scratch_without_pretrained": True,
        },
        {
            "epochs": 1,
            "scheduler_horizon": 1,
            "validation_interval": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "image_size": 640,
            "seed": 42,
            "overrides": {},
            "use_amp": False,
            "run_kind": "adapter_smoke_non_promotable",
        },
        tmp_path / "train.json",
        tmp_path / "val.json",
        tmp_path / "train-images",
        tmp_path / "val-images",
    )

    assert result["best_validation_map"] == 0.25
    assert launched["command"][:3] == [
        sys.executable,
        "-m",
        "scripts.run_mmdetection",
    ]
    assert launched["cwd"] == tmp_path


@pytest.mark.parametrize(
    ("module", "expected_flag"),
    [
        ("scripts.run_rtdetr_training", "--run-dir"),
        ("scripts.run_mmdetection", "--base-config"),
        ("scripts.evaluate", "--drive-root"),
        ("scripts.profile_model", "--drive-root"),
    ],
)
def test_backend_module_entrypoint_can_import_project(
    module: str, expected_flag: str
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    command = python_module_command(module, "--help", host_interpreter=True)

    completed = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert expected_flag in completed.stdout


@pytest.mark.parametrize("module", ["scripts.evaluate", "scripts.profile_model"])
def test_downstream_stage_is_launched_as_module_in_the_model_runtime(
    module: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(MODEL_PYTHON_ENV, sys.executable)
    launched: dict[str, object] = {}

    def capture(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        environment_name: str,
        stage: str,
        python_executable: str,
    ) -> None:
        launched.update(
            command=command,
            cwd=cwd,
            env=env,
            environment_name=environment_name,
            stage=stage,
            python_executable=python_executable,
        )

    monkeypatch.setattr(subprocess_utils, "run_checked", capture)
    subprocess_utils.run_module_in_model_runtime(tmp_path, module, "--help")

    assert launched["command"] == [sys.executable, "-m", module, "--help"]
    assert launched["cwd"] == tmp_path
    assert launched["env"]["MPLBACKEND"] == "Agg"
    assert launched["environment_name"] == "model runtime"
    assert launched["stage"] == module
    assert launched["python_executable"] == sys.executable


@pytest.mark.parametrize(
    "invalid", ["scripts/evaluate.py", "scripts\\evaluate.py", "evaluate.py", ""]
)
def test_repository_module_command_rejects_filepaths(invalid: str) -> None:
    with pytest.raises(ValueError, match="dotted module names"):
        python_module_command(invalid)


def test_module_command_fails_closed_without_a_provisioned_runtime(monkeypatch) -> None:
    """A missing model environment must never fall back to this interpreter."""
    monkeypatch.delenv(MODEL_PYTHON_ENV, raising=False)

    with pytest.raises(EnvironmentNotProvisionedError, match="ensure_model_environment"):
        python_module_command("scripts.run_mmdetection")

    # Repository-management entry points may still opt into the host explicitly.
    assert python_module_command("scripts.evaluate", host_interpreter=True)[0] == (
        sys.executable
    )


def test_module_command_rejects_a_selected_but_missing_interpreter(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(MODEL_PYTHON_ENV, str(tmp_path / "bin" / "python"))

    with pytest.raises(EnvironmentNotProvisionedError, match="missing interpreter"):
        python_module_command("scripts.run_rtdetr_training")
