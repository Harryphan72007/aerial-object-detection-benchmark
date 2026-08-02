import subprocess
import sys
import json
from pathlib import Path

import pytest

from src.subprocess_utils import python_module_command
from src.training.trainer import TrainingOrchestrator
from src.workflows import model_day


def _write_backend_contract(run_dir: Path) -> None:
    checkpoints = {}
    for field, filename in (
        ("checkpoint_best_map", "best_map.pth"),
        ("checkpoint_best_aptiny", "best_aptiny.pth"),
        ("checkpoint_last", "last.pth"),
    ):
        path = run_dir / filename
        path.write_bytes(b"checkpoint")
        checkpoints[field] = str(path)
    (run_dir / "final_metrics.json").write_text(
        json.dumps(
            {
                **checkpoints,
                "best_validation_map": 0.25,
                "best_validation_aptiny": 0.10,
            }
        ),
        encoding="utf-8",
    )


def test_rtdetr_backend_is_launched_as_repository_module(tmp_path: Path) -> None:
    orchestrator = object.__new__(TrainingOrchestrator)
    orchestrator.repo_root = tmp_path
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    _write_backend_contract(run_dir)

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
    _write_backend_contract(run_dir)

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
    command = python_module_command(module, "--help")

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
def test_model_day_downstream_stage_is_launched_as_module(
    module: str, tmp_path: Path, monkeypatch
) -> None:
    launched: dict[str, object] = {}

    def capture(
        command: list[str], *, check: bool, cwd: Path, env: dict[str, str]
    ) -> None:
        launched.update(command=command, check=check, cwd=cwd, env=env)

    monkeypatch.setattr(model_day.subprocess, "run", capture)
    model_day._run_module(tmp_path, module, "--help")

    assert launched["command"] == [sys.executable, "-m", module, "--help"]
    assert launched["check"] is True
    assert launched["cwd"] == tmp_path
    assert launched["env"]["MPLBACKEND"] == "Agg"


@pytest.mark.parametrize(
    "invalid", ["scripts/evaluate.py", "scripts\\evaluate.py", "evaluate.py", ""]
)
def test_repository_module_command_rejects_filepaths(invalid: str) -> None:
    with pytest.raises(ValueError, match="dotted module names"):
        python_module_command(invalid)
