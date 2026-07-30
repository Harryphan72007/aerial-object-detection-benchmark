import subprocess
import sys
from pathlib import Path

from src.training.trainer import TrainingOrchestrator


def test_rtdetr_backend_is_launched_as_repository_module(tmp_path: Path) -> None:
    orchestrator = object.__new__(TrainingOrchestrator)
    orchestrator.repo_root = tmp_path
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "final_metrics.json").write_text("{}", encoding="utf-8")

    launched: dict[str, object] = {}

    def capture(command: list[str], cwd: Path, log_path: Path) -> None:
        launched.update(command=command, cwd=cwd, log_path=log_path)

    orchestrator._run_backend_process = capture
    result = orchestrator._run_rtdetr(
        run_dir,
        {"pretrained_model_name_or_path": "example/model"},
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

    assert result == {}
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

    assert result == {}
    assert launched["command"][:3] == [
        sys.executable,
        "-m",
        "scripts.run_mmdetection",
    ]
    assert launched["cwd"] == tmp_path


def test_rtdetr_module_entrypoint_can_import_project() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "scripts.run_rtdetr_training", "--help"]

    completed = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--run-dir" in completed.stdout
