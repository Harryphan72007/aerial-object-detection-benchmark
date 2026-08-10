import json
import sys
from pathlib import Path

from src.runtime_manifest import write_runtime_environment_manifest
from src.subprocess_utils import MODEL_PYTHON_ENV, python_module_command
from src.workflows.isolated_environment import (
    _install_runtime,
    _runtime_hash,
    resolved_runtime_spec,
)

ROOT = Path(__file__).resolve().parents[1]


def test_family_runtime_contracts_are_exact_and_isolated() -> None:
    rtdetr = resolved_runtime_spec(ROOT, "rtdetrv2_l")
    resnet = resolved_runtime_spec(ROOT, "faster_rcnn_resnet50")
    vmamba = resolved_runtime_spec(ROOT, "faster_rcnn_vmamba_t")

    assert rtdetr["family"] == "rtdetr"
    assert rtdetr["python"] == "3.11.13"
    assert rtdetr["torch"] == "2.7.1+cu128"
    assert resnet["family"] == "openmmlab"
    assert resnet["python"] == "3.10.16"
    assert resnet["torch"] == "2.1.0+cu118"
    assert vmamba["family"] == "vmamba"
    assert vmamba["pretrained"]["train_from_scratch"] is False
    assert _runtime_hash(ROOT, rtdetr) != _runtime_hash(ROOT, resnet)
    assert _runtime_hash(ROOT, resnet) != _runtime_hash(ROOT, vmamba)


def test_runtime_install_targets_selected_environment_only(
    tmp_path: Path, monkeypatch
) -> None:
    spec = resolved_runtime_spec(ROOT, "rtdetrv2_l")
    selected_python = tmp_path / "rtdetr" / "bin" / "python"
    commands: list[list[str]] = []
    monkeypatch.setattr(
        "src.workflows.isolated_environment._run",
        lambda command, **_: commands.append(command),
    )

    _install_runtime(ROOT, selected_python.parent.parent, selected_python, spec)

    assert commands
    assert all(
        str(selected_python) in command
        for command in commands
        if "pip" in command
    )
    assert all(str(tmp_path / "other-family") not in command for command in commands)


def test_repository_modules_use_selected_runtime_python(
    tmp_path: Path, monkeypatch
) -> None:
    selected = tmp_path / "bin" / "python"
    selected.parent.mkdir(parents=True)
    selected.write_text("", encoding="utf-8")
    monkeypatch.setenv(MODEL_PYTHON_ENV, str(selected))
    assert python_module_command("scripts.evaluate", "--help") == [
        str(selected),
        "-m",
        "scripts.evaluate",
        "--help",
    ]


def test_runtime_manifest_records_sources_packages_and_environment(
    tmp_path: Path,
) -> None:
    manifest = write_runtime_environment_manifest(
        tmp_path, "rtdetrv2_l", ROOT
    )
    persisted = json.loads(
        (tmp_path / "runtime_environment.json").read_text(encoding="utf-8")
    )

    assert persisted["model_id"] == "rtdetrv2_l"
    assert persisted["python_executable"] == sys.executable
    assert persisted["source_commit"]
    assert persisted["runtime_contract"]["sources"]
    assert persisted["packages"]
    assert manifest["path"].endswith("runtime_environment.json")
