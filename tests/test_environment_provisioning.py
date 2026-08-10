from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest
import nbformat

import scripts.verify_model_environments as verifier
import src.workflows.isolated_environment as isolated
from scripts.validate_notebooks import validate_notebook
from src.subprocess_utils import CheckedSubprocessError, run_checked
from src.workflows.pretrained_checkpoints import CheckpointVerificationError
from src.utils.serialization import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def _runtime_location(
    tmp_path: Path, model_id: str
) -> tuple[dict[str, object], Path, Path]:
    spec = isolated.resolved_runtime_spec(ROOT, model_id)
    digest = isolated._runtime_hash(ROOT, spec)
    environment = tmp_path / "runtime" / f"{spec['family']}-{digest[:12]}"
    return spec, environment, isolated._python_path(environment)


def _fake_runtime_commands(
    monkeypatch: pytest.MonkeyPatch,
    installs: list[str] | None = None,
) -> None:
    monkeypatch.setattr(isolated, "_ensure_uv", lambda _version: None)

    def fake_run(command: list[str], **kwargs: object) -> None:
        if kwargs.get("stage") == "virtual_environment_creation":
            environment = Path(command[-1])
            python = isolated._python_path(environment)
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("fake python", encoding="utf-8")

    monkeypatch.setattr(isolated, "_run", fake_run)

    def fake_install(
        _repo: Path,
        _environment: Path,
        _python: Path,
        spec: dict[str, object],
    ) -> None:
        if installs is not None:
            installs.append(str(spec["family"]))

    monkeypatch.setattr(isolated, "_install_runtime", fake_install)
    monkeypatch.setattr(
        isolated,
        "_prepare_family",
        lambda _repo, _drive, _python, _spec, **_kwargs: {},
    )


def _successful_probe(
    _python: Path,
    _repo: Path,
    environment: Path,
    spec: dict[str, object],
    _paths: dict[str, Path],
    *,
    quick: bool,
) -> tuple[Path, dict[str, object]]:
    probe = environment / (
        "quick_environment_probe.json" if quick else "environment_probe.json"
    )
    payload = {
        "status": "PASS",
        "environment": spec["family"],
        "quick": quick,
    }
    write_json(probe, payload)
    return probe, payload


def test_checked_subprocess_failure_contains_complete_child_output(
    tmp_path: Path,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["fake-python", "-m", "fake.verifier"],
            1,
            stdout="package probe output\n",
            stderr="selective_scan_cuda_oflex: undefined symbol\n",
        )

    probe = tmp_path / "environment_probe.json"
    with pytest.raises(CheckedSubprocessError) as raised:
        run_checked(
            ["fake-python", "-m", "fake.verifier"],
            cwd=tmp_path,
            environment_name="vmamba",
            stage="vmamba_complete_probe",
            python_executable="fake-python",
            probe_path=probe,
            runner=fail,
        )
    message = str(raised.value)
    assert "fake.verifier" in message
    assert str(tmp_path.resolve()) in message
    assert "Return code: 1" in message
    assert "package probe output" in message
    assert "selective_scan_cuda_oflex: undefined symbol" in message
    assert "vmamba_complete_probe" in message
    assert str(probe) in message


def test_checked_subprocess_diagnostics_redact_command_credentials(
    tmp_path: Path,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [], 1, stdout="", stderr="access_token=child-secret failed"
        )

    with pytest.raises(CheckedSubprocessError) as raised:
        run_checked(
            [
                "git",
                "clone",
                "https://username:secret@example.invalid/repo.git",
                "--token",
                "top-secret",
            ],
            cwd=tmp_path,
            runner=fail,
        )
    message = str(raised.value)
    assert "username" not in message
    assert "top-secret" not in message
    assert "child-secret" not in message
    assert "<redacted>" in message


def test_probe_failure_writes_failed_state_and_preserves_drive_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runtime_commands(monkeypatch)
    drive = tmp_path / "drive"
    protected = {
        drive / "datasets" / "annotations.json": b"dataset",
        drive / "hpo" / "study.db": b"optuna",
        drive / "checkpoints" / "run" / "last.pth": b"checkpoint",
        drive / "checkpoints" / "completed" / "run_manifest.json": b'{"status":"completed"}',
        drive / "predictions" / "rows.json": b"predictions",
    }
    for path, content in protected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def fail_probe(*_args: object, **_kwargs: object) -> tuple[Path, dict[str, object]]:
        raise CheckedSubprocessError(
            ["fake-python", "-m", "scripts.verify_model_environments"],
            cwd=ROOT,
            returncode=1,
            stdout="versions printed",
            stderr="mmcv._ext undefined symbol",
            environment_name="openmmlab",
            stage="mmcv_compiled_operation",
            python_executable="fake-python",
        )

    monkeypatch.setattr(isolated, "_verify_runtime", fail_probe)
    with pytest.raises(isolated.EnvironmentProvisioningError) as raised:
        isolated.provision_isolated_environment(
            "faster_rcnn_resnet50",
            ROOT,
            drive,
            runtime_base=tmp_path / "runtime",
        )
    _, environment, _ = _runtime_location(tmp_path, "faster_rcnn_resnet50")
    marker = read_json(environment / "benchmark_runtime.json")
    assert marker["state"] == "FAILED"
    assert marker["failed_stage"] == "mmcv_compiled_operation"
    assert {
        "schema_version",
        "runtime_hash",
        "model_id",
        "family",
        "state",
        "started_at",
        "completed_at",
        "python_executable",
        "failed_stage",
        "failure_type",
        "failure_message",
        "probe_path",
    }.issubset(marker)
    assert "versions printed" in marker["failure_message"]
    assert "mmcv._ext undefined symbol" in str(raised.value)
    assert not (drive / "environment_manifests").exists()
    for path, content in protected.items():
        assert path.read_bytes() == content


def test_failed_environment_is_rebuilt_and_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installs: list[str] = []
    _fake_runtime_commands(monkeypatch, installs)
    calls = 0

    def probe(*args: object, **kwargs: object) -> tuple[Path, dict[str, object]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("fake child verifier exit 1")
        return _successful_probe(*args, **kwargs)

    monkeypatch.setattr(isolated, "_verify_runtime", probe)
    with pytest.raises(isolated.EnvironmentProvisioningError):
        isolated.provision_isolated_environment(
            "rtdetrv2_l", ROOT, tmp_path / "drive", runtime_base=tmp_path / "runtime"
        )
    result = isolated.provision_isolated_environment(
        "rtdetrv2_l", ROOT, tmp_path / "drive", runtime_base=tmp_path / "runtime"
    )
    assert installs == ["rtdetr", "rtdetr"]
    assert result["state"] == "READY"
    assert result["packages_changed"] is True


def test_interrupted_installing_environment_is_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installs: list[str] = []
    _fake_runtime_commands(monkeypatch, installs)
    monkeypatch.setattr(isolated, "_verify_runtime", _successful_probe)
    spec, environment, python = _runtime_location(tmp_path, "rtdetrv2_l")
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("partial", encoding="utf-8")
    stale = environment / "stale-install-file"
    stale.write_text("partial", encoding="utf-8")
    write_json(
        environment / "benchmark_runtime.json",
        {
            "state": "INSTALLING",
            "runtime_hash": isolated._runtime_hash(ROOT, spec),
            "model_id": "rtdetrv2_l",
        },
    )
    result = isolated.provision_isolated_environment(
        "rtdetrv2_l", ROOT, tmp_path / "drive", runtime_base=tmp_path / "runtime"
    )
    assert result["state"] == "READY"
    assert installs == ["rtdetr"]
    assert not stale.exists()


def test_ready_matching_environment_passes_quick_probe_and_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, environment, python = _runtime_location(tmp_path, "rtdetrv2_l")
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("ready", encoding="utf-8")
    write_json(
        environment / "benchmark_runtime.json",
        {
            "schema_version": 2,
            "state": "READY",
            "runtime_hash": isolated._runtime_hash(ROOT, spec),
            "model_id": "rtdetrv2_l",
            "family": "rtdetr",
            "started_at": "earlier",
            "python_executable": str(python),
        },
    )
    monkeypatch.setattr(isolated, "_verify_runtime", _successful_probe)
    monkeypatch.setattr(
        isolated,
        "_install_runtime",
        lambda *_args, **_kwargs: pytest.fail("READY runtime was reinstalled"),
    )
    result = isolated.provision_isolated_environment(
        "rtdetrv2_l", ROOT, tmp_path / "drive", runtime_base=tmp_path / "runtime"
    )
    assert result["state"] == "READY"
    assert result["packages_changed"] is False
    assert read_json(environment / "quick_environment_probe.json")["quick"] is True


def test_ready_runtime_with_failed_quick_probe_is_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installs: list[str] = []
    _fake_runtime_commands(monkeypatch, installs)
    spec, environment, python = _runtime_location(tmp_path, "rtdetrv2_l")
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("stale", encoding="utf-8")
    write_json(
        environment / "benchmark_runtime.json",
        {
            "state": "READY",
            "runtime_hash": isolated._runtime_hash(ROOT, spec),
            "model_id": "rtdetrv2_l",
            "started_at": "earlier",
        },
    )
    calls: list[bool] = []

    def probe(*args: object, **kwargs: object) -> tuple[Path, dict[str, object]]:
        calls.append(bool(kwargs["quick"]))
        if kwargs["quick"]:
            raise CheckedSubprocessError(
                [str(python), "-m", "scripts.verify_model_environments"],
                cwd=ROOT,
                returncode=1,
                stdout="",
                stderr="stale CUDA extension",
                environment_name="rtdetr",
                stage="quick_probe",
            )
        return _successful_probe(*args, **kwargs)

    monkeypatch.setattr(isolated, "_verify_runtime", probe)
    result = isolated.provision_isolated_environment(
        "rtdetrv2_l", ROOT, tmp_path / "drive", runtime_base=tmp_path / "runtime"
    )
    assert calls == [True, False]
    assert installs == ["rtdetr"]
    assert result["state"] == "READY"


def test_family_verifier_modes_and_vmamba_complete_gate(tmp_path: Path) -> None:
    cases = {
        "faster_rcnn_resnet50": "openmmlab",
        "faster_rcnn_swin_t": "openmmlab",
        "faster_rcnn_vmamba_t": "vmamba",
        "rtdetrv2_l": "rtdetr",
    }
    for model_id, family in cases.items():
        spec = isolated.resolved_runtime_spec(ROOT, model_id)
        paths = isolated._family_paths(tmp_path / "drive", spec)
        command = isolated._verification_command(
            Path("selected-python"),
            ROOT,
            spec,
            paths,
            tmp_path / f"{model_id}.json",
            quick=False,
        )
        assert command[command.index("--environment") + 1] == family
        assert command[command.index("--model-id") + 1] == model_id
        assert ("--construct-model" in command) is (family == "vmamba")


def test_vmamba_probe_runs_openmmlab_base_and_vmamba_specific_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        verifier,
        "_base_probe",
        lambda *_args, **_kwargs: ({"python": "3.10.16", "torch": "2.1.0+cu118"}, object()),
    )
    monkeypatch.setattr(
        verifier,
        "_openmmlab_probe",
        lambda *_args, **_kwargs: calls.append("openmmlab") or {},
    )
    monkeypatch.setattr(
        verifier,
        "_vmamba_probe",
        lambda *_args, **_kwargs: calls.append("vmamba") or {},
    )
    args = verifier.parse_args(
        [
            "--environment",
            "vmamba",
            "--model-id",
            "faster_rcnn_vmamba_t",
            "--repo-root",
            str(ROOT),
        ]
    )
    assert verifier.verify(args)["status"] == "PASS"
    assert calls == ["openmmlab", "vmamba"]


def test_missing_selective_scan_is_actionable_and_writes_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        verifier,
        "verify",
        lambda _args: (_ for _ in ()).throw(
            verifier.ProbeFailure(
                "selective_scan_cuda_oflex_import",
                "could not import selective_scan_cuda_oflex: undefined symbol",
            )
        ),
    )
    output = tmp_path / "probe.json"
    returncode = verifier.main(
        [
            "--environment",
            "vmamba",
            "--model-id",
            "faster_rcnn_vmamba_t",
            "--json-output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    assert returncode == 1
    assert "selective_scan_cuda_oflex_import" in captured.err
    assert read_json(output)["stage"] == "selective_scan_cuda_oflex_import"


def test_parent_runtime_uses_exact_child_probe_stage(tmp_path: Path) -> None:
    probe = tmp_path / "environment_probe.json"
    write_json(probe, {"status": "FAILED", "stage": "selective_scan_cuda_oflex_import"})
    error = CheckedSubprocessError(
        ["python", "-m", "scripts.verify_model_environments"],
        cwd=ROOT,
        returncode=1,
        stdout="",
        stderr="child failed",
        stage="vmamba_complete_probe",
    )
    assert isolated._failed_stage(error, probe) == "selective_scan_cuda_oflex_import"


def test_missing_vmamba_pretrained_fails_before_hpo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = isolated.resolved_runtime_spec(ROOT, "faster_rcnn_vmamba_t")
    drive = tmp_path / "drive"
    framework_root = tmp_path / "frameworks"
    paths = isolated._family_paths(drive, spec, framework_root=framework_root)
    paths["vmamba_config"].parent.mkdir(parents=True, exist_ok=True)
    paths["vmamba_config"].write_text("# config", encoding="utf-8")
    monkeypatch.setattr(isolated, "_clone_pinned", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("VISDRONE_ALLOW_CHECKPOINT_DOWNLOAD", "0")
    with pytest.raises(CheckpointVerificationError, match="is missing"):
        isolated._prepare_family(
            ROOT, drive, Path("python"), spec, framework_root=framework_root
        )


def test_mmcv_compiled_operation_failure_is_detected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = isolated.resolved_runtime_spec(ROOT, "faster_rcnn_resnet50")
    monkeypatch.setattr(verifier, "_require_exact", lambda *_args, **_kwargs: None)

    def fake_import(module: str, stage: str) -> object:
        if module == "mmcv.ops":
            raise verifier.ProbeFailure(stage, "mmcv._ext undefined symbol")
        return object()

    monkeypatch.setattr(verifier, "_require_import", fake_import)
    args = Namespace(mmdet_root=str(tmp_path / "mmdetection"))
    with pytest.raises(verifier.ProbeFailure, match="undefined symbol") as raised:
        verifier._openmmlab_probe(spec, args, {})
    assert raised.value.stage == "mmcv_compiled_operation"


def test_preview_notebooks_skip_provisioning_and_gate_before_workflow_run() -> None:
    contracts = {
        "10_hpo_resnet50.ipynb": "START_HPO",
        "11_hpo_swin_t.ipynb": "START_HPO",
        "12_hpo_vmamba_t.ipynb": "START_HPO",
        "13_hpo_rtdetrv2.ipynb": "START_HPO",
        "20_finetune_resnet50.ipynb": "START_FINETUNING",
        "21_finetune_swin_t.ipynb": "START_FINETUNING",
        "22_finetune_vmamba_t.ipynb": "START_FINETUNING",
        "23_finetune_rtdetrv2.ipynb": "START_FINETUNING",
    }
    for name, flag in contracts.items():
        notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][3]["source"])
        assert f"if {flag}" in source
        assert "SKIPPED_PREVIEW" in source
        assert source.index("ensure_model_environment") < source.index(".run(")
        assert source.index(f"if {flag}") < source.index(".run(")


def test_selecting_another_family_clears_stale_model_runtime_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vmamba_paths = {
        "mmdet_root": tmp_path / "mmdet",
        "vmamba_root": tmp_path / "vmamba",
        "pretrained": tmp_path / "vmamba.pth",
    }
    isolated._select_runtime(
        tmp_path / "vmamba-python",
        tmp_path / "vmamba-manifest.json",
        vmamba_paths,
    )
    isolated._select_runtime(
        tmp_path / "rtdetr-python",
        tmp_path / "rtdetr-manifest.json",
        {},
    )
    assert isolated.os.environ[isolated.MODEL_PYTHON_ENV].endswith("rtdetr-python")
    assert "VMAMBA_ROOT" not in isolated.os.environ
    assert "VMAMBA_T_PRETRAINED" not in isolated.os.environ
    assert "MMDET_ROOT" not in isolated.os.environ
    isolated._clear_runtime_selection()


def test_runtime_cleanup_cannot_escape_configured_local_base(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="outside configured base"):
        isolated._safe_remove_environment(outside, tmp_path / "runtime")
    assert outside.is_dir()


def test_notebook_validation_rejects_inline_setup_and_missing_shared_api(
    tmp_path: Path,
) -> None:
    path = tmp_path / "12_hpo_vmamba_t.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "from src.hpo.workflow import TwoStageRandomHPO\n%pip install mmcv"
            )
        ]
    )
    nbformat.write(notebook, path)
    errors = validate_notebook(path)
    assert any("inline environment setup" in error for error in errors)
    assert any("does not call ensure_model_environment" in error for error in errors)
