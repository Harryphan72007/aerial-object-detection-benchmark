from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import scripts.verify_model_environments as verifier
import src.workflows.isolated_environment as isolated
from src.utils.serialization import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def _git(*arguments: object, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *map(str, arguments)],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _origin(tmp_path: Path) -> tuple[dict[str, str], Path]:
    origin = tmp_path / "origin"
    origin.mkdir()
    _git("init", cwd=origin)
    _git("config", "user.email", "tests@example.invalid", cwd=origin)
    _git("config", "user.name", "Provisioning Tests", cwd=origin)
    (origin / "tracked.txt").write_text("pinned\n", encoding="utf-8")
    _git("add", "tracked.txt", cwd=origin)
    _git("commit", "-m", "fixture", cwd=origin)
    revision = _git("rev-parse", "HEAD", cwd=origin)
    return (
        {
            "name": "FixtureFramework",
            "url": str(origin),
            "revision": revision,
        },
        origin,
    )


def _checkout(tmp_path: Path, source: dict[str, str]) -> Path:
    return isolated._framework_checkout_path(tmp_path / "frameworks", source)


def test_openmmlab_inputs_include_exact_psutil_and_probe_imports_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requirements = (ROOT / "requirements-openmmlab-py310-cu118.txt").read_text(
        encoding="utf-8"
    )
    assert "psutil==7.0.0" in requirements.splitlines()
    imported: list[str] = []

    class Operations:
        nms = object()

    monkeypatch.setattr(verifier, "_require_exact", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        verifier,
        "_require_import",
        lambda module, _stage: imported.append(module)
        or (Operations() if module == "mmcv.ops" else object()),
    )
    monkeypatch.setattr(verifier, "_git_revision", lambda *_args, **_kwargs: "revision")
    spec = isolated.resolved_runtime_spec(ROOT, "faster_rcnn_swin_t")
    verifier._openmmlab_probe(
        spec, Namespace(mmdet_root=str(tmp_path / "mmdetection")), {}
    )
    assert "psutil" in imported


def test_requirements_change_alters_requirements_and_runtime_hash(tmp_path: Path) -> None:
    spec = isolated.resolved_runtime_spec(ROOT, "faster_rcnn_swin_t")
    contract_files = (
        "pyproject.toml",
        "requirements-dataset-colab.txt",
        str(spec["requirements"]),
        "scripts/verify_model_environments.py",
        "src/notebook_utils.py",
        "src/subprocess_utils.py",
        "src/workflows/environment.py",
        "src/workflows/isolated_environment.py",
    )
    repository = tmp_path / "repository"
    for relative in contract_files:
        source = ROOT / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    first_requirements = isolated._requirements_hash(repository, spec)
    first_runtime = isolated._runtime_hash(repository, spec)
    requirement = repository / str(spec["requirements"])
    requirement.write_text(
        requirement.read_text(encoding="utf-8") + "\n# hash-change\n",
        encoding="utf-8",
    )
    assert isolated._requirements_hash(repository, spec) != first_requirements
    assert isolated._runtime_hash(repository, spec) != first_runtime


def test_framework_roots_are_platform_local_and_revision_keyed(tmp_path: Path) -> None:
    assert isolated._runtime_framework_root(tmp_path, platform="colab") == Path(
        "/content/visdrone_frameworks"
    )
    assert isolated._runtime_framework_root(tmp_path, platform="kaggle") == Path(
        "/kaggle/working/visdrone_frameworks"
    )
    assert isolated._runtime_framework_root(tmp_path, platform="local") == (
        tmp_path.resolve() / "visdrone_frameworks"
    )
    spec = isolated.resolved_runtime_spec(ROOT, "faster_rcnn_vmamba_t")
    drive = tmp_path / "drive"
    framework_root = tmp_path / "runtime" / "visdrone_frameworks"
    paths = isolated._family_paths(drive, spec, framework_root=framework_root)
    assert drive.resolve() not in paths["mmdet_root"].resolve().parents
    assert drive.resolve() not in paths["vmamba_root"].resolve().parents
    assert str(isolated._source(spec, "MMDetection")["revision"]) in paths[
        "mmdet_root"
    ].name
    assert str(isolated._source(spec, "VMamba")["revision"]) in paths[
        "vmamba_root"
    ].name


def test_framework_lock_waits_and_never_removes_active_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / ".locks" / "fixture.lock"
    entered = threading.Event()
    release = threading.Event()

    def owner() -> None:
        with isolated._framework_provisioning_lock(
            lock_path, framework="Fixture", revision="abc", timeout=2
        ):
            entered.set()
            assert release.wait(5)

    thread = threading.Thread(target=owner)
    thread.start()
    assert entered.wait(2)
    with pytest.raises(TimeoutError, match="waiting for framework lock"):
        with isolated._framework_provisioning_lock(
            lock_path, framework="Fixture", revision="abc", timeout=0.1
        ):
            pytest.fail("an active framework lock was acquired twice")
    assert lock_path.exists()
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_stale_framework_lock_metadata_is_recovered(tmp_path: Path) -> None:
    lock_path = tmp_path / ".locks" / "fixture.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(b"0")
    metadata = lock_path.with_suffix(".lock.json")
    write_json(
        metadata,
        {
            "framework": "Fixture",
            "revision": "abc",
            "pid": 2_147_483_647,
            "status": "active",
        },
    )
    with isolated._framework_provisioning_lock(
        lock_path, framework="Fixture", revision="abc", timeout=1
    ) as acquired:
        assert acquired["recovered_stale_lock"] is True
    assert read_json(metadata)["status"] == "released"


def test_git_index_lock_recovery_requires_guard_and_rejects_active_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, origin = _origin(tmp_path)
    del source
    index_lock = origin / ".git" / "index.lock"
    index_lock.write_text("stale", encoding="utf-8")
    with pytest.raises(RuntimeError, match="requires the framework provisioning lock"):
        isolated._remove_stale_git_locks(origin)
    monkeypatch.setattr(isolated, "_active_git_processes", lambda _path: [1234])
    with pytest.raises(RuntimeError, match="Refusing to remove active Git lock"):
        isolated._remove_stale_git_locks(origin, provisioning_lock_held=True)
    assert index_lock.exists()
    monkeypatch.setattr(isolated, "_active_git_processes", lambda _path: [])
    assert isolated._remove_stale_git_locks(
        origin, provisioning_lock_held=True
    ) is True
    assert not index_lock.exists()


def test_atomic_checkout_is_clean_pinned_and_reused(tmp_path: Path) -> None:
    source, _ = _origin(tmp_path)
    destination = _checkout(tmp_path, source)
    first = isolated._clone_pinned(
        source, destination, family="fixture", python=Path(sys.executable)
    )
    assert first == destination
    sentinel = read_json(destination / isolated.FRAMEWORK_COMPLETE_SENTINEL)
    assert sentinel["requested_revision"] == source["revision"]
    assert sentinel["resolved_commit"] == source["revision"]
    assert _git("rev-parse", "HEAD", cwd=destination) == source["revision"]
    assert _git("status", "--porcelain", "--untracked-files=all", cwd=destination) == ""
    second = isolated._clone_pinned(
        source, destination, family="fixture", python=Path(sys.executable)
    )
    assert second == first
    assert not list(destination.parent.glob(f"{destination.name}.building-*"))


def test_dirty_cached_checkout_is_quarantined_and_rebuilt_clean(tmp_path: Path) -> None:
    source, _ = _origin(tmp_path)
    destination = _checkout(tmp_path, source)
    isolated._clone_pinned(
        source, destination, family="fixture", python=Path(sys.executable)
    )
    tracked = destination / "tracked.txt"
    tracked.write_text("staged\n", encoding="utf-8")
    _git("add", "tracked.txt", cwd=destination)
    tracked.write_text("working-tree\n", encoding="utf-8")
    (destination / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    isolated._clone_pinned(
        source, destination, family="fixture", python=Path(sys.executable)
    )
    assert tracked.read_text(encoding="utf-8") == "pinned\n"
    assert not (destination / "untracked.txt").exists()
    assert _git("status", "--porcelain", "--untracked-files=all", cwd=destination) == ""
    assert any((tmp_path / "frameworks" / ".quarantine").iterdir())


def test_failed_atomic_build_never_creates_final_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, _ = _origin(tmp_path)
    destination = _checkout(tmp_path, source)
    original_run = isolated._run

    def fail_checkout(command: list[str], **kwargs: object) -> None:
        if kwargs.get("stage") == "FixtureFramework_checkout":
            raise RuntimeError("injected checkout failure")
        original_run(command, **kwargs)

    monkeypatch.setattr(isolated, "_run", fail_checkout)
    with pytest.raises(RuntimeError, match="injected checkout failure"):
        isolated._clone_pinned(
            source, destination, family="fixture", python=Path(sys.executable)
        )
    assert not destination.exists()
    assert not list(destination.parent.glob(f"{destination.name}.building-*"))


def test_concurrent_provisioning_returns_one_uncorrupted_checkout(tmp_path: Path) -> None:
    source, _ = _origin(tmp_path)
    destination = _checkout(tmp_path, source)

    def provision() -> Path:
        return isolated._clone_pinned(
            source, destination, family="fixture", python=Path(sys.executable)
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: provision(), range(2)))
    assert results == [destination, destination]
    assert _git("rev-parse", "HEAD", cwd=destination) == source["revision"]
    assert _git("status", "--porcelain", "--untracked-files=all", cwd=destination) == ""
    assert (destination / isolated.FRAMEWORK_COMPLETE_SENTINEL).is_file()


def test_incomplete_final_checkout_is_never_reused(tmp_path: Path) -> None:
    source, _ = _origin(tmp_path)
    destination = _checkout(tmp_path, source)
    destination.mkdir(parents=True)
    (destination / "partial.txt").write_text("incomplete", encoding="utf-8")
    isolated._clone_pinned(
        source, destination, family="fixture", python=Path(sys.executable)
    )
    assert not (destination / "partial.txt").exists()
    assert (destination / isolated.FRAMEWORK_COMPLETE_SENTINEL).is_file()
    assert _git("status", "--porcelain", "--untracked-files=all", cwd=destination) == ""


def test_vmamba_extension_build_uses_disposable_environment_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = isolated.resolved_runtime_spec(ROOT, "faster_rcnn_vmamba_t")
    drive = tmp_path / "drive"
    framework_root = tmp_path / "frameworks"
    paths = isolated._family_paths(drive, spec, framework_root=framework_root)
    paths["vmamba_config"].parent.mkdir(parents=True, exist_ok=True)
    paths["vmamba_config"].write_text("# fixture\n", encoding="utf-8")
    extension = paths["vmamba_root"] / "kernels" / "selective_scan"
    extension.mkdir(parents=True)
    (extension / "setup.py").write_text("# fixture\n", encoding="utf-8")
    paths["pretrained"].parent.mkdir(parents=True, exist_ok=True)
    paths["pretrained"].write_bytes(b"weights")
    python = tmp_path / "environment" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("fixture", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(isolated, "_clone_pinned", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        isolated, "_run", lambda command, **_kwargs: commands.append(command)
    )
    isolated._prepare_family(
        ROOT,
        drive,
        python,
        spec,
        framework_root=framework_root,
    )
    assert len(commands) == 1
    rendered = [Path(value) for value in commands[0] if "selective_scan-" in value]
    assert len(rendered) == 1
    assert tmp_path / "environment" in rendered[0].parents
    assert paths["vmamba_root"] not in rendered[0].parents
    assert not rendered[0].exists()
