"""Transactional model environments and immutable local framework checkouts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

try:
    import psutil
except ImportError:  # The shared notebook requirements install it before provisioning.
    psutil = None  # type: ignore[assignment]

from src.notebook_environment import detect_notebook_platform
from src.subprocess_utils import (
    CheckedSubprocessError,
    build_model_subprocess_environment,
    run_checked,
)
from src.reproducibility import git_commit
from src.utils.serialization import read_json, read_yaml, write_json

MODEL_PYTHON_ENV = "VISDRONE_MODEL_PYTHON"
RUNTIME_MANIFEST_ENV = "VISDRONE_RUNTIME_MANIFEST"
RUNTIME_CONFIG = "configs/runtime_environments.yaml"
RUNTIME_SCHEMA_VERSION = 2
READY_STATE = "READY"
INCOMPLETE_STATES = {"CREATING", "INSTALLING", "VERIFYING"}
FRAMEWORK_PROVISIONING_VERSION = 3
FRAMEWORK_COMPLETE_SENTINEL = ".provisioning_complete.json"
DEFAULT_FRAMEWORK_LOCK_TIMEOUT_SECONDS = 600.0
FRAMEWORK_LOCK_TIMEOUT_ENV = "VISDRONE_FRAMEWORK_LOCK_TIMEOUT_SECONDS"


class EnvironmentProvisioningError(RuntimeError):
    """Environment setup failed without changing experiment artifacts."""


def runtime_family(model_id: str) -> str:
    if model_id == "rtdetrv2_l":
        return "rtdetr"
    if model_id == "faster_rcnn_vmamba_t":
        return "vmamba"
    return "openmmlab"


def _resolved_spec(repo_root: Path, model_id: str) -> dict[str, Any]:
    config = read_yaml(repo_root / RUNTIME_CONFIG)
    family = runtime_family(model_id)
    if family == "vmamba":
        spec = dict(config["openmmlab"])
        spec["sources"] = [
            *config["openmmlab"]["sources"],
            *config["vmamba"]["sources"],
        ]
        spec["pretrained"] = config["vmamba"]["pretrained"]
        model_config = read_yaml(repo_root / "configs/faster_rcnn_vmamba_t/model.yaml")
        spec["framework_config"] = model_config["framework_config"]
    else:
        spec = dict(config[family])
    spec.update(
        {
            "family": family,
            "model_id": model_id,
            "provisioner": config["provisioner"],
            "framework_provisioning_version": FRAMEWORK_PROVISIONING_VERSION,
            "schema_version": config["schema_version"],
        }
    )
    return spec


def resolved_runtime_spec(repo_root: str | Path, model_id: str) -> dict[str, Any]:
    return _resolved_spec(Path(repo_root).resolve(), model_id)


def _runtime_hash(repo_root: Path, spec: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    contract_files = [
        "pyproject.toml",
        "requirements-dataset-colab.txt",
        str(spec["requirements"]),
        "scripts/verify_model_environments.py",
        "src/notebook_utils.py",
        "src/subprocess_utils.py",
        "src/workflows/environment.py",
        "src/workflows/isolated_environment.py",
    ]
    if spec["family"] == "vmamba":
        contract_files.extend(
            [
                "configs/faster_rcnn_vmamba_t/model.yaml",
                "src/models/vmamba_frcnn/importer.py",
                "src/models/vmamba_frcnn/factory.py",
            ]
        )
    for relative in contract_files:
        path = repo_root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _requirements_hash(repo_root: Path, spec: Mapping[str, Any]) -> str:
    path = repo_root / str(spec["requirements"])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _python_path(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _environment_root_for_python(python: Path) -> Path:
    if python.parent.name.casefold() not in {"bin", "scripts"}:
        raise ValueError(f"Python executable is not inside a managed environment: {python}")
    return python.parent.parent


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment_name: str | None = None,
    stage: str | None = None,
    python_executable: Path | None = None,
    probe_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    log_path: Path | None = None,
) -> None:
    run_checked(
        command,
        cwd=cwd,
        env=env,
        environment_name=environment_name,
        stage=stage,
        python_executable=python_executable,
        probe_path=probe_path,
        log_path=log_path,
    )


def _ensure_uv(version: str) -> None:
    try:
        installed = importlib.metadata.version("uv")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed == version:
        return
    _run(
        [sys.executable, "-m", "pip", "install", f"uv=={version}"],
        environment_name="provisioner",
        stage="uv_installation",
        python_executable=Path(sys.executable),
    )


def _uv(*arguments: object) -> list[str]:
    return [sys.executable, "-m", "uv", *(str(value) for value in arguments)]


def _install_runtime(
    repo_root: Path,
    environment: Path,
    python: Path,
    spec: dict[str, Any],
) -> None:
    del environment
    common = {
        "environment_name": str(spec["family"]),
        "python_executable": python,
    }
    _run(
        _uv(
            "pip",
            "install",
            "--python",
            python,
            f"torch=={spec['torch']}",
            f"torchvision=={spec['torchvision']}",
            "--index-url",
            spec["torch_index"],
        ),
        stage="torch_installation",
        **common,
    )
    if spec["family"] in {"openmmlab", "vmamba"}:
        _run(
            _uv(
                "pip",
                "install",
                "--python",
                python,
                f"mmcv=={spec['mmcv']['version']}",
                "--find-links",
                spec["mmcv"]["wheel_index"],
                "--only-binary",
                "mmcv",
            ),
            stage="mmcv_wheel_installation",
            **common,
        )
    _run(
        _uv(
            "pip",
            "install",
            "--python",
            python,
            "-r",
            repo_root / str(spec["requirements"]),
        ),
        stage="requirements_installation",
        **common,
    )
    _run(
        _uv(
            "pip",
            "install",
            "--python",
            python,
            "--editable",
            repo_root,
            "--no-deps",
        ),
        stage="repository_installation",
        **common,
    )


def _source(spec: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for source in spec.get("sources", []):
        if str(source.get("name")) == name:
            return source
    raise KeyError(f"runtime source {name!r} is not configured")


def _framework_directory_name(source: Mapping[str, Any]) -> str:
    name = str(source["name"])
    if name == "MMDetection":
        return "mmdetection"
    if name == "VMamba":
        return "VMamba"
    rendered = "".join(
        character
        for character in name
        if character.isalnum() or character in "-_"
    )
    if not rendered:
        raise ValueError(f"invalid framework name: {name!r}")
    return rendered


def _source_url_identity(source: Mapping[str, Any]) -> str:
    normalized = str(source["url"]).strip().rstrip("/").casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _framework_checkout_path(
    framework_root: Path, source: Mapping[str, Any]
) -> Path:
    revision = str(source["revision"])
    identity = _source_url_identity(source)
    return (
        framework_root
        / _framework_directory_name(source)
        / f"{revision}-{identity}"
    )


def _runtime_framework_root(
    runtime_base: str | Path,
    *,
    platform: str | None = None,
) -> Path:
    selected = platform or detect_notebook_platform()
    if selected == "colab":
        return Path("/content/visdrone_frameworks")
    if selected == "kaggle":
        return Path("/kaggle/working/visdrone_frameworks")
    return Path(runtime_base).expanduser().resolve() / "visdrone_frameworks"


def _framework_lock_timeout() -> float:
    raw = os.environ.get(FRAMEWORK_LOCK_TIMEOUT_ENV)
    if raw is None:
        return DEFAULT_FRAMEWORK_LOCK_TIMEOUT_SECONDS
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{FRAMEWORK_LOCK_TIMEOUT_ENV} must be positive")
    return value


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if psutil is not None:
        return bool(psutil.pid_exists(pid))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _InterProcessFileLock:
    """Small OS-backed advisory lock whose kernel state is crash-released."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any | None = None

    def acquire(self, timeout: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._handle = handle
                return
            except OSError as error:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise TimeoutError(f"timed out acquiring {self.path}") from error
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle = None
            handle.close()


@contextmanager
def _framework_provisioning_lock(
    lock_path: Path,
    *,
    framework: str,
    revision: str,
    timeout: float | None = None,
) -> Iterator[dict[str, Any]]:
    """Serialize one framework revision without deleting another process's lock."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = lock_path.with_suffix(lock_path.suffix + ".json")
    previous: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            previous = read_json(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError):
            previous = {}
    lock = _InterProcessFileLock(lock_path)
    wait_seconds = _framework_lock_timeout() if timeout is None else timeout
    try:
        lock.acquire(timeout=wait_seconds)
    except TimeoutError as error:
        owner = previous.get("pid", "unknown")
        raise TimeoutError(
            f"Timed out after {wait_seconds:g}s waiting for framework lock "
            f"{lock_path}; recorded owner PID={owner}."
        ) from error
    recovered_stale = bool(
        previous.get("status") == "active"
        and isinstance(previous.get("pid"), int)
        and not _pid_is_alive(int(previous["pid"]))
    )
    acquired = {
        "schema_version": 1,
        "framework": framework,
        "revision": revision,
        "pid": os.getpid(),
        "status": "active",
        "acquired_at": _now(),
        "recovered_stale_lock": recovered_stale,
    }
    write_json(metadata_path, acquired)
    if recovered_stale:
        print(
            f"Framework lock: recovered stale metadata for {framework} {revision}; "
            f"previous PID={previous.get('pid')}"
        )
    try:
        yield acquired
    finally:
        released = {**acquired, "status": "released", "released_at": _now()}
        write_json(metadata_path, released)
        lock.release()


def _active_git_processes(repo_path: Path) -> list[int] | None:
    if psutil is None:
        return None
    repo = repo_path.resolve()
    repo_text = os.path.normcase(str(repo))
    active: list[int] = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
        try:
            info = process.info
            name = str(info.get("name") or "").casefold()
            if not name.startswith("git"):
                continue
            command = os.path.normcase(" ".join(info.get("cmdline") or []))
            cwd_value = info.get("cwd")
            cwd = Path(cwd_value).resolve() if cwd_value else None
            if repo_text in command or cwd == repo or (cwd is not None and repo in cwd.parents):
                active.append(int(info["pid"]))
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError, ValueError):
            continue
    return sorted(set(active))


def _remove_stale_git_locks(
    repo_path: Path,
    *,
    provisioning_lock_held: bool = False,
) -> bool:
    """Remove only this repository's inactive Git index lock under our lock."""
    if not provisioning_lock_held:
        raise RuntimeError("Git lock recovery requires the framework provisioning lock")
    index_lock = repo_path / ".git" / "index.lock"
    if not index_lock.exists():
        return False
    active = _active_git_processes(repo_path)
    if active is None:
        raise RuntimeError(
            f"Refusing to remove Git lock {index_lock}: psutil is unavailable, so "
            "active Git processes cannot be checked safely"
        )
    if active:
        raise RuntimeError(
            f"Refusing to remove active Git lock {index_lock}; Git PID(s): {active}"
        )
    index_lock.unlink()
    print(f"Removed stale Git lock: {index_lock} (no active Git process uses this checkout)")
    return True


def _git_text(
    repo_path: Path,
    arguments: list[str],
    *,
    family: str,
    stage: str,
    python: Path,
) -> str:
    completed = run_checked(
        ["git", "-C", str(repo_path), *arguments],
        environment_name=family,
        stage=stage,
        python_executable=python,
    )
    return (completed.stdout or "").strip()


def _required_framework_paths(source: Mapping[str, Any]) -> tuple[str, ...]:
    name = str(source["name"])
    if name == "MMDetection":
        return ("mmdet",)
    if name == "VMamba":
        return ("detection", "kernels/selective_scan")
    return ()


def _validate_framework_checkout(
    checkout: Path,
    source: Mapping[str, Any],
    *,
    family: str,
    python: Path,
) -> tuple[bool, str, dict[str, Any]]:
    sentinel = checkout / FRAMEWORK_COMPLETE_SENTINEL
    if not checkout.is_dir():
        return False, "checkout directory is missing", {}
    if not (checkout / ".git").is_dir():
        return False, ".git directory is missing", {}
    if not sentinel.is_file():
        return False, "completion sentinel is missing", {}
    if (checkout / ".git" / "index.lock").exists():
        return False, "Git index lock exists", {}
    try:
        record = read_json(sentinel)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return False, f"completion sentinel is unreadable: {error}", {}
    expected = {
        "framework": str(source["name"]),
        "repository_url": str(source["url"]),
        "requested_revision": str(source["revision"]),
        "provisioning_version": FRAMEWORK_PROVISIONING_VERSION,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            return False, f"sentinel {key} mismatch", record
    for relative in _required_framework_paths(source):
        if not (checkout / relative).exists():
            return False, f"required framework path is missing: {relative}", record
    try:
        head = _git_text(
            checkout,
            ["rev-parse", "HEAD"],
            family=family,
            stage=f"{source['name']}_revision_validation",
            python=python,
        )
        resolved = _git_text(
            checkout,
            ["rev-parse", f"{source['revision']}^{{commit}}"],
            family=family,
            stage=f"{source['name']}_requested_revision_validation",
            python=python,
        )
        status = _git_text(
            checkout,
            ["status", "--porcelain", "--untracked-files=all"],
            family=family,
            stage=f"{source['name']}_cleanliness_validation",
            python=python,
        )
        submodules = (
            _git_text(
                checkout,
                ["submodule", "status", "--recursive"],
                family=family,
                stage=f"{source['name']}_submodule_validation",
                python=python,
            )
            if (checkout / ".gitmodules").is_file()
            else ""
        )
    except Exception as error:
        return False, f"Git validation failed: {error}", record
    if head != resolved or head != str(record.get("resolved_commit")):
        return False, f"HEAD mismatch: head={head}, resolved={resolved}", record
    if status:
        return False, f"working tree is dirty: {status}", record
    invalid_submodules = [
        line for line in submodules.splitlines() if line.startswith(("-", "+", "U"))
    ]
    if invalid_submodules:
        return False, f"submodule state is not pinned: {invalid_submodules}", record
    return True, "valid", {**record, "git_status": status, "resolved_commit": head}


def _safe_remove_framework_path(target: Path, framework_root: Path) -> None:
    resolved = target.resolve()
    boundary = framework_root.resolve()
    if resolved == boundary or not resolved.is_relative_to(boundary):
        raise ValueError(f"Refusing to remove framework path outside local cache: {resolved}")
    if resolved.exists():
        def make_writable_and_retry(function: Any, path: str, _error: Any) -> None:
            os.chmod(path, stat.S_IWRITE)
            function(path)

        shutil.rmtree(resolved, onerror=make_writable_and_retry)


def _quarantine_framework_checkout(
    checkout: Path, framework_root: Path, reason: str
) -> Path:
    quarantine = framework_root / ".quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = quarantine / f"{checkout.parent.name}-{checkout.name}-{stamp}-{os.getpid()}"
    os.replace(checkout, destination)
    print(f"Framework checkout quarantined: {checkout} -> {destination}; reason: {reason}")
    return destination


def _clone_pinned(
    source: Mapping[str, Any],
    destination: Path,
    *,
    family: str,
    python: Path,
) -> Path:
    """Create or reuse one locked, atomic, clean, revision-keyed checkout."""
    url = str(source["url"])
    revision = str(source["revision"])
    framework_root = destination.parents[1]
    lock_name = (
        f"{_framework_directory_name(source)}-{revision}-{_source_url_identity(source)}.lock"
    )
    lock_path = framework_root / ".locks" / lock_name
    with _framework_provisioning_lock(
        lock_path,
        framework=str(source["name"]),
        revision=revision,
    ):
        if destination.exists():
            _remove_stale_git_locks(destination, provisioning_lock_held=True)
            valid, reason, record = _validate_framework_checkout(
                destination, source, family=family, python=python
            )
            if valid:
                print(f"Framework checkout: reused {destination}")
                print(f"Resolved framework revision: {record['resolved_commit']}")
                print("Git worktree status: clean")
                return destination
            _quarantine_framework_checkout(destination, framework_root, reason)

        destination.parent.mkdir(parents=True, exist_ok=True)
        for incomplete in destination.parent.glob(f"{destination.name}.building-*"):
            _safe_remove_framework_path(incomplete, framework_root)
        building = destination.parent / (
            f"{destination.name}.building-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        try:
            _run(
                ["git", "clone", "--no-checkout", url, str(building)],
                environment_name=family,
                stage=f"{source['name']}_clone",
                python_executable=python,
            )
            _remove_stale_git_locks(building, provisioning_lock_held=True)
            commands: list[tuple[list[str], str]] = [
                (["fetch", "--all", "--tags", "--prune"], "fetch"),
                (["checkout", "--detach", revision], "checkout"),
                (["reset", "--hard", revision], "reset"),
                (["clean", "-ffdx"], "clean"),
            ]
            if (building / ".gitmodules").is_file():
                commands.extend(
                    [
                        (["submodule", "sync", "--recursive"], "submodule_sync"),
                        (
                            [
                                "submodule",
                                "update",
                                "--init",
                                "--recursive",
                                "--force",
                            ],
                            "submodule_update",
                        ),
                        (
                            ["submodule", "foreach", "--recursive", "git reset --hard"],
                            "submodule_reset",
                        ),
                        (
                            ["submodule", "foreach", "--recursive", "git clean -ffdx"],
                            "submodule_clean",
                        ),
                    ]
                )
            commands.extend(
                [
                    (["reset", "--hard", revision], "final_reset"),
                    (["clean", "-ffdx"], "final_clean"),
                ]
            )
            for arguments, suffix in commands:
                _remove_stale_git_locks(building, provisioning_lock_held=True)
                _run(
                    ["git", "-C", str(building), *arguments],
                    environment_name=family,
                    stage=f"{source['name']}_{suffix}",
                    python_executable=python,
                )
            resolved = _git_text(
                building,
                ["rev-parse", f"{revision}^{{commit}}"],
                family=family,
                stage=f"{source['name']}_resolve_revision",
                python=python,
            )
            head = _git_text(
                building,
                ["rev-parse", "HEAD"],
                family=family,
                stage=f"{source['name']}_verify_head",
                python=python,
            )
            status = _git_text(
                building,
                ["status", "--porcelain", "--untracked-files=all"],
                family=family,
                stage=f"{source['name']}_verify_clean",
                python=python,
            )
            if head != resolved:
                raise RuntimeError(
                    f"Framework revision mismatch for {source['name']}: "
                    f"HEAD={head}, resolved requested revision={resolved}"
                )
            if status:
                raise RuntimeError(
                    f"Framework checkout remains dirty after cleanup: {building}\n{status}"
                )
            for relative in _required_framework_paths(source):
                if not (building / relative).exists():
                    raise FileNotFoundError(
                        f"Required framework path is missing after clone: {building / relative}"
                    )
            exclude = building / ".git" / "info" / "exclude"
            exclude.parent.mkdir(parents=True, exist_ok=True)
            existing_excludes = (
                exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
            )
            if FRAMEWORK_COMPLETE_SENTINEL not in existing_excludes.splitlines():
                with exclude.open("a", encoding="utf-8") as handle:
                    if existing_excludes and not existing_excludes.endswith("\n"):
                        handle.write("\n")
                    handle.write(FRAMEWORK_COMPLETE_SENTINEL + "\n")
            write_json(
                building / FRAMEWORK_COMPLETE_SENTINEL,
                {
                    "schema_version": 1,
                    "framework": str(source["name"]),
                    "repository_url": url,
                    "repository_url_identity": _source_url_identity(source),
                    "requested_revision": revision,
                    "resolved_commit": resolved,
                    "provisioning_version": FRAMEWORK_PROVISIONING_VERSION,
                    "provisioned_at": _now(),
                    "git_status": status,
                    "required_paths": list(_required_framework_paths(source)),
                },
            )
            os.replace(building, destination)
            valid, reason, record = _validate_framework_checkout(
                destination, source, family=family, python=python
            )
            if not valid:
                _quarantine_framework_checkout(destination, framework_root, reason)
                raise RuntimeError(
                    f"Atomic framework checkout validation failed for {destination}: {reason}"
                )
            print(f"Framework checkout: rebuilt {destination}")
            print(f"Resolved framework revision: {record['resolved_commit']}")
            print("Git worktree status: clean")
            return destination
        except Exception:
            if building.exists():
                _safe_remove_framework_path(building, framework_root)
            raise


def _family_paths(
    drive: Path,
    spec: Mapping[str, Any],
    *,
    framework_root: Path | None = None,
    runtime_base: Path | None = None,
) -> dict[str, Path]:
    if framework_root is None:
        configured_base = runtime_base or Path(
            os.environ.get(
                "VISDRONE_MODEL_ENV_ROOT",
                str(Path(tempfile.gettempdir()).resolve() / "visdrone_model_envs"),
            )
        )
        framework_root = _runtime_framework_root(configured_base)
    values: dict[str, Path] = {}
    if spec["family"] in {"openmmlab", "vmamba"}:
        values["mmdet_root"] = _framework_checkout_path(
            framework_root, _source(spec, "MMDetection")
        )
    if spec["family"] == "vmamba":
        vmamba_root = _framework_checkout_path(
            framework_root, _source(spec, "VMamba")
        )
        values.update(
            {
                "vmamba_root": vmamba_root,
                "vmamba_config": vmamba_root / str(spec["framework_config"]),
                "pretrained": drive / "pretrained" / str(spec["pretrained"]["filename"]),
            }
        )
    return values


def _preflight_vmamba_toolchain(python: Path) -> dict[str, str]:
    """Reject host toolchains that cannot compile the pinned Torch CUDA extension."""

    torch_probe = subprocess.run(
        [str(python), "-c", "import torch; print(torch.version.cuda or '')"],
        check=False,
        capture_output=True,
        text=True,
    )
    if torch_probe.returncode:
        raise EnvironmentProvisioningError(
            "VMamba CUDA preflight could not import Torch in the isolated runtime: "
            + (torch_probe.stderr.strip() or "unknown error")
        )
    torch_cuda = torch_probe.stdout.strip()
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = (
        Path(cuda_home) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
        if cuda_home
        else Path(shutil.which("nvcc") or "")
    )
    if not str(nvcc) or not nvcc.is_file():
        raise EnvironmentProvisioningError(
            "VMamba requires a host CUDA compiler; CUDA_HOME/nvcc was not found"
        )
    nvcc_probe = subprocess.run(
        [str(nvcc), "--version"], check=False, capture_output=True, text=True
    )
    nvcc_output = f"{nvcc_probe.stdout}\n{nvcc_probe.stderr}"
    match = re.search(r"release\s+(\d+)\.(\d+)", nvcc_output)
    if nvcc_probe.returncode or match is None:
        raise EnvironmentProvisioningError(
            "VMamba could not determine the host CUDA version from nvcc --version"
        )
    host_cuda = f"{match.group(1)}.{match.group(2)}"
    if torch_cuda and host_cuda.split(".", 1)[0] != torch_cuda.split(".", 1)[0]:
        raise EnvironmentProvisioningError(
            "VMamba CUDA major-version mismatch: "
            f"Torch was built for CUDA {torch_cuda}, but nvcc is CUDA {host_cuda}. "
            "Install a matching toolkit or use a verified prebuilt extension."
        )
    compiler = (shutil.which("cl") or "") if os.name == "nt" else (shutil.which("g++") or "")
    compiler_version = "unknown"
    if os.name == "nt":
        if not compiler:
            raise EnvironmentProvisioningError(
                "VMamba requires the MSVC cl compiler in PATH"
            )
        compiler_probe = subprocess.run(
            [compiler], check=False, capture_output=True, text=True
        )
        compiler_output = f"{compiler_probe.stdout}\n{compiler_probe.stderr}"
        compiler_match = re.search(r"Version\s+([0-9.]+)", compiler_output)
        if compiler_match is None:
            raise EnvironmentProvisioningError(
                "VMamba could not determine the MSVC compiler version"
            )
        compiler_version = compiler_match.group(1)
    else:
        if not compiler:
            raise EnvironmentProvisioningError(
                "VMamba requires g++ to compile selective_scan_cuda_oflex"
            )
        compiler_probe = subprocess.run(
            [compiler, "-dumpfullversion", "-dumpversion"],
            check=False,
            capture_output=True,
            text=True,
        )
        compiler_version = compiler_probe.stdout.strip()
        if compiler_probe.returncode or not compiler_version:
            raise EnvironmentProvisioningError("VMamba could not determine the g++ version")
        if host_cuda.startswith("11.8") and int(compiler_version.split(".", 1)[0]) > 11:
            raise EnvironmentProvisioningError(
                "CUDA 11.8 requires a supported host compiler; use GCC/G++ 11 or older "
                f"instead of {compiler_version}"
            )
    return {
        "torch_cuda": torch_cuda,
        "nvcc": str(nvcc),
        "host_cuda": host_cuda,
        "compiler": str(compiler),
        "compiler_version": compiler_version,
    }


def _prepare_family(
    repo: Path,
    drive: Path,
    python: Path,
    spec: dict[str, Any],
    *,
    framework_root: Path | None = None,
    install_kernel: bool = True,
) -> dict[str, Path]:
    del repo
    family = str(spec["family"])
    paths = _family_paths(drive, spec, framework_root=framework_root)
    if family in {"openmmlab", "vmamba"}:
        legacy = drive / "frameworks" / "mmdetection"
        if legacy.exists():
            print(f"Legacy framework cache ignored (not modified): {legacy}")
        _clone_pinned(
            _source(spec, "MMDetection"),
            paths["mmdet_root"],
            family=family,
            python=python,
        )
    if family == "vmamba":
        legacy = drive / "frameworks" / "VMamba"
        if legacy.exists():
            print(f"Legacy framework cache ignored (not modified): {legacy}")
        _clone_pinned(
            _source(spec, "VMamba"),
            paths["vmamba_root"],
            family=family,
            python=python,
        )
        if not paths["vmamba_config"].is_file():
            raise FileNotFoundError(
                f"VMamba detector configuration is missing: {paths['vmamba_config']}"
            )
        pretrained = paths["pretrained"]
        if not pretrained.is_file() or pretrained.stat().st_size == 0:
            raise FileNotFoundError(
                "VMamba pretrained checkpoint validation failed before HPO: "
                f"expected a non-empty file at {pretrained}. Training from scratch "
                "is intentionally disabled."
            )
        if install_kernel:
            _preflight_vmamba_toolchain(python)
            extension_source = paths["vmamba_root"] / "kernels" / "selective_scan"
            build_source = (
                _environment_root_for_python(python)
                / ".framework-build"
                / f"selective_scan-{str(_source(spec, 'VMamba')['revision'])[:12]}"
            )
            if build_source.exists():
                shutil.rmtree(build_source)
            build_source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(extension_source, build_source)
            try:
                _run(
                    _uv(
                        "pip",
                        "install",
                        "--python",
                        python,
                        build_source,
                        "--no-build-isolation",
                    ),
                    environment_name=family,
                    stage="selective_scan_compilation",
                    python_executable=python,
                )
            finally:
                if build_source.exists():
                    shutil.rmtree(build_source)
    return paths


def _framework_checkout_records(
    paths: Mapping[str, Path], spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path_key, source_name in (
        ("mmdet_root", "MMDetection"),
        ("vmamba_root", "VMamba"),
    ):
        path = paths.get(path_key)
        if path is None:
            continue
        sentinel = path / FRAMEWORK_COMPLETE_SENTINEL
        record = read_json(sentinel)
        records.append(
            {
                **record,
                "path": str(path),
                "configured_source": dict(_source(spec, source_name)),
            }
        )
    return records


def _probe_environment(paths: Mapping[str, Path]) -> dict[str, str]:
    environment = build_model_subprocess_environment()
    if "mmdet_root" in paths:
        environment["MMDET_ROOT"] = str(paths["mmdet_root"])
    if "vmamba_root" in paths:
        environment["VMAMBA_ROOT"] = str(paths["vmamba_root"])
        environment["VMAMBA_T_PRETRAINED"] = str(paths["pretrained"])
    return environment


def _verification_command(
    python: Path,
    repo: Path,
    spec: Mapping[str, Any],
    paths: Mapping[str, Path],
    probe_path: Path,
    *,
    quick: bool,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "scripts.verify_model_environments",
        "--environment",
        str(spec["family"]),
        "--model-id",
        str(spec["model_id"]),
        "--repo-root",
        str(repo),
        "--require-gpu",
        "--json-output",
        str(probe_path),
    ]
    if quick:
        command.append("--quick")
    if "mmdet_root" in paths:
        command.extend(["--mmdet-root", str(paths["mmdet_root"])])
    if "vmamba_root" in paths:
        command.extend(
            [
                "--vmamba-root",
                str(paths["vmamba_root"]),
                "--vmamba-config",
                str(paths["vmamba_config"]),
                "--pretrained-checkpoint",
                str(paths["pretrained"]),
            ]
        )
        if not quick:
            command.append("--construct-model")
    return command


def _verify_runtime(
    python: Path,
    repo: Path,
    environment: Path,
    spec: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    quick: bool,
) -> tuple[Path, dict[str, Any]]:
    probe = environment / ("quick_environment_probe.json" if quick else "environment_probe.json")
    stage = "quick_probe" if quick else f"{spec['family']}_complete_probe"
    _run(
        _verification_command(python, repo, spec, paths, probe, quick=quick),
        cwd=repo,
        env=_probe_environment(paths),
        environment_name=str(spec["family"]),
        stage=stage,
        python_executable=python,
        probe_path=probe,
        log_path=environment / "logs" / f"{stage}.log",
    )
    payload = read_json(probe)
    if payload.get("status") != "PASS":
        raise RuntimeError(f"{stage} returned a non-PASS probe: {probe}")
    return probe, payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_state(
    marker: Path,
    *,
    runtime_hash: str,
    model_id: str,
    family: str,
    state: str,
    python: Path,
    started_at: str,
    probe_path: Path,
    completed_at: str | None = None,
    failed_stage: str | None = None,
    failure_type: str | None = None,
    failure_message: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "runtime_hash": runtime_hash,
        "model_id": model_id,
        "family": family,
        "state": state,
        "status": state,
        "started_at": started_at,
        "completed_at": completed_at,
        "python_executable": str(python),
        "failed_stage": failed_stage,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "probe_path": str(probe_path),
    }
    if extra:
        record.update(extra)
    write_json(marker, record)
    return record


def _safe_remove_environment(environment: Path, runtime_base: Path) -> None:
    resolved = environment.resolve()
    boundary = runtime_base.resolve()
    if resolved.parent != boundary:
        raise ValueError(
            f"Refusing to remove runtime outside configured base: {resolved}"
        )
    if resolved.exists():
        shutil.rmtree(resolved)


def _preserve_failed_runtime_diagnostics(
    environment: Path,
    drive_root: Path,
    model_id: str,
    runtime_hash: str,
) -> Path | None:
    if not environment.is_dir():
        return None
    candidates = [
        environment / "benchmark_runtime.json",
        environment / "environment_probe.json",
        environment / "quick_environment_probe.json",
        environment / "logs",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = (
        drive_root
        / "environment_failures"
        / model_id
        / runtime_hash
        / f"{stamp}-{os.getpid()}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    for source in existing:
        target = destination / source.name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    print(f"Preserved failed environment diagnostics: {destination}")
    return destination


def _clear_runtime_selection() -> None:
    for variable in (
        MODEL_PYTHON_ENV,
        RUNTIME_MANIFEST_ENV,
        "MMDET_ROOT",
        "VMAMBA_ROOT",
        "VMAMBA_T_PRETRAINED",
    ):
        os.environ.pop(variable, None)


def _select_runtime(
    python: Path,
    manifest_path: Path,
    paths: Mapping[str, Path],
) -> None:
    _clear_runtime_selection()
    os.environ[MODEL_PYTHON_ENV] = str(python)
    os.environ[RUNTIME_MANIFEST_ENV] = str(manifest_path)
    if "mmdet_root" in paths:
        os.environ["MMDET_ROOT"] = str(paths["mmdet_root"])
    if "vmamba_root" in paths:
        os.environ["VMAMBA_ROOT"] = str(paths["vmamba_root"])
        os.environ["VMAMBA_T_PRETRAINED"] = str(paths["pretrained"])


def _failed_stage(error: Exception, probe_path: Path | None = None) -> str:
    if probe_path is not None and probe_path.is_file():
        try:
            probe = read_json(probe_path)
        except (OSError, ValueError, json.JSONDecodeError):
            probe = {}
        if probe.get("stage"):
            return str(probe["stage"])
    if isinstance(error, CheckedSubprocessError) and error.stage:
        return error.stage
    if isinstance(error, FileNotFoundError):
        message = str(error).lower()
        if "pretrained" in message:
            return "vmamba_pretrained_checkpoint"
        if "configuration" in message:
            return "vmamba_detector_configuration"
    return "environment_provisioning"


def provision_isolated_environment(
    model_id: str,
    repo_root: str | Path,
    drive_root: str | Path,
    *,
    runtime_base: str | Path = "/content/visdrone_model_envs",
) -> dict[str, Any]:
    """Provision, fully verify, and select a family runtime transactionally."""
    repo = Path(repo_root).resolve()
    drive = Path(drive_root).expanduser().resolve()
    base = Path(runtime_base).expanduser().resolve()
    spec = _resolved_spec(repo, model_id)
    family = str(spec["family"])
    runtime_hash = _runtime_hash(repo, spec)
    requirements_hash = _requirements_hash(repo, spec)
    environment = base / f"{family}-{runtime_hash[:12]}"
    framework_root = _runtime_framework_root(base)
    python = _python_path(environment)
    marker = environment / "benchmark_runtime.json"
    probe_path = environment / "environment_probe.json"
    manifest_path = drive / "environment_manifests" / model_id / f"{runtime_hash}.json"
    print(f"Requirements hash: {requirements_hash}")
    print(f"Runtime hash: {runtime_hash}")
    print(f"Framework root: {framework_root}")
    _clear_runtime_selection()
    try:
        existing = read_json(marker) if marker.is_file() else {}
    except (OSError, ValueError, json.JSONDecodeError):
        existing = {}
    reusable = (
        existing.get("state") == READY_STATE
        and existing.get("runtime_hash") == runtime_hash
        and existing.get("model_id") == model_id
        and python.is_file()
    )
    paths = _family_paths(drive, spec, framework_root=framework_root)
    if reusable:
        try:
            paths = _prepare_family(
                repo,
                drive,
                python,
                spec,
                framework_root=framework_root,
                install_kernel=False,
            )
            _, probe = _verify_runtime(
                python, repo, environment, spec, paths, quick=True
            )
        except Exception as error:
            _write_state(
                marker,
                runtime_hash=runtime_hash,
                model_id=model_id,
                family=family,
                state="FAILED",
                python=python,
                started_at=str(existing.get("started_at") or _now()),
                probe_path=environment / "quick_environment_probe.json",
                completed_at=_now(),
                failed_stage=_failed_stage(
                    error, environment / "quick_environment_probe.json"
                ),
                failure_type=type(error).__name__,
                failure_message=str(error),
                extra={
                    "requirements_hash": requirements_hash,
                    "framework_root": str(framework_root),
                },
            )
            _preserve_failed_runtime_diagnostics(
                environment, drive, model_id, runtime_hash
            )
            _safe_remove_environment(environment, base)
            reusable = False
        else:
            manifest = {
                **existing,
                "state": READY_STATE,
                "status": READY_STATE,
                "completed_at": _now(),
                "packages_changed": False,
                "source_commit": git_commit(repo),
                "log_directory": str(environment / "logs"),
                "environment": probe.get("hardware", probe),
                "verification": probe,
                "manifest_path": str(manifest_path),
                "requirements_hash": requirements_hash,
                "runtime_hash": runtime_hash,
                "framework_root": str(framework_root),
                "framework_checkouts": _framework_checkout_records(paths, spec),
            }
            write_json(marker, manifest)
            write_json(manifest_path, manifest)
            _select_runtime(python, manifest_path, paths)
            print("Environment cache: reused")
            return manifest

    if not reusable and environment.exists():
        _preserve_failed_runtime_diagnostics(environment, drive, model_id, runtime_hash)
        _safe_remove_environment(environment, base)
    started_at = _now()
    packages_changed = True
    try:
        _ensure_uv(str(spec["provisioner"]["version"]))
        environment.parent.mkdir(parents=True, exist_ok=True)
        _run(
            _uv(
                "venv",
                "--python",
                spec["python"],
                "--python-preference",
                "managed",
                environment,
            ),
            environment_name=family,
            stage="virtual_environment_creation",
            python_executable=python,
        )
        _write_state(
            marker,
            runtime_hash=runtime_hash,
            model_id=model_id,
            family=family,
            state="CREATING",
            python=python,
            started_at=started_at,
            probe_path=probe_path,
        )
        if not python.is_file():
            raise FileNotFoundError(f"managed Python executable was not created: {python}")
        _write_state(
            marker,
            runtime_hash=runtime_hash,
            model_id=model_id,
            family=family,
            state="INSTALLING",
            python=python,
            started_at=started_at,
            probe_path=probe_path,
        )
        _install_runtime(repo, environment, python, spec)
        paths = _prepare_family(
            repo,
            drive,
            python,
            spec,
            framework_root=framework_root,
        )
        _write_state(
            marker,
            runtime_hash=runtime_hash,
            model_id=model_id,
            family=family,
            state="VERIFYING",
            python=python,
            started_at=started_at,
            probe_path=probe_path,
        )
        probe_path, probe = _verify_runtime(
            python, repo, environment, spec, paths, quick=False
        )
        manifest = _write_state(
            marker,
            runtime_hash=runtime_hash,
            model_id=model_id,
            family=family,
            state=READY_STATE,
            python=python,
            started_at=started_at,
            completed_at=_now(),
            probe_path=probe_path,
            extra={
                "dependency_lock_hash": runtime_hash,
                "requirements_hash": requirements_hash,
                "source_commit": git_commit(repo),
                "log_directory": str(environment / "logs"),
                "packages_changed": packages_changed,
                "environment": probe.get("hardware", probe),
                "verification": probe,
                "sources": spec["sources"],
                "framework_root": str(framework_root),
                "framework_checkouts": _framework_checkout_records(paths, spec),
                "framework_provisioning_version": FRAMEWORK_PROVISIONING_VERSION,
                "package_contract": {
                    key: spec[key]
                    for key in ("python", "torch", "torchvision", "requirements")
                },
                "compatibility_claim": (
                    "Exact package, CUDA, compiled-operation, source, and family "
                    "probe passed; real training correctness still requires the "
                    "adapter GPU smoke gate."
                ),
                "manifest_path": str(manifest_path),
            },
        )
        write_json(manifest_path, manifest)
        _select_runtime(python, manifest_path, paths)
        print("Environment cache: rebuilt")
        return manifest
    except Exception as error:
        environment.mkdir(parents=True, exist_ok=True)
        stage = _failed_stage(error, probe_path)
        _write_state(
            marker,
            runtime_hash=runtime_hash,
            model_id=model_id,
            family=family,
            state="FAILED",
            python=python,
            started_at=started_at,
            completed_at=_now(),
            probe_path=probe_path,
            failed_stage=stage,
            failure_type=type(error).__name__,
            failure_message=str(error),
            extra={
                "requirements_hash": requirements_hash,
                "framework_root": str(framework_root),
                "framework_provisioning_version": FRAMEWORK_PROVISIONING_VERSION,
            },
        )
        _clear_runtime_selection()
        print("Environment cache: rebuild failed")
        label = "VMamba" if family == "vmamba" else family
        raise EnvironmentProvisioningError(
            f"{label} environment verification failed\n"
            f"Stage: {stage}\n"
            f"Python: {python}\n"
            f"Probe output: {probe_path}\n"
            f"Cause: {error}\n"
            f"Suggested action: rerun the notebook to rebuild only {environment}.\n"
            "Google Drive experiment artifacts were not modified."
        ) from error
