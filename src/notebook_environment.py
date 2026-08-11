"""One environment and path contract for Colab, Kaggle, and local notebooks."""
from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from src.git_utils import is_git_checkout
from src.notebook_utils import in_colab, in_kaggle

SUPPORTED_NOTEBOOK_PLATFORMS = {"colab", "kaggle", "local", "ci"}
HOSTED_NOTEBOOK_PLATFORMS = frozenset({"colab", "kaggle"})
REPOSITORY_NAME = "aerial-object-detection-benchmark"
# The only path on a Colab runtime that outlives the session.
DRIVE_MOUNTPOINT = "/content/drive"
# Kaggle persists /kaggle/working as the notebook's output and caps it at 20 GB.
# /kaggle/temp is session scratch on the same larger disk and is not part of that
# quota, so it is the Kaggle equivalent of Colab's /content: the right home for
# anything rebuildable. Only real artifacts - dataset, studies, checkpoints -
# belong in the persisted budget.
KAGGLE_OUTPUT_ROOT = "/kaggle/working"
KAGGLE_SCRATCH_ROOT = "/kaggle/temp"
ARTIFACT_ROOT_PERSISTENT = "persistent"
ARTIFACT_ROOT_OPERATOR_DECLARED = "operator_declared"
ARTIFACT_ROOT_SESSION = "session"
# Opt-in switch for the rare case of deliberately applying hosted pins to a
# local interpreter. Absent it, a local notebook never rewrites the developer's
# environment.
LOCAL_INSTALL_OPT_IN = "VISDRONE_ALLOW_LOCAL_DEPENDENCY_INSTALL"
# Compiled extension modules whose ABI is fixed at import time. Replacing one of
# these underneath a live kernel is what produces the "numpy.dtype size changed"
# class of failure, so a version change to an already-imported package must stop
# the notebook and ask for a restart instead of continuing.
ABI_SENSITIVE_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "torch": "torch",
    "torchvision": "torchvision",
    "pyarrow": "pyarrow",
}
_REQUIREMENT_PIN = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*(?P<version>[^\s;#]+)"
)


def detect_notebook_platform(
    environ: Mapping[str, str] | None = None,
) -> str:
    values = os.environ if environ is None else environ
    override = values.get("VISDRONE_NOTEBOOK_PLATFORM", "").strip().lower()
    if override:
        if override not in SUPPORTED_NOTEBOOK_PLATFORMS:
            raise ValueError(
                "VISDRONE_NOTEBOOK_PLATFORM must be colab, kaggle, or local"
            )
        return override
    if environ is None and in_colab():
        return "colab"
    if values.get("COLAB_RELEASE_TAG") or values.get("COLAB_BACKEND_VERSION"):
        return "colab"
    if environ is None and in_kaggle():
        return "kaggle"
    if values.get("KAGGLE_KERNEL_RUN_TYPE") or values.get("KAGGLE_URL_BASE"):
        return "kaggle"
    return "local"


def default_repository_root(platform: str, cwd: str | Path | None = None) -> Path:
    if platform == "colab":
        return Path("/content") / REPOSITORY_NAME
    if platform == "kaggle":
        return Path("/kaggle/working") / REPOSITORY_NAME
    current = Path(cwd or Path.cwd()).expanduser().resolve()
    if (current / "pyproject.toml").is_file() and (current / "src").is_dir():
        return current
    return current / REPOSITORY_NAME


def is_git_worktree(path: str | Path) -> bool:
    """Recognize normal clones and linked worktrees without assuming `.git` is a dir."""
    return is_git_checkout(Path(path).expanduser().resolve())


def default_artifact_root(
    platform: str,
    repository_root: str | Path,
    *,
    use_google_drive: bool = True,
) -> Path:
    if platform == "colab" and use_google_drive:
        return Path("/content/drive/MyDrive/visdrone_architecture_benchmark")
    if platform == "kaggle":
        return Path("/kaggle/working/visdrone_architecture_benchmark")
    return Path(repository_root).resolve() / "local_artifacts"


def artifact_root_persistence(
    platform: str,
    artifact_root: str | Path,
    *,
    operator_configured: bool = False,
) -> str:
    """Classify whether artifacts written here survive the end of the session.

    Three states, not two. On Colab only a mounted Drive is *known* to persist
    and everything else under ``/content`` is *known* to be session storage - but
    an operator who sets ``VISDRONE_DRIVE_ROOT`` to a bucket or persistent-disk
    mount has made a deliberate declaration this process cannot verify. Calling
    that "ephemeral" would contradict the remediation ``DriveUnavailableError``
    itself offers, so it gets its own state.
    """
    resolved = Path(artifact_root).expanduser()
    if platform != "colab":
        # Kaggle's /kaggle/working is captured as notebook output; local is a disk.
        return ARTIFACT_ROOT_PERSISTENT
    mountpoint = Path(DRIVE_MOUNTPOINT)
    if resolved == mountpoint or mountpoint in resolved.parents:
        return ARTIFACT_ROOT_PERSISTENT
    if operator_configured:
        return ARTIFACT_ROOT_OPERATOR_DECLARED
    return ARTIFACT_ROOT_SESSION


def artifact_root_is_persistent(
    platform: str,
    artifact_root: str | Path,
    *,
    operator_configured: bool = False,
) -> bool:
    """Whether artifacts written here are expected to outlive the session."""
    return (
        artifact_root_persistence(
            platform, artifact_root, operator_configured=operator_configured
        )
        != ARTIFACT_ROOT_SESSION
    )


def default_local_cache_root(platform: str, repository_root: str | Path) -> Path:
    if platform == "colab":
        return Path("/content/visdrone_cache")
    if platform == "kaggle":
        return Path(KAGGLE_SCRATCH_ROOT) / "visdrone_cache"
    return Path(repository_root).resolve() / ".notebook-cache" / "visdrone"


def default_model_runtime_root(platform: str) -> Path:
    if platform == "colab":
        return Path("/content/visdrone_model_envs")
    if platform == "kaggle":
        return Path(KAGGLE_SCRATCH_ROOT) / "visdrone_model_envs"
    return Path(tempfile.gettempdir()).resolve() / "visdrone_model_envs"


def default_hpo_scratch_root(platform: str) -> Path:
    if platform == "colab":
        return Path("/content/visdrone_hpo_trials")
    if platform == "kaggle":
        return Path(KAGGLE_SCRATCH_ROOT) / "visdrone_hpo_trials"
    return Path(tempfile.gettempdir()).resolve() / "visdrone_hpo_trials"


class KernelRestartRequired(RuntimeError):
    """A binary dependency changed underneath this kernel; restart and rerun."""


@dataclass(frozen=True)
class NotebookEnvironment:
    platform: str
    repository_root: Path
    artifact_root: Path
    local_cache_root: Path
    model_runtime_root: Path
    hpo_scratch_root: Path
    dependencies_installed: bool = False
    dependency_decision: str = ""
    # Carried as real state rather than reported as a literal by ``as_dict``.
    # ``setup_notebook_environment`` raises ``KernelRestartRequired`` instead of
    # returning when a restart is pending, so a *returned* environment always
    # records False - but any future caller that constructs one differently has
    # to state the value, and cannot silently report a stale False.
    restart_required: bool = False

    @property
    def hosted(self) -> bool:
        return self.platform in HOSTED_NOTEBOOK_PLATFORMS

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "platform": self.platform,
            "hosted": self.hosted,
            "repository_root": str(self.repository_root),
            "artifact_root": str(self.artifact_root),
            "local_cache_root": str(self.local_cache_root),
            "model_runtime_root": str(self.model_runtime_root),
            "hpo_scratch_root": str(self.hpo_scratch_root),
            "dependencies_installed": self.dependencies_installed,
            "dependency_decision": self.dependency_decision,
            "restart_required": self.restart_required,
        }


class DriveUnavailableError(RuntimeError):
    """Drive was requested as the artifact root but this session cannot mount it."""


def _mount_google_drive(platform: str, requested: bool) -> None:
    """Mount Drive, or explain the choice instead of guessing one.

    Some Colab session types - draft/preview sessions, restricted and enterprise
    runtimes, local runtimes - refuse ``drive.mount`` with ``NotImplementedError``
    because ``/var/colab/hostname`` is absent. Falling back to local storage
    automatically would be the worst possible response: the artifact root holds
    the dataset, the Optuna studies, and every checkpoint, and a multi-hour
    search would be silently written somewhere that disappears on disconnect.
    So this fails closed and names the options.
    """
    if platform != "colab" or not requested:
        return
    try:
        from google.colab import drive
    except ImportError as error:
        raise RuntimeError("Google Drive mounting is available only in Colab") from error
    try:
        drive.mount(DRIVE_MOUNTPOINT)
    except (NotImplementedError, OSError, ValueError) as error:
        raise DriveUnavailableError(
            f"This Colab session cannot mount Google Drive: {error}\n\n"
            "Drive is the persistent artifact root: it holds the prepared "
            "dataset, the Optuna study database, and every checkpoint. Running "
            "without it is a real choice, not a detail, so it is never made "
            "automatically.\n\n"
            "Colab Enterprise / Vertex AI notebooks never support drive.mount, "
            "and neither do local runtimes: reconnecting will not help there. "
            "You are on Colab Enterprise if the menu bar reads File/Edit/View/"
            "Run/Settings/Add-ons rather than the standard "
            "File/Edit/View/Insert/Runtime/Tools/Help.\n\n"
            "Pick one:\n"
            "1. Use standard Colab (colab.research.google.com) or Kaggle. Both "
            "give a persistent artifact root with no extra setup, and this "
            "project supports both. Kaggle needs no Drive at all - "
            "/kaggle/working persists. This is the option that keeps your work.\n"
            "2. Stay here and set VISDRONE_DRIVE_ROOT to a persistent directory "
            "you have already mounted yourself - a GCS bucket via gcsfuse, or an "
            "attached persistent disk - then rerun. Verify it survives a session "
            "restart before starting a long run.\n"
            "3. Set USE_GOOGLE_DRIVE = False in the notebook's first cell to "
            "run against local session storage. Everything - dataset, studies, "
            "checkpoints - is DELETED when the session ends, so use this only "
            "for a short smoke run, never for an HPO or final-training run."
        ) from error


def parse_pinned_requirements(text: str) -> dict[str, str]:
    """Return the ``name==version`` pins declared by a requirements file."""
    pins: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            continue
        match = _REQUIREMENT_PIN.match(stripped)
        if match:
            pins[match.group("name").lower().replace("_", "-")] = match.group("version")
    return pins


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def restart_required_packages(
    pins: Mapping[str, str],
    *,
    installed: Mapping[str, str | None] | None = None,
    imported_modules: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Report ABI-sensitive packages that would change under a live kernel.

    Installing a different build of an already-imported compiled package leaves
    the running process holding stale C extensions. Rather than guessing, the
    caller stops and asks for one restart.
    """
    modules = set(sys.modules if imported_modules is None else imported_modules)
    changes: list[dict[str, str]] = []
    for distribution, module in sorted(ABI_SENSITIVE_PACKAGES.items()):
        required = pins.get(distribution)
        if not required:
            continue
        current = (
            _installed_version(distribution)
            if installed is None
            else installed.get(distribution)
        )
        if current is None or current == required:
            continue
        if module not in modules:
            continue
        changes.append(
            {
                "package": distribution,
                "installed": current,
                "required": required,
                "imported_module": module,
            }
        )
    return changes


def _install_shared_dependencies(
    repository_root: Path,
    requirements_file: str | Path | None,
) -> None:
    if requirements_file:
        requirement = repository_root / requirements_file
        if not requirement.is_file():
            raise FileNotFoundError(requirement)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirement)],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(repository_root),
            "--no-deps",
            "--no-build-isolation",
        ],
        check=True,
    )


def dependency_installation_decision(
    platform: str,
    requirements_file: str | Path | None,
    *,
    install_dependencies: bool | None,
    environ: Mapping[str, str],
) -> tuple[bool, str]:
    """Decide whether this platform may install the hosted dependency pins.

    Hosted runtimes are disposable and are the environments the pins were
    written for. A local checkout is the developer's own interpreter: installing
    Colab pins there can silently downgrade `.venv`, so it requires an explicit
    opt-in.
    """
    if install_dependencies is not None:
        return bool(install_dependencies), (
            "install_dependencies was set explicitly by the caller"
        )
    if platform in HOSTED_NOTEBOOK_PLATFORMS:
        return True, f"{platform} is a disposable hosted runtime"
    if not requirements_file:
        return False, "no requirements file is configured"
    if environ.get(LOCAL_INSTALL_OPT_IN, "").strip().lower() in {"1", "true", "yes"}:
        return True, f"{LOCAL_INSTALL_OPT_IN} opts this interpreter in"
    return False, (
        f"{platform} execution never installs hosted pins into the active "
        f"interpreter; create a matching environment yourself, or set "
        f"{LOCAL_INSTALL_OPT_IN}=1 to accept the change"
    )


def setup_notebook_environment(
    repository_root: str | Path,
    *,
    artifact_root: str | Path | None = None,
    use_google_drive: bool = True,
    requirements_file: str | Path | None = None,
    install_dependencies: bool | None = None,
    smoke_test: bool = False,
    platform: str | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> NotebookEnvironment:
    """Resolve one runtime contract without changing benchmark semantics."""
    values = os.environ if environ is None else environ
    selected_platform = platform or detect_notebook_platform(
        None if environ is None else values
    )
    if selected_platform not in SUPPORTED_NOTEBOOK_PLATFORMS:
        raise ValueError(f"unsupported notebook platform: {selected_platform}")
    repository = Path(repository_root).expanduser().resolve()
    if not (repository / "pyproject.toml").is_file() or not (
        repository / "src" / "__init__.py"
    ).is_file():
        raise FileNotFoundError(f"not a benchmark repository: {repository}")

    # Dependencies are installed before Google Drive is mounted: importing the
    # Colab tooling loads NumPy, and replacing NumPy underneath an imported
    # module is the ABI hazard this ordering removes.
    should_install, decision = dependency_installation_decision(
        selected_platform,
        requirements_file,
        install_dependencies=install_dependencies,
        environ=values,
    )
    print(f"Dependency installation: {'yes' if should_install else 'no'} ({decision})")
    if should_install:
        pins = (
            parse_pinned_requirements(
                (repository / requirements_file).read_text(encoding="utf-8")
            )
            if requirements_file
            else {}
        )
        pending_restart = restart_required_packages(pins)
        _install_shared_dependencies(repository, requirements_file)
        if pending_restart:
            rendered = "\n- ".join(
                f"{change['package']}: {change['installed']} -> {change['required']} "
                f"(module {change['imported_module']} is already imported)"
                for change in pending_restart
            )
            raise KernelRestartRequired(
                "RESTART REQUIRED. These packages were replaced while this kernel "
                "already held their compiled modules:\n- "
                + rendered
                + "\n\nRestart the runtime (Runtime > Restart session in Colab, "
                "Run > Restart kernel in Kaggle/Jupyter) and run all cells again. "
                "The new versions are already installed, so the rerun continues "
                "without stopping here. No experiment artifact was modified."
            )

    _mount_google_drive(selected_platform, use_google_drive)
    configured_artifacts = artifact_root or values.get("VISDRONE_DRIVE_ROOT")
    artifacts = (
        Path(configured_artifacts).expanduser().resolve()
        if configured_artifacts
        else default_artifact_root(
            selected_platform,
            repository,
            use_google_drive=use_google_drive,
        ).resolve()
    )
    if smoke_test:
        artifacts = (repository / ".notebook-smoke" / "dataset").resolve()
    else:
        persistence = artifact_root_persistence(
            selected_platform,
            artifacts,
            operator_configured=bool(configured_artifacts),
        )
        if persistence == ARTIFACT_ROOT_SESSION:
            # Say so every run: an interrupted 8-epoch final run is recoverable
            # from Drive and unrecoverable from session storage.
            print(
                "WARNING: EPHEMERAL ARTIFACT ROOT\n"
                f"  {artifacts}\n"
                f"  is session storage on {selected_platform}, not Google Drive. "
                "The dataset, Optuna studies, and checkpoints written here are "
                "DELETED when this session ends, and an interrupted run cannot "
                "resume.\n"
                "  Acceptable for a smoke run. Not acceptable for HPO or final "
                "training - use a runtime that can mount Drive, or set "
                "VISDRONE_DRIVE_ROOT to a persistent directory."
            )
        elif persistence == ARTIFACT_ROOT_OPERATOR_DECLARED:
            # VISDRONE_DRIVE_ROOT was set deliberately. Whether that path really
            # persists is the operator's claim; this process cannot verify a
            # bucket or persistent-disk mount, so it reports rather than judges.
            print(
                f"Artifact root set by VISDRONE_DRIVE_ROOT: {artifacts}\n"
                "  Treated as persistent on your declaration - this is not "
                "verified. Confirm it survives a session restart before starting "
                "HPO or final training."
            )
    local_cache = Path(
        values.get(
            "VISDRONE_LOCAL_CACHE_ROOT",
            str(default_local_cache_root(selected_platform, repository)),
        )
    ).expanduser().resolve()
    model_runtime = Path(
        values.get(
            "VISDRONE_MODEL_ENV_ROOT",
            str(default_model_runtime_root(selected_platform)),
        )
    ).expanduser().resolve()
    hpo_scratch = Path(
        values.get(
            "VISDRONE_HPO_SCRATCH_ROOT",
            str(default_hpo_scratch_root(selected_platform)),
        )
    ).expanduser().resolve()

    for path in (artifacts, local_cache, model_runtime, hpo_scratch):
        path.mkdir(parents=True, exist_ok=True)
    values["BENCHMARK_REPO_ROOT"] = str(repository)
    values["VISDRONE_DRIVE_ROOT"] = str(artifacts)
    values["VISDRONE_LOCAL_CACHE_ROOT"] = str(local_cache)
    values["VISDRONE_MODEL_ENV_ROOT"] = str(model_runtime)
    values["VISDRONE_HPO_SCRATCH_ROOT"] = str(hpo_scratch)
    values["VISDRONE_NOTEBOOK_PLATFORM"] = selected_platform

    return NotebookEnvironment(
        platform=selected_platform,
        repository_root=repository,
        artifact_root=artifacts,
        local_cache_root=local_cache,
        model_runtime_root=model_runtime,
        hpo_scratch_root=hpo_scratch,
        dependencies_installed=should_install,
        dependency_decision=decision,
        # Reaching this point means the restart check above did not fire.
        restart_required=False,
    )
