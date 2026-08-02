"""Safe command construction for repository-owned Python entry points."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from src.utils.serialization import write_text_atomic

MODEL_PYTHON_ENV = "VISDRONE_MODEL_PYTHON"
HEADLESS_BACKEND = "Agg"
DETERMINISTIC_CUBLAS_WORKSPACE = ":4096:8"

NOTEBOOK_ONLY_ENVIRONMENT_VARIABLES = (
    "IPYTHONDIR",
    "JUPYTER_PATH",
    "PYTHONSTARTUP",
    "DISPLAY",
)

SINGLE_PROCESS_DISTRIBUTED_VARIABLES = (
    "WORLD_SIZE",
    "LOCAL_RANK",
    "RANK",
)

RECORDED_SUBPROCESS_VARIABLES = (
    "MPLBACKEND",
    "MPLCONFIGDIR",
    "PYTHONPATH",
    "IPYTHONDIR",
    "JUPYTER_PATH",
    "DISPLAY",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
    "TORCH_HOME",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "RANK",
)

SENSITIVE_ARGUMENTS = {
    "--password",
    "--token",
    "--access-token",
    "--auth-token",
    "--api-key",
}
OUTPUT_CREDENTIAL_URL = re.compile(r"(https?://)[^/@\s]+@", re.IGNORECASE)
OUTPUT_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)"
    r"\s*[:=]\s*[^\s]+"
)


def _redacted_output(value: str) -> str:
    value = OUTPUT_CREDENTIAL_URL.sub(r"\1<redacted>@", value)
    return OUTPUT_SECRET_ASSIGNMENT.sub(r"\1=<redacted>", value)


def _redacted_command(command: Sequence[object]) -> list[str]:
    rendered: list[str] = []
    redact_next = False
    for value in command:
        argument = str(value)
        if redact_next:
            rendered.append("<redacted>")
            redact_next = False
            continue
        lowered = argument.lower()
        if lowered in SENSITIVE_ARGUMENTS:
            rendered.append(argument)
            redact_next = True
            continue
        if any(lowered.startswith(f"{flag}=") for flag in SENSITIVE_ARGUMENTS):
            rendered.append(argument.split("=", 1)[0] + "=<redacted>")
            continue
        if "://" in argument:
            parsed = urlsplit(argument)
            if parsed.username is not None or parsed.password is not None:
                hostname = parsed.hostname or ""
                if parsed.port:
                    hostname = f"{hostname}:{parsed.port}"
                argument = urlunsplit(
                    (parsed.scheme, hostname, parsed.path, parsed.query, parsed.fragment)
                )
        rendered.append(argument)
    return rendered


class CheckedSubprocessError(RuntimeError):
    """Actionable, redacted context for a failed repository subprocess."""

    def __init__(
        self,
        command: Sequence[object],
        *,
        cwd: str | Path | None,
        returncode: int | None,
        stdout: str,
        stderr: str,
        environment_name: str | None = None,
        stage: str | None = None,
        python_executable: str | Path | None = None,
        probe_path: str | Path | None = None,
    ) -> None:
        self.command = _redacted_command(command)
        self.cwd = str(Path(cwd).resolve()) if cwd is not None else os.getcwd()
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.environment_name = environment_name
        self.stage = stage
        self.python_executable = (
            str(python_executable) if python_executable is not None else None
        )
        self.probe_path = str(probe_path) if probe_path is not None else None
        lines = [
            f"{environment_name or 'repository'} subprocess failed",
            f"Stage: {stage or 'unspecified'}",
            f"Command: {subprocess.list2cmdline(self.command)}",
            f"Working directory: {self.cwd}",
            f"Return code: {returncode if returncode is not None else 'not started'}",
        ]
        if self.python_executable:
            lines.append(f"Python: {self.python_executable}")
        if self.probe_path:
            lines.append(f"Probe output: {self.probe_path}")
        lines.extend(
            [
                "--- stdout ---",
                stdout.rstrip() or "<empty>",
                "--- stderr ---",
                stderr.rstrip() or "<empty>",
            ]
        )
        super().__init__("\n".join(lines))


def run_checked(
    command: Sequence[object],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    environment_name: str | None = None,
    stage: str | None = None,
    python_executable: str | Path | None = None,
    probe_path: str | Path | None = None,
    log_path: str | Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    """Run a command, surface its output, and raise complete redacted diagnostics."""
    arguments = [str(value) for value in command]
    try:
        completed = runner(
            arguments,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            check=False,
            capture_output=True,
            text=True,
        )
        stdout = _redacted_output(completed.stdout or "")
        stderr = _redacted_output(completed.stderr or "")
        returncode: int | None = completed.returncode
    except OSError as error:
        stdout = ""
        stderr = _redacted_output(f"{type(error).__name__}: {error}")
        returncode = None
        completed = subprocess.CompletedProcess(arguments, 1, stdout, stderr)
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)
    if log_path is not None:
        rendered = (
            f"Command: {subprocess.list2cmdline(_redacted_command(arguments))}\n"
            f"Working directory: {Path(cwd).resolve() if cwd is not None else Path.cwd()}\n"
            f"Return code: {returncode if returncode is not None else 'not started'}\n"
            "--- stdout ---\n"
            f"{stdout.rstrip() or '<empty>'}\n"
            "--- stderr ---\n"
            f"{stderr.rstrip() or '<empty>'}\n"
        )
        write_text_atomic(log_path, rendered)
    if returncode != 0:
        raise CheckedSubprocessError(
            arguments,
            cwd=cwd,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            environment_name=environment_name,
            stage=stage,
            python_executable=python_executable,
            probe_path=probe_path,
        )
    return completed


def model_python_executable() -> str:
    return os.environ.get(MODEL_PYTHON_ENV, sys.executable)


def python_module_command(module: str, *arguments: object) -> list[str]:
    """Return a Python module command and reject filepath-style entry points."""
    if not module or module.endswith(".py") or "/" in module or "\\" in module:
        raise ValueError(
            f"Repository Python entry points must use dotted module names, got {module!r}"
        )
    executable = model_python_executable()
    return [executable, "-m", module, *(str(value) for value in arguments)]


def build_model_subprocess_environment(
    parent: Mapping[str, str] | None = None,
    *,
    matplotlib_config_dir: str | Path | None = None,
) -> dict[str, str]:
    """Build a single-process model environment without notebook-only leakage."""
    environment = dict(os.environ if parent is None else parent)
    for variable in NOTEBOOK_ONLY_ENVIRONMENT_VARIABLES:
        environment.pop(variable, None)
    # The repository launches one backend process directly, never through torchrun.
    for variable in SINGLE_PROCESS_DISTRIBUTED_VARIABLES:
        environment.pop(variable, None)
    # The isolated runtime is editable-installed and launched from the repository;
    # inheriting the notebook kernel's PYTHONPATH can shadow its pinned packages.
    environment.pop("PYTHONPATH", None)
    environment["MPLBACKEND"] = HEADLESS_BACKEND
    environment["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE
    if matplotlib_config_dir is not None:
        config_dir = Path(matplotlib_config_dir).resolve()
        config_dir.mkdir(parents=True, exist_ok=True)
        environment["MPLCONFIGDIR"] = str(config_dir)
    return environment


def configure_headless_matplotlib() -> None:
    """Force a noninteractive backend before pyplot is imported."""
    os.environ["MPLBACKEND"] = HEADLESS_BACKEND
    import matplotlib

    matplotlib.use(HEADLESS_BACKEND, force=True)


def subprocess_environment_record(
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return the non-secret environment policy fields used by model processes."""
    values = os.environ if environment is None else environment
    return {
        "policy": "single_process_headless_v1",
        "variables": {
            variable: values.get(variable)
            for variable in RECORDED_SUBPROCESS_VARIABLES
        },
    }
