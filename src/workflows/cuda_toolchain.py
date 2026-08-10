"""Select a CUDA toolkit and host compiler that can build the pinned extension.

VMamba's ``selective_scan_cuda_oflex`` is a Torch CUDA extension.
``torch.utils.cpp_extension`` refuses to build when the toolkit's major version
differs from the one PyTorch was compiled against, so the pinned
``torch 2.1.0+cu118`` runtime genuinely requires an 11.x ``nvcc`` - and the
pinned torch/CUDA pair is itself forced by the only MMCV 2.1.0 wheels that
exist (``cu118/torch2.1``). Removing the compatibility check would just move the
failure into the compiler.

So instead of relaxing the check, this module finds a compatible toolkit that
already exists on the host, or produces the exact commands to install one. It
never rewrites ``/usr/local/cuda`` or any other global toolkit selection: the
chosen paths are exported only into the build subprocess.

The parsing and selection logic here is pure and unit-tested on CPU; the
compilation it configures is validated only on a GPU host.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

CUDA_RELEASE = re.compile(r"release\s+(\d+)\.(\d+)")
COMPILER_VERSION = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")
# Environment variable an operator can set to point at a toolkit the automatic
# search cannot find (for example a user-local extraction of the runfile).
CUDA_HOME_OVERRIDE = "VISDRONE_CUDA_HOME"
# Set to 0 to refuse any automatic toolkit installation on a hosted runtime.
CUDA_AUTOINSTALL_ENV = "VISDRONE_INSTALL_CUDA_TOOLKIT"

CommandProbe = Callable[[Sequence[str]], str]


class CudaToolchainError(RuntimeError):
    """No toolchain on this host can build the pinned CUDA extension."""


@dataclass(frozen=True)
class CudaToolkit:
    home: Path
    nvcc: Path
    version: str

    @property
    def major(self) -> int:
        return int(self.version.split(".", 1)[0])

    def as_dict(self) -> dict[str, str]:
        return {
            "cuda_home": str(self.home),
            "nvcc": str(self.nvcc),
            "nvcc_version": self.version,
        }


@dataclass(frozen=True)
class HostCompiler:
    cxx: Path
    cc: Path | None
    version: str

    @property
    def major(self) -> int:
        return int(self.version.split(".", 1)[0])

    def as_dict(self) -> dict[str, str | None]:
        return {
            "cxx": str(self.cxx),
            "cc": str(self.cc) if self.cc else None,
            "compiler_version": self.version,
        }


def _default_probe(command: Sequence[str]) -> str:
    completed = subprocess.run(
        [str(value) for value in command],
        check=False,
        capture_output=True,
        text=True,
    )
    return f"{completed.stdout}\n{completed.stderr}"


def parse_cuda_version(text: str) -> str | None:
    """Read ``major.minor`` from ``nvcc --version`` output."""
    match = CUDA_RELEASE.search(text or "")
    return f"{match.group(1)}.{match.group(2)}" if match else None


def parse_compiler_version(text: str) -> str | None:
    """Read a GCC/Clang ``major[.minor[.patch]]`` version string."""
    match = COMPILER_VERSION.search((text or "").strip())
    if not match:
        return None
    return ".".join(part for part in match.groups() if part is not None)


def cuda_versions_compatible(host_cuda: str, torch_cuda: str) -> bool:
    """Torch CUDA extensions require a matching CUDA *major* version."""
    return host_cuda.split(".", 1)[0] == torch_cuda.split(".", 1)[0]


def _nvcc_path(home: Path) -> Path:
    return home / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")


def candidate_cuda_homes(
    search_paths: Sequence[str | Path],
    *,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> list[Path]:
    """Ordered CUDA_HOME candidates: explicit override first, PATH last."""
    values = os.environ if environ is None else environ
    candidates: list[Path] = []
    for variable in (CUDA_HOME_OVERRIDE, "CUDA_HOME", "CUDA_PATH"):
        configured = values.get(variable)
        if configured:
            candidates.append(Path(configured))
    candidates.extend(Path(str(value)) for value in search_paths)
    located = which("nvcc")
    if located:
        candidates.append(Path(located).resolve().parent.parent)
    ordered: list[Path] = []
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def select_cuda_toolkit(
    *,
    torch_cuda: str,
    search_paths: Sequence[str | Path],
    environ: Mapping[str, str] | None = None,
    probe: CommandProbe | None = None,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[Path], bool] = Path.is_file,
) -> CudaToolkit:
    """Return the first toolkit whose CUDA major version matches PyTorch's."""
    runner = probe or _default_probe
    rejected: list[str] = []
    for home in candidate_cuda_homes(search_paths, environ=environ, which=which):
        nvcc = _nvcc_path(home)
        if not exists(nvcc):
            continue
        version = parse_cuda_version(runner([str(nvcc), "--version"]))
        if version is None:
            rejected.append(f"{nvcc}: nvcc --version reported no release")
            continue
        if not cuda_versions_compatible(version, torch_cuda):
            rejected.append(
                f"{nvcc}: CUDA {version} cannot build a CUDA {torch_cuda} extension"
            )
            continue
        return CudaToolkit(home=home, nvcc=nvcc, version=version)
    raise CudaToolchainError(
        f"No CUDA {torch_cuda.split('.', 1)[0]}.x toolkit was found for the pinned "
        f"PyTorch CUDA {torch_cuda} runtime."
        + (f" Rejected candidates: {rejected}." if rejected else "")
    )


def select_host_compiler(
    *,
    candidates: Sequence[str],
    maximum_major: int,
    which: Callable[[str], str | None] = shutil.which,
    probe: CommandProbe | None = None,
) -> HostCompiler:
    """Return the first C++ compiler the CUDA toolkit accepts.

    CUDA 11.8 rejects GCC 12 and newer, which is exactly what a modern hosted
    image ships as the default ``g++``.
    """
    runner = probe or _default_probe
    rejected: list[str] = []
    for name in candidates:
        located = which(name)
        if not located:
            continue
        version = parse_compiler_version(
            runner([located, "-dumpfullversion", "-dumpversion"])
        )
        if version is None:
            rejected.append(f"{located}: version could not be determined")
            continue
        if int(version.split(".", 1)[0]) > maximum_major:
            rejected.append(
                f"{located}: g++ {version} exceeds the maximum supported major "
                f"version {maximum_major}"
            )
            continue
        c_name = name.replace("g++", "gcc") if "g++" in name else None
        c_compiler = which(c_name) if c_name else None
        return HostCompiler(
            cxx=Path(located),
            cc=Path(c_compiler) if c_compiler else None,
            version=version,
        )
    raise CudaToolchainError(
        "No supported host C++ compiler was found. CUDA requires GCC "
        f"{maximum_major} or older; tried {list(candidates)}."
        + (f" Rejected: {rejected}." if rejected else "")
    )


def apt_install_commands(packages: Sequence[str]) -> list[list[str]]:
    """The exact, bounded commands that install the missing toolkit packages."""
    if not packages:
        return []
    return [
        ["apt-get", "update", "-qq"],
        ["apt-get", "install", "-y", "--no-install-recommends", *packages],
    ]


def automatic_install_allowed(
    platform: str,
    allowed_platforms: Sequence[str],
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Only disposable hosted runtimes install system packages, and only opt-out."""
    values = os.environ if environ is None else environ
    configured = values.get(CUDA_AUTOINSTALL_ENV, "").strip().lower()
    if configured in {"0", "false", "no"}:
        return False
    if configured in {"1", "true", "yes"}:
        return True
    return platform in set(allowed_platforms)


def render_install_script(packages: Sequence[str]) -> str:
    """Render the install commands as copy-pasteable shell lines.

    One command per line: ``apt-get update`` and ``apt-get install`` are separate
    invocations, so joining them into a single line produces
    ``apt-get update -qq apt-get install ...``, which apt rejects with
    ``E: Invalid operation``. The operator reaching this text has already been
    blocked once; the instructions must run as written.
    """
    commands = apt_install_commands(list(packages))
    if not commands:
        return "  (no toolkit packages are configured for this runtime)"
    return "\n".join(f"  sudo {' '.join(command)}" for command in commands)


def remediation_message(
    *,
    torch_cuda: str,
    required_version: str,
    packages: Sequence[str],
    search_paths: Sequence[str | Path],
) -> str:
    """Exact commands for an operator whose host cannot be fixed automatically."""
    return (
        f"VMamba's selective_scan extension must be compiled with CUDA "
        f"{required_version} to match the pinned PyTorch CUDA {torch_cuda} build.\n"
        "On a Debian/Ubuntu host (Colab and Kaggle included), install the minimal "
        "toolkit by running these commands in order:\n"
        f"{render_install_script(packages)}\n"
        "Then rerun the provisioning cell. If the toolkit lives elsewhere, point "
        f"the provisioner at it with {CUDA_HOME_OVERRIDE}=/path/to/cuda-"
        f"{required_version}.\n"
        f"Searched: {[str(path) for path in search_paths]}."
    )


def build_environment(
    base: Mapping[str, str], toolkit: CudaToolkit, compiler: HostCompiler
) -> dict[str, str]:
    """Point one build subprocess at the selected toolchain, nothing global."""
    environment = dict(base)
    environment["CUDA_HOME"] = str(toolkit.home)
    environment["CUDA_PATH"] = str(toolkit.home)
    environment["PATH"] = os.pathsep.join(
        [str(toolkit.home / "bin"), environment.get("PATH", "")]
    ).rstrip(os.pathsep)
    environment["CXX"] = str(compiler.cxx)
    if compiler.cc is not None:
        environment["CC"] = str(compiler.cc)
    # nvcc must use the same host compiler the extension is linked with.
    prepend = environment.get("NVCC_PREPEND_FLAGS", "")
    environment["NVCC_PREPEND_FLAGS"] = (
        f"{prepend} -ccbin {compiler.cxx}".strip()
    )
    return environment
