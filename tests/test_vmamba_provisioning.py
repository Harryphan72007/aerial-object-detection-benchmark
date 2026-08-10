"""CUDA-toolchain selection and pretrained-checkpoint integrity for VMamba.

The compilation itself needs a GPU host. Everything that decides *whether* a
compilation may start - version parsing, major-version compatibility, CUDA_HOME
selection, compiler selection, and checkpoint verification - is pure and is
tested here on CPU.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.workflows.cuda_toolchain import (
    CUDA_AUTOINSTALL_ENV,
    CUDA_HOME_OVERRIDE,
    CudaToolchainError,
    CudaToolkit,
    HostCompiler,
    apt_install_commands,
    automatic_install_allowed,
    build_environment,
    candidate_cuda_homes,
    cuda_versions_compatible,
    parse_compiler_version,
    parse_cuda_version,
    remediation_message,
    select_cuda_toolkit,
    select_host_compiler,
)
from src.workflows.pretrained_checkpoints import (
    ALLOW_DOWNLOAD_ENV,
    CheckpointVerificationError,
    PretrainedCheckpointSpec,
    ensure_pretrained_checkpoint,
    family_pretrained_spec,
    load_pretrained_spec,
    verify_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
NVCC_118 = (
    "nvcc: NVIDIA (R) Cuda compiler driver\n"
    "Cuda compilation tools, release 11.8, V11.8.89\n"
)
NVCC_124 = (
    "nvcc: NVIDIA (R) Cuda compiler driver\n"
    "Cuda compilation tools, release 12.4, V12.4.131\n"
)


# --- CUDA version parsing and compatibility ----------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (NVCC_118, "11.8"),
        (NVCC_124, "12.4"),
        ("nvcc: command not found", None),
        ("", None),
    ],
)
def test_cuda_version_parsing(text: str, expected: str | None) -> None:
    assert parse_cuda_version(text) == expected


@pytest.mark.parametrize(
    ("host", "torch_cuda", "compatible"),
    [
        ("11.8", "11.8", True),
        ("11.7", "11.8", True),  # a Torch extension only requires a matching major
        ("12.4", "11.8", False),
        ("11.8", "12.8", False),
    ],
)
def test_cuda_major_compatibility_decision(
    host: str, torch_cuda: str, compatible: bool
) -> None:
    assert cuda_versions_compatible(host, torch_cuda) is compatible


@pytest.mark.parametrize(
    ("text", "expected"),
    [("11.4.0\n11\n", "11.4.0"), ("9\n", "9"), ("", None)],
)
def test_compiler_version_parsing(text: str, expected: str | None) -> None:
    assert parse_compiler_version(text) == expected


# --- CUDA_HOME selection ------------------------------------------------------


def test_explicit_override_is_searched_before_anything_else() -> None:
    homes = candidate_cuda_homes(
        ["/usr/local/cuda-11.8"],
        environ={CUDA_HOME_OVERRIDE: "/opt/mine", "CUDA_HOME": "/usr/local/cuda"},
        which=lambda _name: None,
    )
    assert homes[:3] == [
        Path("/opt/mine"),
        Path("/usr/local/cuda"),
        Path("/usr/local/cuda-11.8"),
    ]


def test_cuda_home_selection_skips_an_incompatible_major_version() -> None:
    # Keyed by CUDA_HOME so the assertions hold on any path separator.
    versions = {"cuda": NVCC_124, "cuda-11.8": NVCC_118}

    toolkit = select_cuda_toolkit(
        torch_cuda="11.8",
        search_paths=["/usr/local/cuda", "/usr/local/cuda-11.8"],
        environ={},
        which=lambda _name: None,
        exists=lambda path: path.parent.parent.name in versions,
        probe=lambda command: versions[Path(command[0]).parent.parent.name],
    )

    assert toolkit.home.name == "cuda-11.8"
    assert toolkit.version == "11.8"
    assert toolkit.major == 11


def test_no_compatible_toolkit_reports_every_rejected_candidate() -> None:
    with pytest.raises(CudaToolchainError, match="cannot build a CUDA 11.8 extension"):
        select_cuda_toolkit(
            torch_cuda="11.8",
            search_paths=["/usr/local/cuda"],
            environ={},
            which=lambda _name: None,
            exists=lambda _path: True,
            probe=lambda _command: NVCC_124,
        )


# --- host compiler selection --------------------------------------------------


def test_host_compiler_selection_rejects_a_too_new_default_gcc() -> None:
    available = {"g++-11": "/usr/bin/g++-11", "gcc-11": "/usr/bin/gcc-11", "g++": "/usr/bin/g++"}
    versions = {"/usr/bin/g++": "13.2.0\n13\n", "/usr/bin/g++-11": "11.4.0\n11\n"}

    compiler = select_host_compiler(
        candidates=["g++", "g++-11"],
        maximum_major=11,
        which=available.get,
        probe=lambda command: versions[command[0]],
    )

    assert compiler.cxx == Path("/usr/bin/g++-11")
    assert compiler.cc == Path("/usr/bin/gcc-11")
    assert compiler.major == 11


def test_no_supported_compiler_blocks_with_the_maximum_version() -> None:
    with pytest.raises(CudaToolchainError, match="11 or older"):
        select_host_compiler(
            candidates=["g++"],
            maximum_major=11,
            which=lambda _name: "/usr/bin/g++",
            probe=lambda _command: "13.2.0\n13\n",
        )


# --- installation policy and build environment --------------------------------


def test_automatic_toolkit_installation_is_hosted_only_and_opt_out() -> None:
    assert automatic_install_allowed("colab", ["colab", "kaggle"], environ={}) is True
    assert automatic_install_allowed("local", ["colab", "kaggle"], environ={}) is False
    assert (
        automatic_install_allowed(
            "colab", ["colab"], environ={CUDA_AUTOINSTALL_ENV: "0"}
        )
        is False
    )
    assert (
        automatic_install_allowed(
            "local", ["colab"], environ={CUDA_AUTOINSTALL_ENV: "1"}
        )
        is True
    )


def test_remediation_message_contains_runnable_commands() -> None:
    packages = ["cuda-nvcc-11-8", "g++-11"]
    message = remediation_message(
        torch_cuda="11.8",
        required_version="11.8",
        packages=packages,
        search_paths=["/usr/local/cuda-11.8"],
    )
    assert CUDA_HOME_OVERRIDE in message

    # Each apt invocation must stand alone on its own line. Concatenating them
    # yields "apt-get update -qq apt-get install ...", which apt rejects with
    # "E: Invalid operation" - a broken instruction on the one path whose entire
    # job is telling a blocked operator what to run.
    command_lines = [
        line.strip()
        for line in message.splitlines()
        if line.strip().startswith("sudo ")
    ]
    assert command_lines == [
        "sudo apt-get update -qq",
        "sudo apt-get install -y --no-install-recommends cuda-nvcc-11-8 g++-11",
    ]
    for line in command_lines:
        # One command per line: "apt-get" may appear exactly once.
        assert line.split().count("apt-get") == 1


def test_remediation_message_without_packages_emits_no_empty_command() -> None:
    message = remediation_message(
        torch_cuda="11.8",
        required_version="11.8",
        packages=[],
        search_paths=["/usr/local/cuda-11.8"],
    )
    assert apt_install_commands([]) == []
    assert not [
        line for line in message.splitlines() if line.strip().startswith("sudo ")
    ]
    assert "no toolkit packages are configured" in message


def test_build_environment_never_mutates_the_global_toolkit() -> None:
    toolkit = CudaToolkit(
        home=Path("/usr/local/cuda-11.8"),
        nvcc=Path("/usr/local/cuda-11.8/bin/nvcc"),
        version="11.8",
    )
    compiler = HostCompiler(
        cxx=Path("/usr/bin/g++-11"), cc=Path("/usr/bin/gcc-11"), version="11.4.0"
    )
    base = {"PATH": "/usr/bin", "CUDA_HOME": "/usr/local/cuda"}

    environment = build_environment(base, toolkit, compiler)

    assert environment["CUDA_HOME"] == str(toolkit.home)
    assert environment["PATH"].startswith(str(toolkit.home / "bin"))
    assert environment["CXX"] == str(compiler.cxx)
    assert f"-ccbin {compiler.cxx}" in environment["NVCC_PREPEND_FLAGS"]
    # The caller's mapping is untouched: nothing global changes.
    assert base["CUDA_HOME"] == "/usr/local/cuda"


# --- pretrained checkpoint integrity -----------------------------------------


def _spec(payload: bytes) -> PretrainedCheckpointSpec:
    return PretrainedCheckpointSpec(
        model="vmamba_t",
        filename="vmamba_tiny_e292.pth",
        source_url="https://example.invalid/checkpoint.pth",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def test_configured_vmamba_checkpoint_is_fully_specified() -> None:
    spec = family_pretrained_spec(ROOT, "vmamba")
    assert spec is not None
    assert spec.filename == "vmamba_tiny_e292.pth"
    assert spec.upstream_filename == "vssmtiny_dp01_ckpt_epoch_292.pth"
    assert spec.source_url.startswith("https://github.com/MzeroMiko/VMamba/releases/")
    assert len(spec.sha256) == 64
    assert spec.size_bytes == 91649482


def test_incomplete_or_malformed_specifications_are_rejected() -> None:
    with pytest.raises(CheckpointVerificationError, match="incomplete"):
        load_pretrained_spec({"model": "vmamba_t", "filename": "x.pth"})
    with pytest.raises(CheckpointVerificationError, match="not a hex digest"):
        load_pretrained_spec(
            {
                "model": "vmamba_t",
                "filename": "x.pth",
                "source_url": "https://example.invalid/x.pth",
                "sha256": "not-a-digest",
                "size_bytes": 4,
            }
        )


def test_valid_checkpoint_is_reused_without_downloading(tmp_path: Path) -> None:
    payload = b"vmamba-weights"
    target = tmp_path / "vmamba_tiny_e292.pth"
    target.write_bytes(payload)

    def refuse(_url: str, _destination: Path) -> None:
        raise AssertionError("a valid checkpoint must not be re-downloaded")

    record = ensure_pretrained_checkpoint(_spec(payload), target, downloader=refuse)

    assert record["action"] == "reused"
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()


def test_truncated_checkpoint_fails_on_size_before_hashing(tmp_path: Path) -> None:
    payload = b"vmamba-weights"
    target = tmp_path / "vmamba_tiny_e292.pth"
    target.write_bytes(payload[:5])

    with pytest.raises(CheckpointVerificationError, match="truncated"):
        verify_checkpoint(target, _spec(payload))


def test_corrupt_checkpoint_of_the_right_size_fails_on_its_digest(tmp_path: Path) -> None:
    payload = b"vmamba-weights"
    target = tmp_path / "vmamba_tiny_e292.pth"
    target.write_bytes(b"x" * len(payload))

    with pytest.raises(CheckpointVerificationError, match="SHA-256"):
        verify_checkpoint(target, _spec(payload))


def test_download_is_atomic_and_a_corrupt_download_is_never_promoted(
    tmp_path: Path,
) -> None:
    payload = b"vmamba-weights"
    target = tmp_path / "vmamba_tiny_e292.pth"

    def corrupt(_url: str, destination: Path) -> None:
        destination.write_bytes(b"nginx 404 page")

    with pytest.raises(CheckpointVerificationError):
        ensure_pretrained_checkpoint(_spec(payload), target, downloader=corrupt)

    assert not target.exists()
    assert list(tmp_path.glob("*.download-*")) == []


def test_valid_download_is_verified_then_moved_into_place(tmp_path: Path) -> None:
    payload = b"vmamba-weights"
    target = tmp_path / "nested" / "vmamba_tiny_e292.pth"

    record = ensure_pretrained_checkpoint(
        _spec(payload),
        target,
        downloader=lambda _url, destination: destination.write_bytes(payload),
    )

    assert record["action"] == "downloaded"
    assert target.read_bytes() == payload
    assert record["source_url"] == "https://example.invalid/checkpoint.pth"


def test_missing_checkpoint_with_downloads_disabled_names_the_fetch_command(
    tmp_path: Path,
) -> None:
    payload = b"vmamba-weights"
    with pytest.raises(CheckpointVerificationError, match="fetch_pretrained_checkpoints"):
        ensure_pretrained_checkpoint(
            _spec(payload),
            tmp_path / "absent.pth",
            environ={ALLOW_DOWNLOAD_ENV: "0"},
        )
