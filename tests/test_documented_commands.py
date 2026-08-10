"""Every command the documentation tells an operator to run must exist.

The runbook and RUN.md are what someone follows on a GPU host after a run has
already been blocked. A documented flag that argparse rejects turns the recovery
path into a second dead end, and nothing else in the suite executes those
snippets - `validate_doc_links` checks links, not commands.

This walks the documented ``python -m scripts.*`` invocations, descends into
subcommands, and asserts every flag is real.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INVOCATION = re.compile(r"python3?\s+-m\s+(scripts\.[a-z_0-9.]+)((?:[^\n`]|\\\n)*)")
FLAG = re.compile(r"--[a-z0-9-]+")
# Placeholders an operator substitutes, e.g. "<artifact_root>".
PLACEHOLDER = re.compile(r"^<.*>$")


def _documented_invocations() -> list[tuple[str, tuple[str, ...], Path]]:
    sources = [*sorted(ROOT.glob("*.md")), *sorted((ROOT / "docs").rglob("*.md"))]
    found: list[tuple[str, tuple[str, ...], Path]] = []
    for path in sources:
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in INVOCATION.finditer(text):
            tail = match.group(2).replace("\\\n", " ")
            found.append((match.group(1), tuple(FLAG.findall(tail)), path))
    return found


def _help_text(module: str, subcommand: str | None) -> str:
    command = [sys.executable, "-m", module]
    if subcommand:
        command.append(subcommand)
    completed = subprocess.run(
        command + ["--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0, (
        f"`python -m {module}"
        + (f" {subcommand}" if subcommand else "")
        + f" --help` failed:\n{completed.stderr}"
    )
    return completed.stdout


def _subcommand(module: str, tail: str) -> str | None:
    """Return the first bare word after the module, if it names a subcommand."""
    words = [word for word in tail.split() if word and not word.startswith("-")]
    if not words or PLACEHOLDER.match(words[0]):
        return None
    listed = _help_text(module, None)
    return words[0] if re.search(rf"\b{re.escape(words[0])}\b", listed) else None


def test_documentation_lists_at_least_one_command() -> None:
    assert _documented_invocations(), "no documented scripts.* commands were found"


@pytest.mark.parametrize(
    ("module", "flags", "source"),
    [
        pytest.param(module, flags, source, id=f"{source.name}-{module}")
        for module, flags, source in _documented_invocations()
    ],
)
def test_documented_flags_exist(
    module: str, flags: tuple[str, ...], source: Path
) -> None:
    text = source.read_text(encoding="utf-8", errors="replace")
    match = next(
        item
        for item in INVOCATION.finditer(text)
        if item.group(1) == module
        and tuple(FLAG.findall(item.group(2).replace("\\\n", " "))) == flags
    )
    tail = match.group(2).replace("\\\n", " ")
    known = set(FLAG.findall(_help_text(module, _subcommand(module, tail))))
    unknown = sorted(set(flags) - known)
    assert not unknown, (
        f"{source.relative_to(ROOT)} documents `python -m {module}` with "
        f"{unknown}, which its CLI does not define"
    )
