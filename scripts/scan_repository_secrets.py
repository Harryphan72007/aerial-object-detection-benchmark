#!/usr/bin/env python
"""Fail on obvious committed secrets or private workstation paths."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
)


def main() -> None:
    files = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    errors = []
    for relative in files:
        path = ROOT / relative
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for pattern in PATTERNS:
            if pattern.search(text):
                errors.append(f"{relative}: {pattern.pattern}")
    if errors:
        raise SystemExit("Potential secrets/private paths:\n" + "\n".join(errors))
    print("Repository secret scan passed.")


if __name__ == "__main__":
    main()
