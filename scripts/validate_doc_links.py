#!/usr/bin/env python
"""Check relative Markdown links in tracked documentation."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def main() -> None:
    errors = []
    documents = [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            target = target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (document.parent / unquote(target)).resolve()
            if not resolved.exists():
                errors.append(f"{document.relative_to(ROOT)} -> {target}")
    if errors:
        raise SystemExit("Broken documentation links:\n" + "\n".join(errors))
    print("Documentation links passed.")


if __name__ == "__main__":
    main()
