#!/usr/bin/env python
"""Strip outputs, execution counts, and transient Colab metadata in place."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nbformat


TRANSIENT_CELL_METADATA = {
    "collapsed",
    "colab",
    "execution",
    "jupyter",
    "outputId",
    "scrolled",
}
TRANSIENT_NOTEBOOK_METADATA = {"colab", "widgets", "varInspector"}


def clean_notebook(path: str | Path, *, check: bool = False) -> bool:
    notebook_path = Path(path)
    raw = json.loads(notebook_path.read_text(encoding="utf-8"))
    missing_id = False
    for index, cell in enumerate(raw.get("cells", [])):
        if not cell.get("id"):
            identity = hashlib.sha256(
                f"{index}:{cell.get('cell_type')}:{cell.get('source')}".encode(
                    "utf-8"
                )
            ).hexdigest()[:12]
            cell["id"] = f"cell-{identity}"
            missing_id = True
    notebook = nbformat.from_dict(raw)
    original = nbformat.writes(notebook)
    for key in TRANSIENT_NOTEBOOK_METADATA:
        notebook.metadata.pop(key, None)
    for cell in notebook.cells:
        for key in TRANSIENT_CELL_METADATA:
            cell.metadata.pop(key, None)
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    cleaned = nbformat.writes(notebook)
    changed = missing_id or cleaned != original
    if changed and not check:
        nbformat.write(notebook, notebook_path)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["notebooks"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    notebooks = []
    for value in args.paths:
        path = Path(value)
        notebooks.extend(path.rglob("*.ipynb") if path.is_dir() else [path])
    changed = [str(path) for path in notebooks if clean_notebook(path, check=args.check)]
    if changed:
        print("\n".join(changed))
        if args.check:
            raise SystemExit("Notebooks require cleaning")


if __name__ == "__main__":
    main()
