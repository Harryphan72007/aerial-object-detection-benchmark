#!/usr/bin/env python
"""Strip outputs, execution counts, and transient Colab metadata in place."""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


TRANSIENT_CELL_METADATA = {
    "collapsed",
    "execution",
    "jupyter",
    "outputId",
    "scrolled",
}
TRANSIENT_NOTEBOOK_METADATA = {"widgets", "varInspector"}


def clean_notebook(path: str | Path, *, check: bool = False) -> bool:
    notebook_path = Path(path)
    notebook = nbformat.read(notebook_path, as_version=4)
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
    changed = cleaned != original
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
