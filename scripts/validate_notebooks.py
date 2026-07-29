#!/usr/bin/env python
"""Validate canonical notebooks without executing them."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATHS = (
    re.compile(r"[A-Za-z]:\\Users\\"),
    re.compile(r"/Users/[^/\s]+/"),
    re.compile(r"/home/[^/\s]+/"),
)
SECRET_ASSIGNMENT = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|password|secret)\s*=\s*['\"][^'\"]+"
)


def main() -> None:
    transformer = TransformerManager()
    errors: list[str] = []
    for path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        try:
            nbformat.validate(notebook)
        except Exception as error:
            errors.append(f"{path}: nbformat: {error}")
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            if cell.outputs:
                errors.append(f"{path}: cell {index} has outputs")
            if cell.execution_count is not None:
                errors.append(f"{path}: cell {index} has execution_count")
            try:
                ast.parse(transformer.transform_cell(cell.source))
            except SyntaxError as error:
                errors.append(f"{path}: cell {index} syntax: {error}")
            if SECRET_ASSIGNMENT.search(cell.source):
                errors.append(f"{path}: cell {index} contains a secret-like assignment")
            for pattern in PRIVATE_PATHS:
                if pattern.search(cell.source):
                    errors.append(f"{path}: cell {index} contains a private path")
    if errors:
        raise SystemExit("\n".join(errors))
    print("Notebook validation passed.")


if __name__ == "__main__":
    main()
