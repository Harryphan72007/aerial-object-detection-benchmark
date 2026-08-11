#!/usr/bin/env python
"""Validate canonical notebooks without executing them."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

from src.notebook_bootstrap import render_bootstrap_cell

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATHS = (
    re.compile(r"[A-Za-z]:\\Users\\"),
    re.compile(r"/Users/[^/\s]+/"),
    re.compile(r"/home/[^/\s]+/"),
)
SECRET_ASSIGNMENT = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|password|secret)\s*=\s*['\"][^'\"]+"
)
TRANSIENT_NOTEBOOK_METADATA = {"colab", "varInspector", "widgets"}
TRANSIENT_CELL_METADATA = {
    "collapsed",
    "colab",
    "execution",
    "jupyter",
    "outputId",
    "scrolled",
}
# One notebook per model, then one report. Every canonical notebook carries the
# same bootstrap cell and the same dependency policy; MODEL_NOTEBOOKS are the
# four that drive a GPU run through the pipeline.
MODEL_NOTEBOOKS = {
    "10_resnet50.ipynb",
    "11_swin_t.ipynb",
    "12_vmamba_t.ipynb",
    "13_rtdetrv2.ipynb",
}
REPORT_NOTEBOOK = "30_report.ipynb"
CANONICAL_NOTEBOOKS = MODEL_NOTEBOOKS | {REPORT_NOTEBOOK}
MODEL_ENVIRONMENT_NOTEBOOKS = CANONICAL_NOTEBOOKS
# The pipeline provisions the model environment itself, so a notebook proves it
# reaches provisioning by calling the pipeline, not by importing the installer.
REQUIRED_ENTRY_POINTS = {
    **{name: "run_model_pipeline" for name in MODEL_NOTEBOOKS},
    REPORT_NOTEBOOK: "build_benchmark_report",
}
NOTEBOOK_REQUIREMENTS = "requirements-hpo-colab.txt"
INLINE_ENVIRONMENT_SETUP = re.compile(
    r"(?im)(?:^\s*[!%]\s*(?:pip|uv|conda)|"
    r"\b(?:pip|uv)\s+install\b|\bpython\s+-m\s+venv\b)"
)


def validate_notebook(path: Path) -> list[str]:
    transformer = TransformerManager()
    errors: list[str] = []
    try:
        notebook = nbformat.read(path, as_version=4)
    except Exception as error:
        return [f"{path}: unreadable notebook: {error}"]
    try:
        nbformat.validate(notebook)
    except Exception as error:
        errors.append(f"{path}: nbformat: {error}")
    notebook_metadata = TRANSIENT_NOTEBOOK_METADATA.intersection(notebook.metadata)
    if notebook_metadata:
        errors.append(
            f"{path}: transient notebook metadata: "
            + ", ".join(sorted(notebook_metadata))
        )
    package_import_found = False
    notebook_source: list[str] = []
    for index, cell in enumerate(notebook.cells):
        cell_metadata = TRANSIENT_CELL_METADATA.intersection(cell.metadata)
        if cell_metadata:
            errors.append(
                f"{path}: cell {index} transient metadata: "
                + ", ".join(sorted(cell_metadata))
            )
        if cell.cell_type != "code":
            continue
        notebook_source.append(cell.source)
        if cell.outputs:
            errors.append(f"{path}: cell {index} has outputs")
        if cell.execution_count is not None:
            errors.append(f"{path}: cell {index} has execution_count")
        try:
            tree = ast.parse(transformer.transform_cell(cell.source))
        except SyntaxError as error:
            errors.append(f"{path}: cell {index} syntax: {error}")
            tree = None
        if tree is not None and path.name in CANONICAL_NOTEBOOKS:
            package_import_found = package_import_found or any(
                (
                    isinstance(node, ast.ImportFrom)
                    and node.module is not None
                    and (node.module == "src" or node.module.startswith("src."))
                )
                or (
                    isinstance(node, ast.Import)
                    and any(alias.name == "src" for alias in node.names)
                )
                for node in ast.walk(tree)
            )
            local_definitions = [
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]
            if local_definitions:
                errors.append(
                    f"{path}: cell {index} contains notebook-local definitions: "
                    + ", ".join(local_definitions)
                )
        if SECRET_ASSIGNMENT.search(cell.source):
            errors.append(f"{path}: cell {index} contains a secret-like assignment")
        for pattern in PRIVATE_PATHS:
            if pattern.search(cell.source):
                errors.append(f"{path}: cell {index} contains a private path")
    if path.name in CANONICAL_NOTEBOOKS and not package_import_found:
        errors.append(f"{path}: canonical notebook does not delegate to the src package")
    if path.name in MODEL_ENVIRONMENT_NOTEBOOKS:
        combined = "\n".join(notebook_source)
        if INLINE_ENVIRONMENT_SETUP.search(combined):
            errors.append(
                f"{path}: model notebook contains inline environment setup; use the shared API"
            )
        entry_point = REQUIRED_ENTRY_POINTS[path.name]
        if entry_point not in combined:
            errors.append(f"{path}: notebook does not call {entry_point}")
    if path.name in CANONICAL_NOTEBOOKS:
        errors.extend(_bootstrap_cell_errors(path, notebook_source))
    return errors


def _bootstrap_cell_errors(path: Path, sources: list[str]) -> list[str]:
    """The shared bootstrap cell must be identical in every canonical notebook.

    Copying it by hand is what let one notebook end up with a different
    dependency policy than its siblings while every other check still passed.
    """
    expected = render_bootstrap_cell(NOTEBOOK_REQUIREMENTS).rstrip("\n")
    if any(source.rstrip("\n") == expected for source in sources):
        return []
    return [
        f"{path}: bootstrap cell does not match "
        "src.notebook_bootstrap.render_bootstrap_cell; regenerate it"
    ]


def main() -> None:
    errors: list[str] = []
    for path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
        errors.extend(validate_notebook(path))
    if errors:
        raise SystemExit("\n".join(errors))
    print("Notebook validation passed.")


if __name__ == "__main__":
    main()
