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
TRANSIENT_NOTEBOOK_METADATA = {"colab", "varInspector", "widgets"}
TRANSIENT_CELL_METADATA = {
    "collapsed",
    "colab",
    "execution",
    "jupyter",
    "outputId",
    "scrolled",
}
CANONICAL_NOTEBOOKS = {
    "00_prepare_visdrone.ipynb",
    "07_performance_tiling.ipynb",
    "10_hpo_resnet50.ipynb",
    "11_hpo_swin_t.ipynb",
    "12_hpo_vmamba_t.ipynb",
    "13_hpo_rtdetrv2.ipynb",
    "20_finetune_resnet50.ipynb",
    "21_finetune_swin_t.ipynb",
    "22_finetune_vmamba_t.ipynb",
    "23_finetune_rtdetrv2.ipynb",
    "30_evaluate_all_models.ipynb",
    "31_publish_results.ipynb",
}
MODEL_ENVIRONMENT_NOTEBOOKS = {
    "10_hpo_resnet50.ipynb",
    "11_hpo_swin_t.ipynb",
    "12_hpo_vmamba_t.ipynb",
    "13_hpo_rtdetrv2.ipynb",
    "20_finetune_resnet50.ipynb",
    "21_finetune_swin_t.ipynb",
    "22_finetune_vmamba_t.ipynb",
    "23_finetune_rtdetrv2.ipynb",
    "30_evaluate_all_models.ipynb",
    "31_publish_results.ipynb",
}
DIRECT_ENVIRONMENT_NOTEBOOKS = {
    "10_hpo_resnet50.ipynb",
    "11_hpo_swin_t.ipynb",
    "12_hpo_vmamba_t.ipynb",
    "13_hpo_rtdetrv2.ipynb",
    "20_finetune_resnet50.ipynb",
    "21_finetune_swin_t.ipynb",
    "22_finetune_vmamba_t.ipynb",
    "23_finetune_rtdetrv2.ipynb",
}
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
        if (
            path.name in DIRECT_ENVIRONMENT_NOTEBOOKS
            and "ensure_model_environment" not in combined
        ):
            errors.append(
                f"{path}: model notebook does not call ensure_model_environment"
            )
    return errors


def main() -> None:
    errors: list[str] = []
    for path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
        errors.extend(validate_notebook(path))
    if errors:
        raise SystemExit("\n".join(errors))
    print("Notebook validation passed.")


if __name__ == "__main__":
    main()
