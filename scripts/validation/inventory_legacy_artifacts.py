#!/usr/bin/env python
"""Inventory artifact references in the repository's current notebooks.

This is deliberately a static, read-only inventory. It uses only the Python
standard library so it can run before any model-family environment is installed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ARTIFACT_SUFFIXES = (
    ".ckpt",
    ".csv",
    ".db",
    ".html",
    ".json",
    ".jsonl",
    ".onnx",
    ".png",
    ".pt",
    ".pth",
    ".txt",
    ".yaml",
    ".yml",
)
ARTIFACT_TERMS = (
    "artifact",
    "bundle",
    "checkpoint",
    "dataset",
    "drive_root",
    "evaluation",
    "manifest",
    "metric",
    "prediction",
    "profile",
    "registry",
    "report",
    "result",
)
ARTIFACT_PATH_PARTS = {
    "cache",
    "checkpoints",
    "datasets",
    "evaluation",
    "experiment_registry",
    "exports",
    "logs",
    "manifests",
    "predictions",
    "pretrained",
    "profiling",
    "reports",
    "result_bundles",
}
OPERATION_PATTERNS = {
    "read": re.compile(
        r"\b(read_(?:json|yaml|csv)|read_text|json\.load|json\.loads|"
        r"pd\.read_csv|torch\.load|load_checkpoint|load_model)\b",
        re.IGNORECASE,
    ),
    "write": re.compile(
        r"\b(write_(?:json|yaml|csv)|write_text|json\.dump|json\.dumps|"
        r"to_csv|torch\.save|export_|publish_)\w*\b",
        re.IGNORECASE,
    ),
    "discover": re.compile(
        r"\b(glob|rglob|list_available_runs|get_best_run|"
        r"load_checkpoint_from_registry)\b",
        re.IGNORECASE,
    ),
}
QUOTED_LITERAL = re.compile(r"(?P<quote>['\"])(?P<value>.*?)(?P=quote)")


def _source_text(source: Any) -> str:
    if isinstance(source, list):
        return "".join(str(line) for line in source)
    return str(source or "")


def _imports(source: str) -> list[str]:
    sanitized = "\n".join(
        line
        for line in source.splitlines()
        if not line.lstrip().startswith(("!", "%"))
    )
    try:
        tree = ast.parse(sanitized)
    except SyntaxError:
        return []
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return sorted(names)


def _artifact_literals(line: str, *, broad_terms: bool = True) -> list[str]:
    values: set[str] = set()
    for match in QUOTED_LITERAL.finditer(line):
        value = match.group("value")
        lowered = value.lower()
        is_artifact_path = lowered in ARTIFACT_PATH_PARTS or any(
            part in ARTIFACT_PATH_PARTS
            for part in re.split(r"[/\\]", lowered)
        )
        if (
            lowered.endswith(ARTIFACT_SUFFIXES)
            or is_artifact_path
            or (broad_terms and any(term in lowered for term in ARTIFACT_TERMS))
        ):
            values.add(value)
    return sorted(values)


def _artifact_references(
    source: str, cell_index: int, *, include_term_only: bool = True
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        line = raw_line.strip()
        lowered = line.lower()
        literals = _artifact_literals(line, broad_terms=include_term_only)
        operations = sorted(
            name for name, pattern in OPERATION_PATTERNS.items() if pattern.search(line)
        )
        has_term = any(term in lowered for term in ARTIFACT_TERMS)
        if not literals and not operations and not (include_term_only and has_term):
            continue
        references.append(
            {
                "cell": cell_index,
                "line": line_number,
                "operations": operations,
                "literals": literals,
                "source": line[:300],
            }
        )
    return references


def inventory_notebook(path: Path, repository_root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    notebook = json.loads(raw.decode("utf-8"))
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"notebook has no cells array: {path}")
    imports: set[str] = set()
    references: list[dict[str, Any]] = []
    code_cell_count = 0
    for cell_index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        code_cell_count += 1
        source = _source_text(cell.get("source"))
        imports.update(_imports(source))
        references.extend(_artifact_references(source, cell_index))
    return {
        "path": path.relative_to(repository_root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "nbformat": notebook.get("nbformat"),
        "code_cell_count": code_cell_count,
        "imports": sorted(imports),
        "artifact_references": references,
    }


def inventory_python_source(path: Path, repository_root: Path) -> dict[str, Any] | None:
    raw = path.read_bytes()
    source = raw.decode("utf-8")
    references = _artifact_references(source, 0, include_term_only=False)
    if not references:
        return None
    return {
        "path": path.relative_to(repository_root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "imports": _imports(source),
        "artifact_references": references,
    }


def build_inventory(
    repository_root: Path,
    notebook_root: Path,
    excluded_sources: set[str] | None = None,
) -> dict[str, Any]:
    notebooks = sorted(notebook_root.rglob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError(f"no notebooks found under {notebook_root}")
    records = [inventory_notebook(path, repository_root) for path in notebooks]
    excluded = excluded_sources or set()
    source_paths = sorted(
        path
        for source_root in (repository_root / "src", repository_root / "scripts")
        for path in source_root.rglob("*.py")
        if path.relative_to(repository_root).as_posix() not in excluded
    )
    source_records = [
        record
        for path in source_paths
        if (record := inventory_python_source(path, repository_root)) is not None
    ]
    return {
        "schema_version": 1,
        "repository_contract": "legacy-artifacts-v1",
        "notebook_root": notebook_root.relative_to(repository_root).as_posix(),
        "notebook_count": len(records),
        "notebooks": records,
        "python_source_count": len(source_records),
        "python_sources": source_records,
    }


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--notebook-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help="Repository-relative Python source to omit from a historical snapshot.",
    )
    parser.add_argument(
        "--check",
        type=Path,
        help="Fail when the generated inventory differs from this snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = args.repo_root.expanduser().resolve()
    notebook_root = (
        args.notebook_root.expanduser().resolve()
        if args.notebook_root
        else repository_root / "notebooks"
    )
    inventory = build_inventory(
        repository_root, notebook_root, set(args.exclude_source)
    )
    rendered = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    if args.check:
        expected = args.check.expanduser().resolve().read_text(encoding="utf-8")
        if expected != rendered:
            raise SystemExit(
                "legacy notebook artifact inventory is stale; regenerate it with "
                "--output and review the contract change"
            )
        if not args.output:
            return
    if args.output:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
