"""Turn completed final runs into the benchmark's reported output.

This is the last stage of the workflow: evaluate whatever is still missing,
aggregate across seeds, build the comparison tables and figures, and — only on
an explicit request — publish the lightweight bundles.

Publishing is deliberately the one step that does not happen by default.
Everything above it writes to the artifact root; publishing pushes to a public
repository, so it requires ``publish=True`` and ``dry_run=False`` together.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.evaluation.report_generator import generate_report
from src.paths import ProjectPaths
from src.utils.serialization import read_json
from src.workflows.comparison import compare_completed_models
from src.workflows.contract import PRIMARY_MODELS
from src.workflows.evaluation_runner import evaluate_pending_runs, pending_evaluations
from src.workflows.hpo_comparison import aggregate_hpo_results
from src.workflows.publishing import publish_results


def _metric_rows(paths: ProjectPaths) -> list[dict[str, Any]]:
    return [
        read_json(path) for path in sorted(paths.evaluation.glob("*__metrics.json"))
    ]


def build_benchmark_report(
    repo_root: str | Path,
    drive_root: str | Path,
    dataset_track: str = "2class",
    *,
    evaluate_missing: bool = True,
    publish: bool = False,
    dry_run: bool = True,
    model_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Evaluate, aggregate, and report every completed run; optionally publish."""
    # Checked before any work, so a mistyped publish request costs nothing.
    if publish and dry_run:
        raise ValueError("Publishing requires publish=True and dry_run=False")
    repo = Path(repo_root).resolve()
    paths = ProjectPaths.from_value(drive_root).create()
    models = tuple(model_ids) if model_ids is not None else tuple(PRIMARY_MODELS)

    evaluation: dict[str, Any]
    if evaluate_missing:
        evaluation = evaluate_pending_runs(
            repo, paths.root, dataset_track, model_ids=models
        )
    else:
        evaluation = {
            "dataset_track": dataset_track,
            "evaluated": [],
            "still_missing": [
                {"model_id": str(run["model_id"]), "run_id": str(run["run_id"])}
                for run in pending_evaluations(
                    paths.root, dataset_track, model_ids=models
                )
            ],
        }

    rows = _metric_rows(paths)
    aggregate = aggregate_hpo_results(paths.root, dataset_track)
    # The cross-model comparison needs at least two compatible completed models
    # and refuses otherwise. That is the normal state of a partly-run benchmark,
    # so it is reported as unavailable rather than allowed to abort the report
    # and discard the per-model tables that were produced above.
    try:
        comparison: dict[str, Any] = compare_completed_models(paths.root)
    except (RuntimeError, ValueError) as error:
        comparison = {"status": "UNAVAILABLE", "reason": str(error)}
    # generate_report writes CSV/JSON tables unconditionally and figures only
    # when a plotting stack is available, so an environment without matplotlib
    # still produces the numbers.
    report = generate_report(rows, paths.reports) if rows else {}

    published: list[dict[str, Any]] = []
    for model_id in models if publish else ():
        published.append(
            publish_results(
                repo,
                paths.root,
                model_id,
                dataset_track=dataset_track,
                publish_results=True,
                dry_run=False,
            )
        )

    return {
        "dataset_track": dataset_track,
        "models": list(models),
        "evaluation": evaluation,
        "evaluated_metric_files": len(rows),
        "aggregate": aggregate,
        "comparison": comparison,
        "report": report,
        "published": published,
        "publish_status": "PUBLISHED" if published else "NOT_PUBLISHED",
    }
