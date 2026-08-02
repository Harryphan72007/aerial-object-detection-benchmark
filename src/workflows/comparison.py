"""Strict comparison of completed controlled final runs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.paths import ProjectPaths
from src.optional_outputs import run_optional_output
from src.subprocess_utils import configure_headless_matplotlib
from src.training.checkpointing import RunRegistry
from src.utils.serialization import read_json, read_yaml, write_json, write_text_atomic
from src.workflows.contract import PRIMARY_MODELS, validate_final_config
from src.config.benchmark_tracks import require_comparison_track


def compare_completed_models(
    drive_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    benchmark_track: str = "controlled",
) -> dict[str, Any]:
    paths = ProjectPaths.from_value(drive_root)
    output = (
        Path(output_dir)
        if output_dir
        else paths.reports / "comparison" / benchmark_track
    )
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    completed = RunRegistry(paths).list_available_runs(
        dataset_track="2class", status="completed"
    )
    by_model = {}
    for run in completed:
        if run.get("model_id") in PRIMARY_MODELS:
            by_model.setdefault(run["model_id"], []).append(run)
    for model_id in PRIMARY_MODELS:
        candidates = by_model.get(model_id, [])
        if not candidates:
            rows.append({"Model": model_id, "status": "MISSING"})
            continue
        accepted = None
        for run in sorted(candidates, key=lambda item: str(item.get("created_at", "")), reverse=True):
            run_dir = Path(
                str(
                    run.get("run_dir")
                    or paths.final_checkpoints / model_id / run["run_id"]
                )
            )
            try:
                require_comparison_track(run, benchmark_track)
                config = read_yaml(run_dir / "training_config.yaml")
                require_comparison_track(config, benchmark_track)
                validate_final_config(config)
                metric_path = (
                    paths.evaluation / f"{run['run_id']}__res640__metrics.json"
                )
                profile_path = paths.evaluation / f"{run['run_id']}__profile.json"
                if not metric_path.is_file() or not profile_path.is_file():
                    raise ValueError("evaluation or profiling output is missing")
                metric = read_json(metric_path)
                if (
                    metric.get("dataset_track") != "2class"
                    or int(metric.get("evaluation_resolution", -1)) != 640
                    or int(metric.get("seed", -1)) != 42
                ):
                    raise ValueError("evaluation protocol is incompatible")
                profile = read_json(profile_path)
                batch_one = next(
                    (
                        value
                        for value in profile.get("profiles", [])
                        if int(value.get("batch_size", -1)) == 1
                        and value.get("status") == "completed"
                    ),
                    {},
                )
                per_class = metric.get("per_class", {})
                selected_lr = float(config["overrides"]["learning_rate"])
                accepted = {
                    "Model": model_id,
                    "Architecture family": run.get("architecture_family"),
                    "Selected LR": selected_lr,
                    "mAP50-95": metric.get("mAP"),
                    "APtiny": metric.get("APtiny"),
                    "person AP": per_class.get("person", {}).get("AP"),
                    "vehicle AP": per_class.get("vehicle", {}).get("AP"),
                    "parameters": metric.get("total_parameters"),
                    "training time": metric.get("total_training_seconds"),
                    "peak memory": batch_one.get("peak_inference_vram_bytes")
                    or metric.get("peak_vram_mb"),
                    "latency": batch_one.get("median_latency_ms")
                    or metric.get("median_latency_ms"),
                    "FPS": batch_one.get("fps") or metric.get("fps"),
                    "status": "COMPLETE",
                    "run_id": run["run_id"],
                }
                break
            except (KeyError, OSError, TypeError, ValueError) as error:
                rejected.append(
                    {"model_id": model_id, "run_id": run["run_id"], "reason": str(error)}
                )
        rows.append(accepted or {"Model": model_id, "status": "INCOMPATIBLE"})
    complete_rows = [row for row in rows if row["status"] == "COMPLETE"]
    if len(complete_rows) < 2:
        raise RuntimeError(
            "At least two compatible completed models are required; "
            f"found {len(complete_rows)}."
        )
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "comparison.csv", index=False)
    rendered_table, _ = run_optional_output(
        "render_comparison_markdown_table",
        output,
        lambda: frame.to_markdown(index=False),
    )
    markdown = [
        f"# {benchmark_track.title()} 2-class model comparison",
        "",
        rendered_table if rendered_table is not None else frame.to_string(index=False),
        "",
        "Only seed-42, 640-pixel, 25-epoch, effective-batch-8 final runs are included.",
        "",
    ]
    write_text_atomic(output / "comparison.md", "\n".join(markdown))
    write_json(output / "comparison.json", {"models": rows, "rejected": rejected})

    def save_plots() -> None:
        configure_headless_matplotlib()
        import matplotlib.pyplot as plt

        plot_rows = pd.DataFrame(complete_rows)
        for column, label, filename in (
            ("latency", "Median latency (ms)", "accuracy_latency.png"),
            ("peak memory", "Peak memory (bytes or recorded MB)", "accuracy_memory.png"),
        ):
            values = plot_rows.dropna(subset=[column, "mAP50-95"])
            if values.empty:
                continue
            figure, axis = plt.subplots(figsize=(7, 5))
            axis.scatter(values[column], values["mAP50-95"])
            for _, row in values.iterrows():
                axis.annotate(row["Model"], (row[column], row["mAP50-95"]))
            axis.set_xlabel(label)
            axis.set_ylabel("mAP50-95")
            axis.set_title(f"Accuracy–{column} trade-off")
            figure.tight_layout()
            figure.savefig(output / filename, dpi=160)
            plt.close(figure)

    run_optional_output("save_comparison_plots", output, save_plots)
    return {"models": rows, "rejected": rejected, "output": str(output)}
