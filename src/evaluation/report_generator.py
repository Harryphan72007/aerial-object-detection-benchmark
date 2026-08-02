"""Generate Markdown, HTML, CSV, JSON, figures, and PDF summaries."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.evaluation.efficiency_metrics import pareto_frontier
from src.optional_outputs import run_optional_output
from src.subprocess_utils import configure_headless_matplotlib
from src.utils.serialization import write_json, write_text_atomic


def _finite_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            output.append(row)
    return output


def _best(
    rows: list[dict[str, Any]], key: str, maximize: bool = True
) -> str:
    candidates = _finite_rows(rows, key)
    if not candidates:
        return "not measured"
    selected = (max if maximize else min)(
        candidates, key=lambda row: float(row[key])
    )
    return str(selected.get("model_id", "unknown"))


def recommendation_matrix(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Generate conditional winners only where the relevant metric exists."""
    completed = _finite_rows(rows, "mAP")
    if not completed:
        return []
    latency_rows = [
        row
        for row in completed
        if isinstance(row.get("mean_latency_ms"), (int, float))
    ]
    memory_rows = [
        row
        for row in completed
        if isinstance(row.get("peak_inference_vram_bytes"), (int, float))
    ]
    latency_pareto = (
        pareto_frontier(latency_rows, "mAP", "mean_latency_ms")
        if latency_rows
        else []
    )
    memory_pareto = (
        pareto_frontier(memory_rows, "mAP", "peak_inference_vram_bytes")
        if memory_rows
        else []
    )
    return [
        {"Scenario": "Maximum accuracy", "Recommended model": _best(completed, "mAP")},
        {
            "Scenario": "Dense tiny-object scenes",
            "Recommended model": _best(completed, "APtiny"),
        },
        {
            "Scenario": "Real-time batch-1 inference",
            "Recommended model": _best(completed, "mean_latency_ms", False),
        },
        {
            "Scenario": "Highest measured throughput",
            "Recommended model": _best(
                completed, "throughput_images_per_second"
            ),
        },
        {
            "Scenario": "Limited inference memory",
            "Recommended model": _best(
                completed, "peak_inference_vram_bytes", False
            ),
        },
        {
            "Scenario": "Shortest training time",
            "Recommended model": _best(
                completed, "total_training_seconds", False
            ),
        },
        {
            "Scenario": "Fewest trainable parameters",
            "Recommended model": _best(
                completed, "trainable_parameters", False
            ),
        },
        {
            "Scenario": "Accuracy–latency Pareto",
            "Recommended model": ", ".join(
                str(row["model_id"]) for row in latency_pareto
            )
            if latency_pareto
            else "not measured",
        },
        {
            "Scenario": "Accuracy–memory Pareto",
            "Recommended model": ", ".join(
                str(row["model_id"]) for row in memory_pareto
            )
            if memory_pareto
            else "not measured",
        },
        {
            "Scenario": "Simplest deployment",
            "Recommended model": _best(completed, "export_success_rate"),
        },
        {
            "Scenario": "Best practical research baseline",
            "Recommended model": _best(completed, "mAP"),
        },
    ]


def _label(row: dict[str, Any]) -> str:
    resolution = row.get("evaluation_resolution", row.get("input_resolution"))
    return f"{row.get('model_id', 'unknown')}@{resolution}" if resolution else str(row.get("model_id", "unknown"))


def _save_figure(figure: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")


def _placeholder(destination: Path, title: str, requirement: str) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.text(
        0.5,
        0.52,
        "Not generated from measurements",
        ha="center",
        va="center",
        fontsize=15,
    )
    axis.text(
        0.5,
        0.40,
        requirement,
        ha="center",
        va="center",
        wrap=True,
        fontsize=10,
    )
    axis.set_title(title)
    axis.set_axis_off()
    _save_figure(figure, destination)
    plt.close(figure)


def _bar_figure(
    rows: list[dict[str, Any]], key: str, destination: Path, title: str
) -> None:
    import matplotlib.pyplot as plt

    candidates = _finite_rows(rows, key)
    if not candidates:
        _placeholder(destination, title, f"Requires numeric `{key}` values.")
        return
    figure, axis = plt.subplots(figsize=(10, 5.5))
    labels = [_label(row) for row in candidates]
    axis.bar(labels, [float(row[key]) for row in candidates])
    axis.set_title(title)
    axis.set_ylabel(key)
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, destination)
    plt.close(figure)


def _scatter_figure(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    destination: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    candidates = [
        row
        for row in rows
        if row in _finite_rows(rows, x_key) and row in _finite_rows(rows, y_key)
    ]
    if not candidates:
        _placeholder(
            destination,
            title,
            f"Requires numeric `{x_key}` and `{y_key}` values.",
        )
        return
    figure, axis = plt.subplots(figsize=(8, 5.5))
    x_values = [float(row[x_key]) for row in candidates]
    y_values = [float(row[y_key]) for row in candidates]
    axis.scatter(x_values, y_values)
    for x_value, y_value, row in zip(x_values, y_values, candidates):
        axis.annotate(_label(row), (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel(x_key)
    axis.set_ylabel(y_key)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, destination)
    plt.close(figure)


def _line_by_model(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    destination: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    candidates = [
        row
        for row in rows
        if isinstance(row.get(x_key), (int, float))
        and isinstance(row.get(y_key), (int, float))
    ]
    if not candidates:
        _placeholder(
            destination,
            title,
            f"Requires numeric `{x_key}` and `{y_key}` values.",
        )
        return
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for model_id, group in pd.DataFrame(candidates).groupby("model_id"):
        ordered = group.sort_values(x_key)
        axis.plot(ordered[x_key], ordered[y_key], marker="o", label=model_id)
    axis.set_xlabel(x_key)
    axis.set_ylabel(y_key)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, destination)
    plt.close(figure)


def _nested_per_class(
    rows: list[dict[str, Any]], destination: Path, title: str
) -> None:
    import matplotlib.pyplot as plt

    records = []
    for row in rows:
        for class_name, values in (row.get("per_class") or {}).items():
            if isinstance(values, dict) and isinstance(values.get("AP"), (int, float)):
                records.append(
                    {
                        "label": _label(row),
                        "class": class_name,
                        "AP": float(values["AP"]),
                    }
                )
    if not records:
        _placeholder(destination, title, "Requires nested per-class AP outputs.")
        return
    frame = pd.DataFrame(records).pivot(index="class", columns="label", values="AP")
    axis = frame.plot(kind="bar", figsize=(11, 6))
    axis.set_title(title)
    axis.set_ylabel("AP")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    figure = axis.get_figure()
    figure.tight_layout()
    _save_figure(figure, destination)
    plt.close(figure)


def _nested_curve(
    rows: list[dict[str, Any]], destination: Path, title: str
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5.5))
    plotted = False
    for row in rows:
        curves = row.get("confidence_curves") or {}
        recall = curves.get("recall")
        precision = curves.get("precision")
        if recall and precision:
            axis.plot(recall, precision, label=_label(row))
            plotted = True
    if not plotted:
        plt.close(figure)
        _placeholder(destination, title, "Requires confidence/PR curve arrays.")
        return
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, destination)
    plt.close(figure)


def _calibration_figure(
    rows: list[dict[str, Any]], destination: Path, title: str
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6))
    plotted = False
    for row in rows:
        bins = (row.get("calibration") or {}).get("bins", [])
        x_values = [item.get("confidence") for item in bins if item.get("count", 0)]
        y_values = [item.get("accuracy") for item in bins if item.get("count", 0)]
        if x_values and y_values:
            axis.plot(x_values, y_values, marker="o", label=_label(row))
            plotted = True
    if not plotted:
        plt.close(figure)
        _placeholder(destination, title, "Requires detection calibration bins.")
        return
    axis.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Empirical precision")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, destination)
    plt.close(figure)


def _error_figure(
    rows: list[dict[str, Any]], destination: Path, title: str
) -> None:
    import matplotlib.pyplot as plt

    records = []
    for row in rows:
        for error_type, value in (row.get("error_counts") or {}).items():
            if isinstance(value, (int, float)):
                records.append(
                    {"label": _label(row), "error": error_type, "count": value}
                )
    if not records:
        _placeholder(destination, title, "Requires error decomposition outputs.")
        return
    frame = pd.DataFrame(records).pivot(index="error", columns="label", values="count")
    axis = frame.plot(kind="bar", figsize=(11, 6))
    axis.set_title(title)
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", rotation=30)
    figure = axis.get_figure()
    figure.tight_layout()
    _save_figure(figure, destination)
    plt.close(figure)


def generate_figures(rows: list[dict[str, Any]], figure_dir: Path) -> list[Path]:
    """Generate all required figure names, using honest placeholders if absent."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Callable[[], None]]] = [
        ("01_map_comparison", lambda: _bar_figure(rows, "mAP", figure_dir / "01_map_comparison", "mAP comparison")),
        ("02_aptiny_comparison", lambda: _bar_figure(rows, "APtiny", figure_dir / "02_aptiny_comparison", "APtiny comparison")),
        ("03_per_class_ap", lambda: _nested_per_class(rows, figure_dir / "03_per_class_ap", "Per-class AP")),
        ("04_accuracy_latency_pareto", lambda: _scatter_figure(rows, "mean_latency_ms", "mAP", figure_dir / "04_accuracy_latency_pareto", "Accuracy versus batch-1 latency")),
        ("05_accuracy_vram_pareto", lambda: _scatter_figure(rows, "peak_inference_vram_bytes", "mAP", figure_dir / "05_accuracy_vram_pareto", "Accuracy versus peak inference VRAM")),
        ("06_aptiny_latency_pareto", lambda: _scatter_figure(rows, "mean_latency_ms", "APtiny", figure_dir / "06_aptiny_latency_pareto", "APtiny versus latency")),
        ("07_aptiny_training_time", lambda: _scatter_figure(rows, "total_training_seconds", "APtiny", figure_dir / "07_aptiny_training_time", "APtiny versus training time")),
        ("08_map_trainable_parameters", lambda: _scatter_figure(rows, "trainable_parameters", "mAP", figure_dir / "08_map_trainable_parameters", "mAP versus trainable parameters")),
        ("09_map_flops", lambda: _scatter_figure(rows, "FLOPs", "mAP", figure_dir / "09_map_flops", "mAP versus FLOPs")),
        ("10_latency_resolution", lambda: _line_by_model(rows, "evaluation_resolution", "mean_latency_ms", figure_dir / "10_latency_resolution", "Latency versus input resolution")),
        ("11_vram_resolution", lambda: _line_by_model(rows, "evaluation_resolution", "peak_inference_vram_bytes", figure_dir / "11_vram_resolution", "VRAM versus input resolution")),
        ("12_training_time_epoch", lambda: _line_by_model(rows, "epoch", "epoch_seconds", figure_dir / "12_training_time_epoch", "Training time versus epoch")),
        ("13_time_to_convergence", lambda: _bar_figure(rows, "time_to_95pct_final_map", figure_dir / "13_time_to_convergence", "Time to 95% of final mAP")),
        ("14_precision_recall", lambda: _nested_curve(rows, figure_dir / "14_precision_recall", "Precision-recall curves")),
        ("15_calibration", lambda: _calibration_figure(rows, figure_dir / "15_calibration", "Detection reliability diagrams")),
        ("16_error_distribution", lambda: _error_figure(rows, figure_dir / "16_error_distribution", "Error-type distribution")),
    ]
    for name, function in jobs:
        function()
    for name, title, requirement in [
        ("17_occlusion", "Performance by occlusion level", "Requires attribute-slice evaluation outputs."),
        ("18_truncation", "Performance by truncation level", "Requires attribute-slice evaluation outputs."),
        ("19_density", "Performance by object-density level", "Requires density-slice evaluation outputs."),
        ("20_qualitative_side_by_side", "Qualitative side-by-side predictions", "Requires rendered shared-image prediction panels."),
    ]:
        _placeholder(figure_dir / name, title, requirement)
    return sorted(figure_dir.glob("*.png"))


def _scalar_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    serializable = []
    for row in rows:
        serializable.append(
            {
                key: value
                if isinstance(value, (str, int, float, bool)) or value is None
                else json.dumps(value, sort_keys=True)
                for key, value in row.items()
            }
        )
    return pd.DataFrame(serializable)


def _statistical_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            row
            for row in rows
            if isinstance(row.get("mAP"), (int, float))
            and row.get("model_id") is not None
        ]
    )
    if frame.empty:
        return pd.DataFrame()
    group_columns = [
        column
        for column in ("model_id", "dataset_track", "evaluation_resolution")
        if column in frame.columns
    ]
    output = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        for metric in ("mAP", "APtiny", "mean_latency_ms", "peak_inference_vram_bytes"):
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
            row[f"{metric}_ci95_half_width"] = (
                1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))
                if len(values) > 1
                else None
            )
        row["number_of_runs"] = len(group)
        output.append(row)
    return pd.DataFrame(output)


def generate_report(
    rows: list[dict[str, Any]], report_dir: str | Path
) -> dict[str, str]:
    output = Path(report_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataframe = _scalar_frame(rows)
    dataframe.to_csv(output / "final_results.csv", index=False)
    write_json(output / "final_results.json", rows)
    recommendations = recommendation_matrix(rows)
    recommendations_frame = pd.DataFrame(recommendations)
    recommendations_frame.to_csv(
        output / "model_recommendation_matrix.csv", index=False
    )
    statistical = _statistical_summary(rows)
    statistical.to_csv(output / "statistical_summary.csv", index=False)
    def render_figures() -> list[Path]:
        configure_headless_matplotlib()
        return generate_figures(rows, output / "figures")

    generated_figures, _ = run_optional_output(
        "generate_report_figures",
        output,
        render_figures,
    )
    figure_paths = generated_figures or []

    def markdown_table(frame: pd.DataFrame, operation: str, empty: str) -> str:
        if frame.empty:
            return empty
        rendered, _ = run_optional_output(
            operation, output, lambda: frame.to_markdown(index=False)
        )
        return rendered if rendered is not None else frame.to_string(index=False)

    section_text = {
        "Abstract": "Generated from registered measured runs. Missing experiments remain explicitly missing; no values are invented.",
        "Research questions": "Accuracy, tiny-object detection, latency, memory, scaling, convergence, exportability, internal behavior, and deployment trade-offs.",
        "Dataset and class mapping": "Two-class and ten-class tracks are separated. Internal two-class mAP is never compared directly with published ten-class mAP.",
        "Model architectures": "CNN, Swin Transformer, VMamba selective state-space, and end-to-end RT-DETRv2 integrations are represented by concrete run manifests.",
        "License review": "See LICENSES.md. Dataset usage remains research-only unless the dataset owner grants broader rights.",
        "Experimental protocol": "Compare only matching class mappings, splits, seeds, recipes, resolutions, hardware, and precision modes.",
        "Learning-rate search": "Deterministic LR-search candidates are stored separately and must be distinguished from complete-train final runs.",
        "Training results": markdown_table(
            dataframe, "render_training_markdown_table", "No completed evaluation rows."
        ),
        "Detection results": "COCO, tiny-object, per-class, localization, confidence, and error outputs are stored in final_results.json.",
        "Efficiency results": "Latency values require synchronized warm-up and timing on shared hardware. Export failures remain valid measured outcomes.",
        "Resolution scaling": "640, 1024, 1280, and native resolution rows are identified by evaluation_resolution.",
        "Ablation studies": "P2, resolution, tiling, query count, max detections, pretraining, frozen stages, augmentation, multi-scale, precision, and recipe must be isolated as controlled run metadata.",
        "Statistical analysis": markdown_table(
            statistical,
            "render_statistical_markdown_table",
            "At least three compatible seeds are needed for stable summaries.",
        ),
        "Architecture visualization": "Feature stages, FPN levels, proposals/RoIs, selective-scan stages, queries, references, and decoder refinement are generated in notebook 08.",
        "Error analysis": "Classification, localization, duplicate/background, and miss counts are retained alongside confidence and localization diagnostics.",
        "Published benchmark comparison": "Only ten-class, protocol-compatible literature rows belong here. Missing literature values are left null or not reported.",
        "Limitations": "GPU-specific frameworks, custom CUDA kernels, TensorRT, energy estimates, and unpublished test labels may limit complete execution on a given runtime.",
        "Deployment considerations": "Use measured export status, latency, VRAM, throughput, supported operators, resolution, and postprocessing costs.",
        "Final model recommendations": markdown_table(
            recommendations_frame,
            "render_recommendations_markdown_table",
            "Insufficient measured results.",
        ),
        "Reproduction instructions": "Follow README notebook order, preserve manifests/configs/environment files, and rerun evaluation/report generation without retraining.",
    }
    lines = ["# VisDrone Architecture Benchmark Final Report", ""]
    for heading, body in section_text.items():
        lines.extend([f"## {heading}", "", body, ""])
    lines.extend(["## Generated figures", ""])
    for path in figure_paths:
        lines.append(f"- `{path.relative_to(output)}`")
    markdown = "\n".join(lines) + "\n"
    write_text_atomic(output / "final_report.md", markdown)

    html_sections = ["<h1>VisDrone Architecture Benchmark Final Report</h1>"]
    for heading, body in section_text.items():
        html_sections.append(f"<h2>{html.escape(heading)}</h2>")
        html_sections.append(f"<pre>{html.escape(body)}</pre>")
    html_sections.append("<h2>Generated figures</h2>")
    for path in figure_paths:
        relative = path.relative_to(output)
        html_sections.append(
            f'<figure><img src="{html.escape(str(relative))}" style="max-width:100%"><figcaption>{html.escape(path.stem)}</figcaption></figure>'
        )
    write_text_atomic(
        output / "final_report.html",
        "<html><head><meta charset='utf-8'><title>VisDrone Benchmark</title></head><body>"
        + "".join(html_sections)
        + "</body></html>",
    )

    def generate_pdf() -> None:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(output / "experiment_summary.pdf") as pdf:
            title_figure = plt.figure(figsize=(8.27, 11.69))
            title_figure.text(0.08, 0.95, "VisDrone Architecture Benchmark", fontsize=18)
            title_figure.text(
                0.08,
                0.90,
                "Generated from measured registry outputs. Missing figures are labeled placeholders.",
                fontsize=10,
                wrap=True,
            )
            pdf.savefig(title_figure)
            plt.close(title_figure)
            for figure_path in figure_paths:
                image = mpimg.imread(figure_path)
                figure, axis = plt.subplots(figsize=(11.69, 8.27))
                axis.imshow(image)
                axis.set_axis_off()
                axis.set_title(figure_path.stem)
                figure.tight_layout()
                pdf.savefig(figure)
                plt.close(figure)

    run_optional_output("generate_report_pdf", output, generate_pdf)

    return {
        "markdown": str(output / "final_report.md"),
        "html": str(output / "final_report.html"),
        "csv": str(output / "final_results.csv"),
        "json": str(output / "final_results.json"),
        "recommendations": str(output / "model_recommendation_matrix.csv"),
        "pdf": str(output / "experiment_summary.pdf"),
        "figures": str(output / "figures"),
    }
