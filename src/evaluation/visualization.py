"""Publication plots and framework-neutral activation inspection helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def save_bar(
    dataframe: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    output_base: str | Path,
) -> None:
    output = Path(output_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    dataframe.plot.bar(x=x, y=y, ax=axis, legend=False)
    axis.set_title(title)
    axis.set_ylabel(y)
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def save_scatter(
    dataframe: pd.DataFrame,
    x: str,
    y: str,
    label: str,
    title: str,
    output_base: str | Path,
) -> None:
    output = Path(output_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(dataframe[x], dataframe[y])
    for _, row in dataframe.iterrows():
        axis.annotate(
            str(row[label]),
            (row[x], row[y]),
            xytext=(4, 4),
            textcoords="offset points",
        )
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def select_module_names(
    model: Any, keywords: Sequence[str], limit: int = 16
) -> list[str]:
    """Select real installed module names matching architecture-specific terms."""
    lowered = [keyword.lower() for keyword in keywords]
    names = [
        name
        for name, _ in model.named_modules()
        if name and any(keyword in name.lower() for keyword in lowered)
    ]
    return names[:limit]


def capture_module_outputs(
    model: Any, module_names: Iterable[str]
) -> tuple[dict[str, Any], list[Any]]:
    """Attach forward hooks to exact names returned by ``named_modules``."""
    requested = set(module_names)
    outputs: dict[str, Any] = {}
    handles = []
    for name, module in model.named_modules():
        if name not in requested:
            continue

        def hook(_module: Any, _inputs: Any, output: Any, key: str = name) -> None:
            outputs[key] = output

        handles.append(module.register_forward_hook(hook))
    missing = requested - set(outputs) - {
        name for name, _ in model.named_modules() if name in requested
    }
    if missing:
        raise KeyError(f"module names were not found: {sorted(missing)}")
    return outputs, handles


def _first_tensor(value: Any) -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for child in value.values():
            result = _first_tensor(child)
            if result is not None:
                return result
    if isinstance(value, (list, tuple)):
        for child in value:
            result = _first_tensor(child)
            if result is not None:
                return result
    for attribute in ("last_hidden_state", "hidden_states"):
        if hasattr(value, attribute):
            result = _first_tensor(getattr(value, attribute))
            if result is not None:
                return result
    return None


def activation_views(value: Any) -> dict[str, np.ndarray] | None:
    """Return mean, maximum, and one-component PCA views for a feature tensor."""
    tensor = _first_tensor(value)
    if tensor is None:
        return None
    array = tensor.detach().float().cpu()
    if array.ndim == 3:  # B,N,C token sequence; map only if N is square.
        batch, tokens, channels = array.shape
        side = int(round(tokens**0.5))
        if side * side != tokens:
            return None
        array = array.transpose(1, 2).reshape(batch, channels, side, side)
    if array.ndim != 4:
        return None
    feature = array[0]
    mean_map = feature.mean(dim=0).numpy()
    maximum_map = feature.max(dim=0).values.numpy()
    channels, height, width = feature.shape
    samples = feature.reshape(channels, -1).T.numpy()
    samples = samples - samples.mean(axis=0, keepdims=True)
    # SVD avoids a hard dependency on scikit-learn in visualization utilities.
    if samples.shape[0] > 8192:
        indices = np.linspace(0, samples.shape[0] - 1, 8192).astype(int)
        fit_samples = samples[indices]
    else:
        fit_samples = samples
    _, _, right = np.linalg.svd(fit_samples, full_matrices=False)
    component = samples @ right[0]
    pca_map = component.reshape(height, width)
    return {"mean": mean_map, "maximum": maximum_map, "pca": pca_map}


def plot_activation_views(
    outputs: dict[str, Any], maximum_modules: int = 6
) -> Any:
    """Plot interpretable activation summaries for captured modules."""
    available = []
    for name, output in outputs.items():
        views = activation_views(output)
        if views is not None:
            available.append((name, views))
        if len(available) >= maximum_modules:
            break
    if not available:
        raise RuntimeError("No captured outputs had a plottable 2D feature shape.")
    figure, axes = plt.subplots(
        len(available), 3, figsize=(12, 3.5 * len(available)), squeeze=False
    )
    for row, (name, views) in enumerate(available):
        for column, key in enumerate(("mean", "maximum", "pca")):
            axes[row, column].imshow(views[key])
            axes[row, column].set_title(f"{name}: {key}")
            axes[row, column].set_axis_off()
    figure.tight_layout()
    return figure


def draw_predictions(
    image: Image.Image,
    prediction: dict[str, Any],
    class_names: Sequence[str],
    threshold: float = 0.25,
) -> Image.Image:
    """Draw a normalized adapter prediction without architecture-specific code."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for box, score, label in zip(
        prediction["boxes"], prediction["scores"], prediction["labels"]
    ):
        if float(score) < threshold:
            continue
        x1, y1, x2, y2 = map(float, box)
        draw.rectangle((x1, y1, x2, y2), width=2)
        class_index = int(label) - 1
        name = (
            class_names[class_index]
            if 0 <= class_index < len(class_names)
            else str(label)
        )
        draw.text((x1, y1), f"{name} {float(score):.2f}")
    return canvas
