"""Latency statistics and accuracy-efficiency utilities."""
from __future__ import annotations
from typing import Any
import numpy as np

def summarize_latencies(milliseconds:list[float])->dict[str,float]:
    arr=np.asarray(milliseconds,dtype=float)
    if arr.size==0:return {k:0.0 for k in ("mean_latency_ms","median_latency_ms","p90_latency_ms","p95_latency_ms","p99_latency_ms","fps")}
    return {"mean_latency_ms":float(arr.mean()),"median_latency_ms":float(np.median(arr)),"p90_latency_ms":float(np.quantile(arr,.90)),"p95_latency_ms":float(np.quantile(arr,.95)),"p99_latency_ms":float(np.quantile(arr,.99)),"fps":float(1000/arr.mean())}
def latency_report(
    milliseconds: list[float],
    *,
    batch_size: int,
    warmup: int = 0,
    hardware: str | None = None,
) -> dict[str, Any]:
    """Latency percentiles labelled with the batch size they were measured at.

    Batch latency must never be presented as single-image latency: the report
    records ``batch_size``, a ``single_image_latency`` flag, and an explicit
    ``latency_label``. A missing measurement serialises as ``None`` (null), never
    ``0`` — which would read as an infinitely fast model.
    """
    if not milliseconds:
        stats = {
            key: None
            for key in (
                "mean_latency_ms",
                "median_latency_ms",
                "p50_latency_ms",
                "p90_latency_ms",
                "p95_latency_ms",
                "p99_latency_ms",
                "fps",
            )
        }
    else:
        stats = dict(summarize_latencies(milliseconds))
        stats["p50_latency_ms"] = stats["median_latency_ms"]
    single_image = int(batch_size) == 1
    return {
        **stats,
        "batch_size": int(batch_size),
        "warmup": int(warmup),
        "hardware": hardware,
        "single_image_latency": single_image,
        "latency_label": "single-image" if single_image else f"batch-{int(batch_size)}",
    }


def assert_latency_labeling(report: dict[str, Any]) -> None:
    """Reject a report that labels multi-image batch latency as single-image."""
    batch_size = int(report.get("batch_size", 0))
    if batch_size < 1:
        raise ValueError("latency report must record a batch_size >= 1")
    single = bool(report.get("single_image_latency"))
    if single and batch_size != 1:
        raise ValueError(
            f"batch-{batch_size} latency cannot be labelled single-image"
        )
    if report.get("latency_label") == "single-image" and batch_size != 1:
        raise ValueError("latency_label 'single-image' requires batch_size == 1")


def accuracy_gain_per_cost(rows:list[dict[str,Any]],accuracy_key:str="mAP",cost_key:str="mean_latency_ms")->list[dict[str,Any]]:
    rows=sorted(rows,key=lambda r:float(r[cost_key])); out=[]
    for prev,cur in zip(rows,rows[1:]):
        dc=float(cur[cost_key])-float(prev[cost_key]); da=float(cur[accuracy_key])-float(prev[accuracy_key]); out.append({"from":prev.get("label"),"to":cur.get("label"),"accuracy_gain":da,"cost_gain":dc,"accuracy_gain_per_cost":da/dc if dc else None})
    return out

def pareto_frontier(rows:list[dict[str,Any]],accuracy_key:str,cost_key:str)->list[dict[str,Any]]:
    result=[]
    for row in rows:
        dominated=any(float(other[accuracy_key])>=float(row[accuracy_key]) and float(other[cost_key])<=float(row[cost_key]) and (float(other[accuracy_key])>float(row[accuracy_key]) or float(other[cost_key])<float(row[cost_key])) for other in rows)
        if not dominated:result.append(row)
    return sorted(result,key=lambda x:float(x[cost_key]))
