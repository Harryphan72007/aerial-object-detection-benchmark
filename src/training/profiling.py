"""Parameter, memory, timing, and optional NVML measurements."""
from __future__ import annotations
import os, time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping
import psutil

def parameter_counts(model:Any)->dict[str,int]:
    total=sum(p.numel() for p in model.parameters()); trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters":int(total),"trainable_parameters":int(trainable),"frozen_parameters":int(total-trainable)}

def parameter_counts_by_module(model:Any)->dict[str,int]:
    return {name or "<root>":sum(p.numel() for p in module.parameters(recurse=False)) for name,module in model.named_modules() if any(True for _ in module.parameters(recurse=False))}

@contextmanager
def timed_section() -> Iterator[dict[str,float]]:
    result={"seconds":0.0}; start=time.perf_counter()
    try: yield result
    finally: result["seconds"]=time.perf_counter()-start

def memory_snapshot()->dict[str,int]:
    result={"cpu_rss_bytes":psutil.Process(os.getpid()).memory_info().rss}
    try:
        import torch
        if torch.cuda.is_available(): result.update({"gpu_allocated_bytes":torch.cuda.memory_allocated(),"gpu_reserved_bytes":torch.cuda.memory_reserved(),"gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated(),"gpu_peak_reserved_bytes":torch.cuda.max_memory_reserved()})
    except ImportError: pass
    return {k:int(v) for k,v in result.items()}

def optimizer_state_bytes(optimizer:Any)->int:
    total=0
    for state in optimizer.state.values():
        for value in state.values():
            if hasattr(value,"numel") and hasattr(value,"element_size"): total+=value.numel()*value.element_size()
    return int(total)


def _default_synchronize() -> None:
    """Flush any pending CUDA work so wall-clock timing is accurate."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def measure_iteration_seconds(
    step: Callable[[int], Any],
    *,
    warmup: int,
    iterations: int,
    synchronize: Callable[[], None] | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Time ``iterations`` calls of ``step`` after excluding ``warmup`` calls.

    The device is synchronized once after warm-up (so warm-up work never leaks
    into the timed window) and once after the timed loop (so asynchronous GPU
    work is flushed before the clock stops). On CPU the default synchronize is a
    no-op. Warm-up iterations are never counted toward ``seconds_per_iteration``.
    """
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    sync = synchronize if synchronize is not None else _default_synchronize
    for index in range(warmup):
        step(index)
    sync()
    started = timer()
    for index in range(iterations):
        step(warmup + index)
    sync()
    elapsed = float(timer() - started)
    return {
        "warmup": int(warmup),
        "iterations": int(iterations),
        "total_seconds": elapsed,
        "seconds_per_iteration": elapsed / iterations,
    }


def render_runtime_budget(
    throughput: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    official_train_images: int,
    search_train_fraction: float = 0.20,
    model_selection_fraction: float = 0.05,
) -> dict[str, Any]:
    """Compute a per-model GPU-hour budget from measured seconds/iteration.

    The controlled protocol uses batch size 1, so iterations == images. HPO runs
    ``phase_trials`` trials in each phase at the phase epoch count over the search
    subset; the headline final run trains ``final_recipes x final_seeds`` over
    official train minus the model-selection holdout. The full opt-in matrix uses
    ``full_matrix_recipes x full_matrix_seeds``. ``t_iter`` is measured, never
    guessed; a model missing from ``throughput`` is reported as ``null``.
    """
    search_images = round(official_train_images * search_train_fraction)
    final_images = official_train_images - round(
        official_train_images * model_selection_fraction
    )
    phase_trials = int(protocol["phase_trials"])
    hpo_iters = phase_trials * (
        int(protocol["phase_a_epochs"]) + int(protocol["phase_b_epochs"])
    ) * search_images
    final_epochs = int(protocol["final_train_epochs"])

    def _final_iters(recipes: Any, seeds: Any) -> int:
        return len(list(recipes)) * len(list(seeds)) * final_epochs * final_images

    headline_final = _final_iters(
        protocol["final_recipes"], protocol["final_seeds"]
    )
    full_final = _final_iters(
        protocol["full_matrix_recipes"], protocol["full_matrix_seeds"]
    )
    models: dict[str, Any] = {}
    for model_id in sorted(throughput):
        record = throughput[model_id]
        t_iter = record.get("seconds_per_iteration")
        if t_iter is None:
            models[model_id] = {
                "seconds_per_iteration": None,
                "reason": "no measured throughput",
            }
            continue
        t_iter = float(t_iter)
        hpo_hours = hpo_iters * t_iter / 3600.0
        models[model_id] = {
            "seconds_per_iteration": t_iter,
            "gpu_name": record.get("gpu_name"),
            "hpo_hours": round(hpo_hours, 2),
            "headline_final_hours": round(headline_final * t_iter / 3600.0, 2),
            "headline_total_hours": round(
                (hpo_iters + headline_final) * t_iter / 3600.0, 2
            ),
            "full_matrix_total_hours": round(
                (hpo_iters + full_final) * t_iter / 3600.0, 2
            ),
        }
    return {
        "iterations": {
            "search_images": search_images,
            "final_train_images": final_images,
            "hpo_iterations": hpo_iters,
            "headline_final_iterations": headline_final,
            "full_matrix_final_iterations": full_final,
        },
        "models": models,
    }
