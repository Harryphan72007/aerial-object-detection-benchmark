"""Parameter, memory, timing, and optional NVML measurements."""
from __future__ import annotations
import os, time
from contextlib import contextmanager
from typing import Any, Iterator
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
