# Benchmark methodology

Track A (`2class`) maps pedestrian+people to PERSON and the eight vehicle-like
classes to VEHICLE. Track B (`10class`) preserves the original ten VisDrone
classes. The tracks are stored, discovered, evaluated, and reported separately.

`lr_controlled_v1` (the seed-42, 640-pixel, effective-batch-8, LR-only
successive-halving workflow of notebooks 01-03) is **retired**. Its entry
point raises rather than starting a new run; it is described here only so
historical artifacts remain interpretable. `two_stage_random_hpo_v1` is the
only live protocol.

`two_stage_random_hpo_v1` uses search seed 42, primary mAP50-95 and APtiny
tie-break. Learning rate is the only suggested parameter. Phase A runs five
broad model-specific random trials. Phase B runs five more; it derives a narrower
range from the strongest valid Phase A trials except for RT-DETRv2, whose
controlled `1e-6` to `5e-4` range remains unchanged in both phases. A trial fails
if the requested LR is unsupported, ignored, or changed by the backend.

Every trial starts in a distinct directory with resume disabled, so model,
optimizer, scheduler, and scaler state cannot leak between LR candidates.
RT-DETRv2 numerical divergence and CUDA OOM are persisted as `PRUNED` trials.
Other runtime or implementation errors remain fatal.

Search train and validation are deterministic subsets of official train.
Official validation never tunes a model, and it never selects a checkpoint. The
selected config is frozen once. Baseline and tuned recipes restart from original
pretrained weights and train on official train **minus a fixed, seeded
model-selection holdout** (`final_train_seed42.json`). The final `best.pth` is
selected on that held-out `model_selection_seed42.json` split — which is disjoint
from both search subsets and from official validation — and official validation
is evaluated exactly once, at the end, as the reported number. This removes the
checkpoint-selection bias of the earlier per-epoch official-validation selection.
Final manifests record `protocol_id`, `run_kind`, and `baseline_or_tuned`.

The headline run matrix is one tuned recipe at seed 42 on the `2class` track per
model — the smallest scientifically defensible matrix that fits about one model
per GPU-day. Because a single seed gives no variance estimate, architecture
differences below typical VisDrone seed noise (~0.3–0.8 mAP) are not claimed from
it. The full `baseline`+`tuned` × seed `17/42/3407` matrix is an explicit opt-in
(`full_matrix=True`); the default-LR baseline is recorded only as a diagnostic
(it is a second, worse learning rate, not a scientific control), and the extra
seeds provide the variance estimate required before publishing conclusions. Every
matrix value is read from `configs/controlled/benchmark.yaml`, not hardcoded.

Evaluation includes COCO AP, AP50/AP75, size and per-class metrics,
precision/recall/F1 and PR curves, calibration, error decomposition, training
and convergence facts, parameters, checkpoint size, valid FLOPs/MACs, memory,
latency percentiles (p50/p90/p95/p99), FPS, throughput, and resolution scaling
when measured. Nothing is fabricated when a model, run, or measurement is
missing: a missing metric serialises as `null`, and a FLOPs/MACs measurement that
fails or is unavailable records `null` with a reason — never `0`, which would
read as a free model. Latency is always labelled with the batch size it was
measured at; batch latency is never presented as single-image latency.

## Reproducibility policy

GPU training is **best-effort deterministic**, not guaranteed bitwise deterministic.
Runs record their seeds, environment, source revision, and relevant child-process
environment. Model subprocesses set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing
PyTorch and request deterministic algorithms with `warn_only=True`. CUDA operators
without deterministic implementations, including the RT-DETR grid-sampling backward
path in the observed stack, may still vary between runs. These warnings are retained;
strict deterministic mode is not claimed because it can terminate supported models.
