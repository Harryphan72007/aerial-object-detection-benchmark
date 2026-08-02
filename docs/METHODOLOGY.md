# Benchmark methodology

Track A (`2class`) maps pedestrian+people to PERSON and the eight vehicle-like
classes to VEHICLE. Track B (`10class`) preserves the original ten VisDrone
classes. The tracks are stored, discovered, evaluated, and reported separately.

`lr_controlled_v1` preserves the existing seed-42, 640-pixel,
effective-batch-8, LR-only successive-halving workflow.

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
Official validation never tunes a model. The selected config is frozen once.
Baseline and tuned recipes restart from original pretrained weights, use full
official train, and run seeds 17, 42, and 3407. Final manifests record
`protocol_id`, `run_kind`, and `baseline_or_tuned`.

Evaluation includes COCO AP, AP50/AP75, size and per-class metrics,
precision/recall/F1 and PR curves, calibration, error decomposition, training
and convergence facts, parameters, checkpoint size, valid FLOPs/MACs, memory,
latency percentiles, FPS, throughput, and resolution scaling when measured.
Nothing is fabricated when a model, run, or measurement is missing.

## Reproducibility policy

GPU training is **best-effort deterministic**, not guaranteed bitwise deterministic.
Runs record their seeds, environment, source revision, and relevant child-process
environment. Model subprocesses set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing
PyTorch and request deterministic algorithms with `warn_only=True`. CUDA operators
without deterministic implementations, including the RT-DETR grid-sampling backward
path in the observed stack, may still vary between runs. These warnings are retained;
strict deterministic mode is not claimed because it can terminate supported models.
