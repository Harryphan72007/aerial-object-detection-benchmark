# Benchmark methodology

Track A (`2class`) maps pedestrian+people to PERSON and the eight vehicle-like
classes to VEHICLE. Track B (`10class`) preserves the original ten VisDrone
classes. The tracks are stored, discovered, evaluated, and reported separately.

`lr_controlled_v1` preserves the existing seed-42, 640-pixel,
effective-batch-8, LR-only successive-halving workflow.

`two_stage_random_hpo_v1` uses search seed 42, primary mAP50-95 and APtiny
tie-break. Phase A runs five broad model-specific random trials. Phase B derives
narrower ranges from the strongest valid Phase A trials and runs five more. A
trial fails if any sampled parameter is unsupported, ignored, or changed by the
backend.

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
