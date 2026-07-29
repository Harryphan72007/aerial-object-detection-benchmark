# LR-controlled benchmark implementation audit

## Baseline optimizer resolution

The old per-model `2class.yaml` files contain generic `1e-4` values and are not
authoritative for MMDetection. The LR search now calls
`resolve_baseline_optimizer()` before generating candidates.

| Model | Authoritative source | Optimizer details |
|---|---|---|
| `faster_rcnn_resnet50` | Final config loaded from `MMDET_ROOT/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py` | Runtime-resolved `optim_wrapper.optimizer.lr`, type, weight decay, `paramwise_cfg`, warmup, and scheduler |
| `faster_rcnn_swin_t` | Final config loaded from `MMDET_ROOT/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py` | Runtime-resolved fields; mask branch is stripped only after the optimizer audit |
| `faster_rcnn_vmamba_t` | Final config loaded from `VMAMBA_ROOT/detection/configs/vssm/mask_rcnn_vssm_fpn_coco_tiny.py` | Runtime-resolved fields, including any VMamba parameter-wise LR rules |
| `rtdetrv2_l` | `src/training/recipes.py`, shared with optimizer construction | AdamW, LR `1e-4`, weight decay `0.05`, cosine scheduler, no warmup, no separate parameter groups |

MMDetection values are deliberately not copied into repository constants:
upstream and VMamba checkouts are environment-specific, so the final loaded
configuration is the only authoritative value. The candidate YAML records the
resolved path and fields for each actual run.

## Previous behavior and corrections

- `scripts/tune.py` previously used Optuna to vary resolution, accumulation,
  weight decay, architecture parameters, thresholds, augmentation, and other
  dimensions. It also evaluated against official validation during tuning.
  That implementation has been replaced by the LR-only entry point.
- `TrainingOrchestrator` previously always selected canonical train and
  official validation files. It now accepts explicit, recorded annotation and
  image roots, allowing search validation to come only from official train.
- RT-DETR previously constructed `CosineAnnealingLR(T_max=args.epochs)`.
  Resuming a candidate at 2, 5, 10, and 15 epochs therefore changed its
  schedule horizon at every rung. `--scheduler-horizon 15` is now distinct from
  the current target epoch and is verified against the checkpoint on resume.
- RT-DETR checkpoints already contained model, optimizer, scheduler, scaler,
  epoch, and best-metric state. They now also contain Python, NumPy, CPU/CUDA
  Torch RNG, DataLoader generator, and scheduler-horizon state.
- MMEngine resume restores model/optimizer/scheduler/scaler/epoch and framework
  random state. Its runtime config now fixes the epoch scheduler policy to the
  declared horizon once, stores a scheduler contract, rejects horizon changes,
  and enables deterministic execution.
- Search runs use isolated candidate directories and are not entered in the
  final checkpoint registry. Final training uses a new run directory, passes no
  resume ID, and therefore reloads the configured original pretrained weights.

## Common evaluation

`scripts/evaluate.py` remains the reporting authority. Every adapter exports
complete low-floor (`0.001`) predictions to repository COCO JSON, caps the same
maximum detections, and calls the shared COCO/aerial/latency evaluation code.
The final workflow disables in-training validation, aliases the final epoch
checkpoint for registry compatibility, then evaluates exactly once on complete
official validation through this common path.

## Runtime-specific verification still required

CPU tests prove deterministic grids/manifests/ranking, full-train identity,
effective batch policy, range-test artifacts, selected-YAML round trips, and
fixed-horizon scheduler resume equivalence. The following remain GPU/framework
integration checks:

- actual baseline values and parameter groups from each pinned MMDetection or
  VMamba checkout;
- MMEngine checkpoint/sampler restoration with those concrete versions;
- CUDA AMP range tests and all successive-halving rungs;
- VMamba extension build and verified pretrained checkpoint availability;
- measured one-day workload and full 25-epoch final runs;
- complete common evaluation/profiling latency on the target GPU.
