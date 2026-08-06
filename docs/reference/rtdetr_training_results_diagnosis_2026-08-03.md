# RT-DETRv2-L training-results diagnosis and remediation report

**Report date:** 2026-08-03

**Repository:** `Harryphan72007/aerial-object-detection-benchmark`

**Model:** `rtdetrv2_l` (`PekingU/rtdetr_v2_r101vd`)

> **Historical scope (post-PR-01 note):** this diagnosis was conducted against
> the earlier `rtdetrv2_l` configuration, which pinned the R101 (**X**) checkpoint.
> The repository has since corrected `rtdetrv2_l` to the intended **L** variant
> (`PekingU/rtdetr_v2_r50vd`). The `r101vd` reference below is preserved as an
> accurate record of the run that was actually observed, not the current config.

**Dataset track:** VisDrone2019-DET two-class PERSON/VEHICLE track

**Evidence reviewed:** the supplied Colab log through epoch 12 of the first
observed run and epoch 1 of the next observed run, the generated training-curve
image, the checked-in model recipe, and the current training/search/checkpoint
implementation.

## 1. Executive conclusion

The live GPU execution is technically healthy, but the first observed run has a
clear generalization failure:

- training loss falls by **50.53%**, from `25.2410` to `12.4874`;
- validation mAP peaks at **0.27883 at epoch 2** and falls to **0.16624 at
  epoch 12**, a **40.38% decline from the peak**;
- APtiny peaks later, at **0.09115 at epoch 5**, but does not sustain the gain;
- the optimizer, CUDA runtime, memory use, data loader, and evaluator remain
  operational and finite throughout;
- early stopping behaves as configured: the mAP best is at epoch 2, followed by
  ten non-improving epochs, so patience 10 stops the run at epoch 12;
- the next independent run begins successfully and reports `mAP=0.27710` and
  `APtiny=0.08351` at its first epoch.

This is not evidence that RT-DETR is incapable of learning VisDrone. It is
evidence that the current fine-tuning recipe lets the raw model specialize to
the training set after it has already reached its best validation state.

The dominant confirmed contributors are:

1. **No stochastic train-only image augmentation is applied.** Training and
   validation both use the same image processor/resize path.
2. **The whole pretrained model is trainable from the first update.** There is
   no initial backbone freeze or staged unfreezing.
3. **The learning-rate schedule has a 100-epoch horizon even for short search
   runs.** At epoch 12 the LR is still 97.48% of its peak value.
4. **EMA is updated and checkpointed but not used for validation or checkpoint
   selection.** All reported mAP values are for raw weights.
5. **Early-stopping patience 10 is long relative to a 15-epoch candidate.** It
   protects the best checkpoint but spends many GPU hours confirming a decline.
6. **A high-capacity R101 detector sees only about 6,471 unique training images
   per epoch.** Without augmentation, additional epochs repeat the same visual
   evidence rather than expanding the effective training distribution.

Reducing the number of images per epoch is **not** the appropriate correction.
It would normally reduce diversity and increase estimator variance. The fix is
to increase the effective diversity of the full dataset and reduce destructive
optimization after the early validation peak.

## 2. Scope, terminology, and limitations

### 2.1 What this report calls “accuracy”

The logged value is COCO-style validation **mAP at IoU 0.50:0.95**, not simple
classification accuracy. Object-detection mAP depends on:

- classification confidence and ranking;
- bounding-box localization;
- duplicate and false-positive detections;
- IoU thresholds from 0.50 through 0.95;
- object size, especially tiny VisDrone instances.

A lower training loss does not mathematically require a higher mAP because the
training objective is a weighted combination of classification, L1 box, GIoU,
auxiliary-decoder, and denoising losses. mAP is a discrete ranking/localization
metric computed on different images.

### 2.2 Run identity limitation

The pasted log does not include the candidate ID, run ID, selected LR label, or
stage banner. It contains 12 sequential epochs, then model construction occurs
again and a new epoch 1 is printed. This report therefore calls them
**observed run 1** and **observed run 2**. It does not assume an identity that is
not present in the supplied text.

### 2.3 What is proven versus inferred

The following are directly observed or confirmed in code:

- the epoch metrics in this report;
- environment versions and GPU reported by the live probe;
- identical processor-only train/validation transforms;
- full-model optimization with no freeze policy;
- the 100-epoch scheduler horizon;
- raw-weight validation despite EMA being enabled;
- patience-10 early stopping;
- checkpoint selection by validation mAP.

The following remain plausible but unverified without inspecting the persistent
dataset and prediction files:

- annotation noise or missing boxes;
- duplicate/near-duplicate video frames;
- class imbalance after the two-class collapse;
- train/validation domain differences;
- sequence leakage or unusually similar frames within a split;
- per-class false-positive and false-negative failure modes.

These possibilities should be audited, but they are not required to explain the
current curve because the confirmed optimization/data-pipeline facts already
provide a sufficient mechanism.

## 3. Runtime and software verification status

### 3.1 Live Colab/GPU evidence

The supplied run establishes the following real-runtime facts:

| Component | Observed result | Interpretation |
|---|---:|---|
| Environment probe | `rtdetr environment: PASS` | Pinned environment reached its READY gate |
| Python | 3.11.13 | Matches the RT-DETR environment contract |
| PyTorch | 2.7.1+cu128 | Correct pinned CUDA build loaded |
| CUDA | 12.8 | CUDA runtime is visible |
| GPU | NVIDIA L4 | Real GPU execution, not a CPU/synthetic test |
| Forward/backward | Completed for 12+ epochs | Adapter and differentiable training path work |
| Validation | COCO metrics emitted every epoch | Prediction conversion and evaluator work |
| Checkpoint lifecycle | Run advances and early-stops | Persistent state and stop policy are active |
| Numerical state | Finite losses and metrics | No NaN/Inf failure is visible |
| Peak reserved VRAM | about 5.50 GiB | Stable and well below L4 capacity |

### 3.2 Expected warnings

The log reports mismatched classifier shapes between the 80-class pretrained
checkpoint and the two-class model. This is expected. The two-class decoder
classification layers, denoising embedding, and encoder score head must be
reinitialized because their output dimensions change. It is not evidence of a
failed checkpoint load.

The `grid_sampler_2d_backward_cuda` deterministic-algorithm warning is also
non-fatal. The repository requests deterministic behavior with `warn_only=True`;
PyTorch warns that this CUDA backward operation has no deterministic
implementation and continues.

### 3.3 Focused local software tests rerun for this report

The following focused files were executed with the repository virtual
environment:

- `test_checkpoint_selection_early_stopping.py`
- `test_training_backend_launch.py`
- `test_rtdetr_shared_trainer.py`
- `test_rtdetr_parameter_groups.py`
- `test_rtdetr_optimizer_scheduler.py`
- `test_rtdetr_hpo_v2.py`
- `test_rtdetrv2_factory.py`
- `test_notebook_first_workflow.py`
- `test_lr_search.py`

Result: **46 passed, 1 skipped**. The skipped test requires local PyTorch, which
is not installed in the Windows test environment. The 212 warnings are
Matplotlib/PyParsing deprecation warnings, not training-contract failures.

These tests verify configuration contracts, optimizer grouping, scheduler
math, checkpoint selection, persistence, workflow state, and command launch.
They do **not** demonstrate live GPU generalization, effective augmentation, or
validation improvement. Passing software tests and poor empirical model quality
are therefore not contradictory.

## 4. Dataset and workload inferred from the live metrics

The run uses batch size 1 with gradient accumulation 8 and reports 809 optimizer
updates per epoch. This is consistent with approximately 6,471 training images:

```text
ceil(6471 / 8) = 809 optimizer updates per epoch
```

The validation output contains 164,400 predictions. With the configured 300
queries/detections per image, that corresponds to 548 validation images:

```text
164400 / 300 = 548 validation images
```

The validation set is not invalid because it is smaller than the training set,
but metrics for rare/tiny cases will have more sampling variability. The much
larger sustained mAP decline cannot be dismissed as ordinary noise alone.

## 5. Complete observed metric history

### 5.1 Observed run 1

| Epoch | Training loss | Detector LR | Gradient norm before clipping | mAP | AP50 | AP75 | APtiny | APsmall | APmedium | APlarge |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25.2410 | 5.050e-5 | 30.17 | 0.2675 | 0.4987 | 0.2492 | 0.0765 | 0.2367 | 0.4858 | 0.8418 |
| 2 | 15.3798 | 1.000e-4 | 17.00 | **0.2788** | **0.5233** | **0.2594** | 0.0829 | **0.2506** | **0.4969** | 0.8220 |
| 3 | 14.5097 | 9.997e-5 | 39.83 | 0.2142 | 0.3992 | 0.1989 | 0.0565 | 0.1799 | 0.3820 | 0.7601 |
| 4 | 14.0381 | 9.990e-5 | 40.84 | 0.2638 | 0.4868 | 0.2475 | 0.0681 | 0.2364 | 0.4824 | 0.7976 |
| 5 | 13.7174 | 9.977e-5 | 32.88 | 0.2731 | 0.5052 | 0.2577 | **0.0912** | 0.2484 | 0.4677 | 0.8207 |
| 6 | 13.4562 | 9.959e-5 | 37.30 | 0.2644 | 0.4983 | 0.2460 | 0.0805 | 0.2369 | 0.4652 | 0.8265 |
| 7 | 13.2552 | 9.937e-5 | 28.78 | 0.1992 | 0.3894 | 0.1785 | 0.0662 | 0.1735 | 0.3446 | 0.7359 |
| 8 | 13.0658 | 9.909e-5 | 26.79 | 0.1860 | 0.3497 | 0.1733 | 0.0428 | 0.1453 | 0.3631 | 0.7615 |
| 9 | 12.9122 | 9.876e-5 | 40.72 | 0.1847 | 0.3718 | 0.1610 | 0.0733 | 0.1778 | 0.3042 | 0.6457 |
| 10 | 12.7699 | 9.838e-5 | 23.16 | 0.1383 | 0.3107 | 0.1050 | 0.0603 | 0.1341 | 0.2253 | 0.5965 |
| 11 | 12.6411 | 9.795e-5 | 53.37 | 0.1518 | 0.3356 | 0.1188 | 0.0650 | 0.1499 | 0.2360 | 0.6481 |
| 12 | 12.4874 | 9.748e-5 | 55.94 | 0.1662 | 0.3545 | 0.1366 | 0.0763 | 0.1711 | 0.2342 | 0.6156 |

### 5.2 Observed run 2, currently available evidence

| Epoch | Training loss | Detector LR | Gradient norm before clipping | mAP | AP50 | AP75 | APtiny | APsmall | APmedium | APlarge |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23.0385 | 5.050e-5 | 33.50 | 0.2771 | 0.5140 | 0.2549 | 0.0835 | 0.2447 | 0.4959 | 0.8457 |

One epoch is insufficient to judge the second run. Its initial metrics are close
to the first run's best, but the repeated pattern reported by the operator makes
the shared recipe the primary object of investigation.

## 6. Quantitative interpretation

### 6.1 Generalization gap behavior

The training objective improves monotonically while validation performance does
not:

| Quantity | Best/start | Final observed | Change |
|---|---:|---:|---:|
| Training loss | 25.2410 at epoch 1 | 12.4874 at epoch 12 | **-50.53%** |
| Validation mAP | 0.27883 at epoch 2 | 0.16624 at epoch 12 | **-40.38% from peak** |
| APtiny | 0.09115 at epoch 5 | 0.07626 at epoch 12 | **-16.34% from peak** |
| APmedium | 0.49694 at epoch 2 | 0.23418 at epoch 12 | **-52.88% from peak** |
| APlarge | 0.84185 at epoch 1 | 0.61558 at epoch 12 | **-26.88% from peak** |

The decline affects small, medium, and large objects, so it is not merely an
APtiny fluctuation. APmedium loses more than half of its peak value.

### 6.2 Runtime behavior

- Mean epoch duration after the cold first epoch: **2,146.86 seconds**, or about
  **35.78 minutes**.
- Total duration for the first 12 epochs: approximately **7.42 GPU-hours**.
- Mean training throughput after epoch 1: **3.139 images/second**.
- Peak reserved VRAM: approximately **5.50 GiB**.
- Maximum reported gradient norm before clipping: **55.94**.

Epoch 1 is slower because data loading and validation warm their caches. Later
epochs are stable. There is no progressive memory leak or throughput collapse.

Because the best mAP occurred at epoch 2, approximately six additional GPU
hours were spent establishing that the run did not recover before patience 10
stopped it. Early stopping preserved correctness but was not cost-efficient.

### 6.3 Gradient norms

The logged gradient norm is the return value of `clip_grad_norm_`, which is the
norm **before** clipping. Values such as 53.37 and 55.94 therefore do not mean
the optimizer applied gradients of that magnitude. The recipe clips them to
0.1. The high pre-clip norms do, however, show that clipping is active and that
the raw optimization signal can be sharp late in the run.

## 7. Confirmed implementation findings

### 7.1 No train-only stochastic augmentation

In [`scripts/run_rtdetr_training.py`](../../scripts/run_rtdetr_training.py),
`encode_record` passes the original image directly to
`RTDetrImageProcessor`. The same function is supplied to both
`train_records` and `validation_records`.

Current behavior:

```python
encoded = processor(images=record["image"], annotations=..., return_tensors="pt")
train_records = CocoDetectionDataset(..., transform=encode_record)
validation_records = CocoDetectionDataset(..., transform=encode_record)
```

The processor resizes/normalizes and encodes boxes, but the repository adds no
random horizontal flip, scale jitter, photometric jitter, box-safe crop, or
other train-only transformation. Consequently, each epoch presents essentially
the same 6,471 images in a different order.

**Effect:** a large detector can progressively specialize to image-specific
textures, backgrounds, scales, and annotation peculiarities. Shuffling changes
batch order, not the data distribution.

### 7.2 The full pretrained network is updated immediately

The model is loaded with `ignore_mismatched_sizes=True` and moved to the GPU.
No parameters are marked `requires_grad=False`; optimizer parameter discovery
therefore includes all trainable backbone and detector parameters.

**Effect:** the newly initialized two-class heads need relatively large updates,
but the pretrained backbone also moves from the beginning. A backbone LR
multiplier of 0.1 helps, but it does not prevent loss of general pretrained
representations. This is a credible catastrophic-forgetting mechanism when the
target dataset is small and repetitive.

### 7.3 Scheduler horizon is much longer than the observed run

[`configs/rtdetrv2_l/performance_recipe_v2.yaml`](../../configs/rtdetrv2_l/performance_recipe_v2.yaml)
sets:

```yaml
detector_learning_rate: 0.0001
backbone_lr_multiplier: 0.1
warmup_epochs: 2
scheduler_horizon_epochs: 100
minimum_lr_factor: 0.01
```

The scheduler is update-based warm-up plus cosine decay. With 809 optimizer
updates per epoch, its LR factor is:

| Epoch | 15-epoch horizon | 25-epoch horizon | Current 100-epoch horizon |
|---:|---:|---:|---:|
| 1 | 0.505000 | 0.505000 | 0.505000 |
| 2 | 1.000000 | 1.000000 | 1.000000 |
| 5 | 0.875513 | 0.959020 | 0.997713 |
| 10 | 0.329471 | 0.732732 | 0.983811 |
| 12 | 0.134487 | 0.605711 | 0.974783 |
| 15 | 0.010000 | 0.404289 | 0.957634 |
| 20 | — | 0.121023 | 0.919854 |
| 25 | — | 0.010000 | 0.871438 |

At epoch 12, the current detector LR is still `9.748e-5`. A schedule aligned to
a 15-epoch search candidate would be near `1.345e-5` at the same point.

**Effect:** the current run spends nearly all of its short lifetime near peak
LR. It never reaches the regularizing/settling part of cosine decay. This is
consistent with an early validation peak followed by continued destructive
updates.

### 7.4 EMA is maintained but not evaluated

The recipe enables EMA with decay `0.9998`, and the training loop updates EMA
after every optimizer step. However, validation explicitly calls `model.eval()`
and runs `outputs = model(...)` on the raw model. The summary records:

```text
evaluation_weights: raw
weight_variant: raw
```

The checkpoint selector receives only `raw_metric`; `ema_metric` is never
provided.

**Effect:** the repository pays the memory/update cost of EMA but receives none
of its potential validation/checkpoint-selection benefit. EMA often reduces the
effect of noisy or overly aggressive late updates, precisely the behavior shown
by this curve. EMA is not guaranteed to win, so both variants must be measured.

### 7.5 Early stopping is correct but too patient for this workload

The current values are:

```yaml
early_stopping_patience: 10
early_stopping_min_delta: 0.0001
```

The best mAP is epoch 2. Epochs 3–12 fail to exceed the best by `0.0001`, so the
counter reaches 10 and stops. The observed behavior is therefore exactly what
the implementation specifies.

**Effect:** best-checkpoint safety is maintained, but a short candidate may
spend most of its budget after the useful peak. A smaller patience reduces
cost; it does not itself fix the underlying generalization problem.

### 7.6 Raw loss and validation mAP optimize different objectives

The logged loss is the sum of many RT-DETR losses across decoder layers and
denoising branches. COCO mAP is sensitive to confidence order, duplicate
detections, and localization thresholds. The optimizer can make train-set loss
smaller by becoming more confident or more specialized without improving the
ranking/localization of unseen images.

**Effect:** `training_loss ↓` and `validation_mAP ↓` is possible and is the
standard empirical signature of a widening generalization gap.

## 8. Root-cause assessment

| Finding | Confidence | Severity | Evidence | Role in current behavior |
|---|---|---|---|---|
| No train-only augmentation | Confirmed | High | Same `encode_record` for train and validation | Reduces effective data diversity; encourages memorization |
| 100-epoch horizon in short run | Confirmed | High | Recipe and LR log | Keeps LR near peak; prevents late settling |
| Full backbone trainable from update 1 | Confirmed | High | No freeze policy; all trainable parameters enter optimizer | Allows catastrophic forgetting |
| EMA not evaluated/selected | Confirmed | Medium–High | Raw model used in validation; summary says raw | Discards a likely stabilizing weight variant |
| Patience 10 | Confirmed | Medium | Recipe and exact stop at epoch 12 | Wastes compute after peak; symptom control only |
| Small effective dataset for R101 | Confirmed/derived | High | 6,471 records inferred from updates | High model-capacity-to-data ratio |
| Validation sampling noise | Probable | Medium | 548 images and volatile APtiny | Explains some oscillation, not the sustained 40% drop |
| Annotation noise/occlusion | Plausible | Medium | Typical VisDrone difficulty, not audited here | Could amplify loss/metric mismatch |
| Duplicate video frames | Plausible | Medium | Dataset origin, not measured here | Could lower effective sample diversity |
| Train/validation leakage | Not established | Critical if present | No evidence in supplied log | Must be checked separately, not assumed |

## 9. Why reducing data per epoch is not the fix

Randomly using fewer training images per epoch would:

- expose the optimizer to less visual diversity;
- increase gradient and metric variance;
- make the meaning of “epoch” inconsistent across experiments;
- potentially underrepresent rare classes and tiny-object conditions;
- make comparisons with existing candidates invalid;
- hide, rather than correct, the scheduler mismatch.

The correct objective is **more effective data diversity per optimizer update**,
not fewer unique images. Use the complete training set, shuffle it, and apply
box-safe stochastic transformations.

A fixed, stratified subset is acceptable only as a clearly labeled HPO proxy to
reduce cost. All candidates must use the identical subset, and the final model
must return to the complete official training split.

If near-duplicate video frames are later confirmed, deduplicate or sample them
with a documented sequence-aware policy once. Do not randomly change the number
of images from epoch to epoch as an ad hoc regularizer.

## 10. How each configuration or implementation change affects the model

### 10.1 Add train-only box-aware augmentation

Recommended initial policy:

- horizontal flip probability: `0.5`;
- scale jitter: approximately `0.8–1.2`;
- mild brightness/contrast jitter probability: `0.2`;
- conservative box-safe crop, with minimum retained box visibility;
- no vertical flip unless an explicit aerial-orientation audit supports it;
- no stochastic transform on validation.

**Mechanism:** generates different pixels, scales, and contexts from the same
image, forcing the network to learn transferable object cues rather than exact
backgrounds or frame-specific textures.

**Expected metric behavior:** training loss may be higher and less smooth;
validation mAP should peak later or remain closer to its peak. A higher train
loss with a higher validation mAP is a successful outcome.

**Risks:** incorrect box transformation silently corrupts labels. Every
augmentation must transform boxes and labels together and be covered by visual
and geometry tests. Aggressive crop can disproportionately remove tiny objects.

**Implementation note:** augmentation keys are not currently supported by the
backend. Adding YAML keys alone will not activate augmentation; the data path in
`run_rtdetr_training.py` must be implemented and tested.

### 10.2 Align scheduler horizon with actual run length

Recommended:

- search/candidate horizon: actual candidate maximum, for example `15`;
- final-training horizon: actual final length, `25`.

**Mechanism:** retains warm-up, then gives the optimizer a meaningful decay
phase inside the run that is actually executed.

**Expected effect:** faster stabilization after early learning, smaller late
updates, reduced destruction of a good early solution.

**Risks:** if the model needs a sustained high LR to escape a poor basin,
decaying too quickly can underfit. This must be tested against the current
100-horizon control using identical seeds and data.

### 10.3 Reduce detector LR

Conservative proposed peak:

```yaml
detector_learning_rate: 0.00005
```

This halves the peak update scale for detector parameters.

**Expected effect:** slower head adaptation and less overshoot; likely smoother
validation. **Risk:** the randomly initialized two-class heads may learn too
slowly if the LR is reduced excessively. The useful range to test first is
`3e-5`, `5e-5`, and the current `1e-4` control.

### 10.4 Reduce backbone LR or freeze it initially

Proposed multiplier:

```yaml
backbone_lr_multiplier: 0.05
```

With detector LR `5e-5`, peak backbone LR becomes `2.5e-6`.

Alternative staged policy:

1. freeze backbone for epochs 1–2;
2. train the new two-class heads;
3. unfreeze backbone at low LR.

**Mechanism:** protects general pretrained features while the new output heads
become usable.

**Expected effect:** reduces catastrophic forgetting and may preserve the strong
epoch-1/2 validation result. **Risk:** VisDrone differs from COCO; freezing too
long can prevent useful aerial-domain adaptation.

**Implementation note:** staged freezing is not currently a recipe option and
requires optimizer/scheduler-aware implementation. Simply toggling
`requires_grad` after optimizer creation is insufficient unless parameter
groups and resume state are handled correctly.

### 10.5 Evaluate raw and EMA weights

Recommended validation contract:

1. evaluate raw weights;
2. evaluate EMA weights on the same validation images;
3. record both metrics;
4. select and materialize the better variant explicitly;
5. persist `weight_variant`, epoch, and metric in checkpoint identity.

**Mechanism:** EMA averages weights across updates and can suppress harmful
short-term movement.

**Expected effect:** smoother validation and possibly a later/higher best mAP.
**Risk:** EMA decay `0.9998` may adapt too slowly in a short run; lower decays
such as `0.999` should be evaluated only after raw-vs-current-EMA evidence is
available.

Evaluating both variants approximately doubles the validation portion, not the
whole epoch. Since validation is about 85 seconds of a 2,147-second steady
epoch, the additional cost is roughly 4% per epoch.

### 10.6 Shorten early-stopping patience

Proposed initial values:

```yaml
early_stopping_patience: 4
early_stopping_min_delta: 0.002
```

**Mechanism:** stops candidates that fail to improve materially for four
validations. A `0.002` delta prevents tiny metric noise from resetting patience.

**Expected effect:** substantial GPU savings on clearly deteriorating runs.
**Risk:** a curve can recover after several weak epochs, as the current run
partially did at epochs 4–6. Patience 3–5 should be compared; patience 1–2 is too
aggressive for the observed volatility.

### 10.7 Weight decay

The current performance recipe uses `0.05`, which is already meaningful AdamW
regularization. Do not change weight decay simultaneously with augmentation,
LR, scheduler, and freeze policy in the first experiment; doing so prevents
causal attribution.

If overfitting persists after schedule and augmentation fixes, compare `0.05`
with a modestly stronger value such as `0.1`. Excessive weight decay can reduce
both training fit and localization quality.

The separate Optuna-v2 HPO file contains much smaller categorical values. Those
belong to a different protocol. Results from performance-recipe and HPO-recipe
runs must not be mixed without recording the resolved configuration.

### 10.8 Dropout and attention dropout

The backend supports these model-configuration overrides, but they should be a
later experiment.

**Expected effect:** regularizes decoder representations. **Risk:** dropout can
slow convergence of newly initialized heads and may not address backbone
forgetting. Test `0.05–0.1` only after the data and schedule corrections.

### 10.9 Batch size and gradient clipping

Keep effective batch size 8 and clip norm 0.1 unchanged initially. They are
stable in the live run. Changing them with the main remediation would obscure
which change fixed the curve.

### 10.10 Image size and detection count

Keep the benchmark image size, validation preprocessing, and maximum-detection
policy fixed. Changing them can alter mAP independently of generalization and
would invalidate comparison with the existing result.

## 11. Recommended remediation sequence

### Phase 0 — preserve evidence

1. Allow the current epoch to finish before interrupting.
2. Preserve the current study DB, metrics JSONL, resolved configuration,
   predictions, and canonical best checkpoint.
3. Record the existing protocol/configuration hash.
4. Do not delete the current best epoch-2 checkpoint.
5. Do not resume an old checkpoint under a changed optimizer/scheduler policy.

### Phase 1 — implement missing measurement and data safeguards

1. Add deterministic, box-aware train augmentation with an explicit seed.
2. Keep validation preprocessing deterministic and unchanged.
3. Add tests proving transformed boxes remain valid and inside image bounds.
4. Add raw-versus-EMA validation and selection.
5. Add configuration support and resume validation for any freeze policy.
6. Include augmentation, freeze, scheduler, and EMA-selection fields in the
   configuration hash and run manifest.

### Phase 2 — minimal controlled pilot

Use the same split, seed 42, effective batch 8, image size, pretrained revision,
and evaluator. Compare:

| Pilot | Augmentation | Peak detector LR | Backbone multiplier | Horizon | Patience | Validation weights |
|---|---|---:|---:|---:|---:|---|
| Control | none | 1e-4 | 0.1 | 100 | 10 | raw |
| A | none | 1e-4 | 0.1 | actual run length | 4 | raw |
| B | train-only | 1e-4 | 0.1 | actual run length | 4 | raw + EMA |
| C | train-only | 5e-5 | 0.05 | actual run length | 4 | raw + EMA |
| D | train-only | 5e-5 | 0.05 or 2-epoch freeze | actual run length | 4 | raw + EMA |

Do not introduce all changes in one unexplained run. Pilots A–D identify which
mechanism contributes useful improvement.

### Phase 3 — robust confirmation

After selecting a candidate policy, run the repository's final seeds
`17`, `42`, and `3407`. Report:

- per-seed and mean/std mAP;
- per-seed and mean/std APtiny;
- best epoch and terminal epoch;
- raw and EMA results;
- time to best metric;
- GPU-hours;
- configuration and checkpoint hashes.

Only then should the changed policy become the default recipe.

## 12. Acceptance and stop criteria

### 12.1 Pilot success criteria

A corrected pilot should satisfy all of the following:

1. No NaN/Inf, OOM, environment, or adapter failure.
2. Best validation mAP exceeds the current `0.27883` baseline or achieves the
   same result with materially lower compute.
3. APtiny does not regress materially from `0.09115`.
4. Terminal mAP is at least 90% of the run's peak, or early stopping terminates
   promptly after a confirmed decline.
5. The best epoch is not systematically epoch 1–2 across corrected runs.
6. Raw/EMA selection is explicit and reproducible.

### 12.2 Reasons to stop a live run immediately

- non-finite loss or metric;
- repeated CUDA OOM;
- corrupted/missing checkpoint state;
- invalid transformed boxes;
- a sustained validation decline that has already reached the revised patience;
- accidental use of validation augmentation;
- resolved configuration does not match the intended protocol.

Ordinary epoch-to-epoch mAP noise is not by itself a reason to terminate before
the configured policy makes a decision.

## 13. Additional dataset audits before final publication

These checks are not prerequisites for the first remediation pilot, but they are
required before making strong scientific claims:

1. Count PERSON and VEHICLE instances per split after category collapse.
2. Plot object-area distributions for train and validation.
3. Measure ignored/truncated/occluded object rates by split.
4. Hash images and detect exact duplicates across train and validation.
5. Use perceptual similarity or sequence metadata to identify near-duplicate
   video frames.
6. Visually audit a stratified sample of tiny-object boxes.
7. Compare false positives/negatives at the current best epoch and a degraded
   late epoch.
8. Verify that category IDs are contiguous and the two-class mapping is
   identical in train, validation, prediction conversion, and evaluation.

## 14. Proposed conservative recipe target

This is a starting hypothesis, not a validated replacement:

```yaml
recipe_version: rtdetr_recipe_v3_candidate
detector_learning_rate: 0.00005
backbone_lr_multiplier: 0.05
weight_decay: 0.05
gradient_clip_norm: 0.1
warmup_epochs: 2
warmup_start_factor: 0.01
minimum_lr_factor: 0.01
scheduler_horizon_epochs: "match_actual_run"
ema_enabled: true
ema_decay: 0.9998
evaluate_raw_and_ema: true
effective_batch_size: 8
early_stopping_patience: 4
early_stopping_min_delta: 0.002
augmentation:
  horizontal_flip_probability: 0.5
  scale_range: [0.8, 1.2]
  brightness_contrast_probability: 0.2
  box_safe_crop: true
```

The string and augmentation fields above are design-level proposals. The
current backend does not support them as-is. They must not be copied into the
current YAML and assumed to work without implementation and contract tests.

## 15. Final recommendation

Do not reduce the number of images per epoch and do not continue spending GPU
hours on the unchanged recipe merely because training loss is falling.

Preserve the current best checkpoint and study evidence, then create a new
versioned training protocol that:

1. uses the complete dataset with train-only box-aware augmentation;
2. aligns LR decay with the actual search/final run length;
3. protects the pretrained backbone through lower LR or short staged freezing;
4. validates and selects between raw and EMA weights;
5. uses shorter, materially thresholded early stopping;
6. confirms the result across seeds 17, 42, and 3407.

The current run is valuable: it is the first live evidence that the environment,
adapter, training loop, evaluator, and checkpoint controls work together on an
L4 GPU. It also exposes a model-quality gap that static/CPU tests could not
detect. The next step is a versioned empirical recipe correction, not data
reduction.

## 16. Repository evidence map

- Model identity and pretrained source:
  [`configs/rtdetrv2_l/model.yaml`](../../configs/rtdetrv2_l/model.yaml)
- Current performance recipe:
  [`configs/rtdetrv2_l/performance_recipe_v2.yaml`](../../configs/rtdetrv2_l/performance_recipe_v2.yaml)
- Separate Optuna search space:
  [`configs/rtdetrv2_l/hpo_recipe_v2.yaml`](../../configs/rtdetrv2_l/hpo_recipe_v2.yaml)
- RT-DETR training/data/validation implementation:
  [`scripts/run_rtdetr_training.py`](../../scripts/run_rtdetr_training.py)
- Update-based warm-up/cosine scheduler:
  [`src/models/rtdetrv2/scheduler.py`](../../src/models/rtdetrv2/scheduler.py)
- Differential optimizer groups:
  [`src/models/rtdetrv2/optimizer.py`](../../src/models/rtdetrv2/optimizer.py)
- EMA implementation:
  [`src/training/ema.py`](../../src/training/ema.py)
- Early stopping:
  [`src/training/early_stopping.py`](../../src/training/early_stopping.py)
- Checkpoint selection:
  [`src/training/checkpoint_selection.py`](../../src/training/checkpoint_selection.py)
- LR-search settings and promotion policy:
  [`src/training/lr_search.py`](../../src/training/lr_search.py)
- LR workflow orchestration:
  [`src/training/lr_workflow.py`](../../src/training/lr_workflow.py)
