# Legacy RT-DETRv2 R50/L training observation

**Observation date:** 1 August 2026

**Source:** researcher-provided `04_train_rtdetrv2_l.ipynb` snapshot

**Status:** diagnostic evidence only; not a canonical benchmark result

## Scope and project compatibility

The notebook records a seed-42, two-class VisDrone fine-tuning run using
`PekingU/rtdetr_v2_r50vd`. It is useful for diagnosing training behavior, but it
does not belong to either maintained project protocol and must not be added to a
published result bundle.

| Contract item | Observed notebook | Maintained project |
|---|---|---|
| Architecture | RT-DETRv2 R50/L (`r50vd`) | `rtdetrv2_l`: RT-DETRv2-L (`r50vd`) — same architecture |
| Workflow | External notebook 04 | Notebooks 13, 23, 30, and 31 |
| Batch contract | Batch size 2 | Effective batch size 8 for `lr_controlled_v1` |
| Final seeds | 42 only | 17, 42, and 3407 for `two_stage_random_hpo_v1` |
| Selection data | Official validation evaluated every epoch | Search validation is a deterministic subset of train; official validation does not tune |
| Environment | Python 3.12.13, PyTorch 2.11.0+cu128 | Pinned RT-DETR environment in `configs/runtime_environments.yaml` |
| Checkpoint names | `best.pt`, `latest.pt` | Canonical `.pth` checkpoint contract |

The legacy run uses the same RT-DETRv2-L (`r50vd`) architecture that the
repository now trains, so it is architecturally comparable. It remains
**methodologically** incompatible: the batch contract, seed count, and
selection-data policy differ, which affects optimization, selection bias, and
reproducibility. The observed AP values are therefore still not comparable with
canonical repository results. The run should be identified as
`legacy_rtdetrv2_r50vd_2class_seed42` in discussion or notes.

## Run status

The notebook configured 50 epochs at 640 px with AdamW, weight decay `1e-4`, a
cosine scheduler, batch size 2, horizontal-flip probability 0.5, and selected LR
`4.128205343826226e-5`. All 42,731,082 parameters were trainable. Classification
and score heads were newly initialized because the 80-class pretrained shapes
did not match the two-class target.

Epochs 0-39 completed. The saved execution ended with a manual
`KeyboardInterrupt` during epoch 40 at batch 1,092 of 3,236. This was not CUDA
OOM or numerical divergence. Under that notebook's own checkpoint contract, a
matching Drive run can resume from epoch 40 because epoch 39 was saved to
`latest.pt`. The validation-selected checkpoint is `best.pt` from epoch 0.

## Evaluation

| Metric | Epoch 0 (best) | Epoch 39 (latest complete) | Change |
|---|---:|---:|---:|
| mAP@[0.50:0.95] | 0.211 | 0.127 | -0.084 (-39.7%) |
| AP50 | 0.458 | 0.312 | -0.146 (-31.9%) |
| AP75 | 0.167 | 0.083 | -0.084 (-50.3%) |
| AP small | 0.184 | 0.116 | -0.068 (-37.0%) |
| AP medium | 0.654 | 0.480 | -0.174 (-26.6%) |
| AP large | 0.903 | 0.515 | -0.388 (-43.0%) |
| AR, maxDets=100 | 0.346 | 0.353 | +0.007 (+2.0%) |
| PERSON AP | 0.134 | 0.078 | -0.056 (-41.8%) |
| VEHICLE AP | 0.288 | 0.176 | -0.112 (-38.9%) |

No epoch after epoch 0 exceeded 0.1712 mAP. Mean mAP across epochs 1-39 was
0.1382, and the minimum was 0.1090 at epoch 21. Mean mAP for the last ten
completed epochs was 0.1336, compared with 0.1563 for the first ten. This is a
persistent decline rather than a single noisy measurement.

Both classes deteriorated. Meanwhile, aggregate recall remained nearly flat
(0.346 to 0.353), so the later model continued to retrieve a similar fraction
of objects but ranked or localized them less effectively. The 50.3% AP75 decline
is particularly consistent with worsening box localization. Large-object AP is
based on fewer examples and should not be interpreted independently of support
counts.

## Diagnosis

The trajectory is strong evidence of validation degradation and is consistent
with over-fine-tuning. Likely contributors are:

1. The first completed epoch was already the validation optimum, but training
   continued for 39 more complete epochs.
2. All 42.7 million parameters were updated for a two-class task while several
   detection heads began from random initialization. Jointly updating the new
   heads and the pretrained feature extractor can cause early catastrophic
   forgetting.
3. Horizontal flipping was the only explicit augmentation. It does not cover
   VisDrone's large changes in scale, altitude, illumination, density, and
   occlusion.
4. The LR was selected through a three-epoch search but used for a 50-epoch final
   schedule. This favors short-lived early performance instead of long-horizon
   stability.
5. Small objects dominate the task and AP small fell from 0.184 to 0.116. At
   640 px, weak multiscale and crop augmentation can encourage scene-specific
   features.
6. The two collapsed classes combine categories with different frequencies and
   appearances. Unmeasured class and scene imbalance may reinforce training-set
   specialization.

Classic overfitting is only confirmed when training loss keeps improving while
held-out performance worsens. The notebook writes `total_loss` and component
losses to its Drive `history.csv` and `history.json`, but those values are absent
from the saved cell output. Until that history is inspected, the precise finding
is **severe validation degradation consistent with overfitting or catastrophic
forgetting**, not proof of overfitting alone. If loss rises or becomes unstable,
optimization instability is the better diagnosis.

Repeated checkpoint selection on official validation also makes epoch-0 AP a
selection metric, not an unbiased final evaluation. The maintained project
avoids this by tuning on a deterministic train subset and reserving official
validation for final evaluation.

## Disposition

- Preserve this document as diagnostic provenance only.
- Do not copy its metrics into `results/`, a comparison table, or a publication.
- For the legacy run itself, prefer epoch-0 `best.pt` over epoch-39 `latest.pt`.
- Do not spend more GPU time completing the legacy schedule unless a fixed-budget
  reproduction explicitly requires it.
- Run the canonical RT-DETRv2 path through notebooks 13 and 23, then evaluate
  through notebook 30.
- For a corrective experiment outside the locked benchmark, test shorter
  schedules, early stopping with patience 3-5, head warm-up or gradual
  unfreezing, a lower backbone LR, and stronger detection-safe augmentation.
- Plot train loss against search-validation mAP before assigning the final
  overfitting label.

## Assessment

The external run is resumable, but it has passed its useful validation point.
Its mAP declined from 0.211 at epoch 0 to 0.127 at epoch 39. That finding is
valuable for refining the training recipe, but the run is methodologically
incompatible with the maintained benchmark (batch contract, seed count, and
selection-data policy) and cannot serve as the project's RT-DETRv2-L result,
even though it shares the R50/L architecture.
