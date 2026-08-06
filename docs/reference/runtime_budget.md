# Runtime budget

**Status: `t_iter` is not measured yet.** No seconds-per-iteration measurement
exists in this repository. Run `scripts/measure_throughput.py` on the target GPU
(Colab/T4/L4) to produce `throughput.json`; that script regenerates this file
with measured hours. Until then the per-model hour columns are `null` — a guessed
runtime is never presented as measured.

## What is known without a GPU: iteration counts

The controlled protocol uses batch size 1, so iterations equal images. These
counts come directly from `configs/controlled/benchmark.yaml` and the split
fractions in `src/training/lr_search.py` (20% search subset, 5% selection
holdout on the official 6,471-image VisDrone train split):

| Quantity | Value |
|---|---:|
| Search subset images | 1,294 |
| Final-train images (train − 5% holdout) | 6,147 |
| HPO iterations (5 trials × (3+3) epochs × 1,294) | 38,820 |
| Headline final iterations (1 recipe × 1 seed × 8 × 6,147) | 49,176 |
| Full-matrix final iterations (2 recipes × 3 seeds × 8 × 6,147) | 295,056 |

## Hours require a measured `t_iter`

`hours = (hpo_iterations + final_iterations) × t_iter / 3600`. The table below is
**illustrative only** — it shows what the budget would be at three plausible
`t_iter` values so the feasibility question is concrete. These are **not
measurements** and must not be reported as results.

| `t_iter` (illustrative) | HPO h | Headline total h | Full-matrix total h |
|---|---:|---:|---:|
| 0.15 s (best case, T4/L4, R50) | 1.62 | **3.67** | 13.91 |
| 0.30 s (realistic) | 3.23 | **7.33** | 27.82 |
| 0.60 s (worst, Swin/VMamba/RT-DETR) | 6.47 | **14.67** | 55.65 |

## Reading the feasibility claim

- The **headline** matrix (1 tuned recipe, seed 42) plausibly fits one model per
  GPU-day: ~4–15 h/model across the illustrative range, within a single Colab
  session at the faster end.
- The **full opt-in matrix** (baseline+tuned × 3 seeds) does **not** fit one
  model per day at realistic `t_iter`; it is a multi-session, multi-day job and
  is intended only as separately-reported multi-seed validation.
- These conclusions must be reconfirmed against the measured `t_iter` from
  `scripts/measure_throughput.py`. If the measured numbers contradict the
  one-model-per-day target, the claim is withdrawn, not the measurement.

The probe measures training-representative iterations with the device
synchronized around the timed window and warm-up iterations excluded; it records
the measurement mode, peak GPU memory, GPU name, and torch/CUDA versions in
`throughput.json`.
