# Run LR search, fine-tune, and upload results

This is the main run guide for the one-day-per-model VisDrone 2-class
benchmark. Use one persistent Google Drive root:

```text
/content/drive/MyDrive/visdrone_architecture_benchmark
```

# Quick start

1. Run
   [notebook 00](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/00_visdrone_colab_setup.ipynb)
   once to prepare and validate VisDrone.
2. Pick one `MODEL_ID`: `faster_rcnn_resnet50`,
   `faster_rcnn_swin_t`, `faster_rcnn_vmamba_t`, or `rtdetrv2_l`.
3. Open
   [notebook 12](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/12_learning_rate_search.ipynb),
   set `MODEL_ID` and `START_EXPENSIVE_STAGE=True`, then run all cells.
4. Confirm
   `$DRIVE_ROOT/lr_search_configs/MODEL_ID_2class_selected.yaml` exists.
5. Open
   [notebook 13](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/13_full_dataset_finetune.ipynb),
   set the same `MODEL_ID` and `START_EXPENSIVE_STAGE=True`, then run all cells.
   The selected YAML and any compatible interrupted run are found automatically.
6. Run
   [notebook 07](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/07_evaluate_all_models.ipynb)
   in the same model environment.
7. Run
   [notebook 10](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/10_generate_final_report.ipynb).
8. Run
   [notebook 11](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/11_sync_results_to_github.ipynb)
   with `PUBLISH_RESULTS=False` and `DRY_RUN=True`.
9. Review the bundle contents, validation result, size, destinations, and Git
   diff preview.
10. Authenticate safely, set `PUBLISH_RESULTS=True` and `DRY_RUN=False`, and
    rerun the publishing cell to update `experiment-results` and open a PR to
    `main`.

When unsure, run:

```bash
python -m scripts.show_benchmark_status --drive-root "$DRIVE_ROOT"
python -m scripts.next_benchmark_step \
  --drive-root "$DRIVE_ROOT" \
  --model-id rtdetrv2_l
```

Both commands are read-only.

## The model-day workflow

Day 0:

1. Prepare VisDrone.
2. Validate both the converted data and deterministic search manifests.

Day N, for exactly one primary model:

1. Validate the model environment.
2. Run or resume LR search.
3. Export the selected LR.
4. Restart from the model's original pretrained weights and fine-tune for 25
   epochs on the complete official train split.
5. Evaluate the final checkpoint on official validation.
6. Generate the report.
7. Create and validate one lightweight result bundle.
8. Preview Git changes.
9. Publish to `experiment-results` and open a PR to `main`.

Repeat for:

```text
faster_rcnn_resnet50
faster_rcnn_swin_t
faster_rcnn_vmamba_t
rtdetrv2_l
```

YOLOX is not part of this controlled workflow.

## Which notebook do I open?

| Step | Notebook and direct link | Runtime | What to edit | Expensive part | Success and persistent output | Resume |
|---|---|---|---|---|---|---|
| 0 | [`00_visdrone_colab_setup.ipynb`](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/00_visdrone_colab_setup.ipynb) | Hosted Colab; CPU is sufficient | Dataset source, tracks, and storage in its configuration cell | Download, extraction, and conversion | Valid COCO files under `$DRIVE_ROOT/datasets/VisDrone2019-DET/processed/` and validated manifests | Rerun all cells; valid archives, extraction, and conversions are reused |
| 1 | [`12_learning_rate_search.ipynb`](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/12_learning_rate_search.ipynb) | RT-DETR: hosted GPU. Faster R-CNN: documented Python 3.10 custom/local GPU runtime | `MODEL_ID`, `START_EXPENSIVE_STAGE`, range-test/boundary switches, and optional batch controls | Runtime calibration, optional 300-step LR range test, and 2/5/10/15-epoch successive halving | Ends with `LR SEARCH COMPLETE`; selected YAML and search summary appear in `$DRIVE_ROOT/lr_search_configs/` | Reopen with the same `MODEL_ID`, seed 42, and Drive root; completed rungs/candidates are skipped |
| 2 | [`13_full_dataset_finetune.ipynb`](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/13_full_dataset_finetune.ipynb) | Same compatible GPU environment as notebook 12 | `MODEL_ID`, `START_EXPENSIVE_STAGE`, and optional batch controls | 25 final epochs plus common evaluation | Prints `FULL OFFICIAL TRAINING SPLIT VERIFIED: YES` and `FINAL TRAINING COMPLETE`; outputs are under `$DRIVE_ROOT/checkpoints/final/MODEL_ID/RUN_ID/` | Latest incomplete run is auto-resumed only when model, track, LR, seed, size, horizon, and run kind match |
| 3 | [`07_evaluate_all_models.ipynb`](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/07_evaluate_all_models.ipynb) | Same model GPU environment | `MODEL_ID`; set `RUN_ID` only if the displayed table is ambiguous | Official-validation inference and profiling | Prints `RESULTS READY FOR REVIEW`; metrics appear in `$DRIVE_ROOT/evaluation/` | Rerun discovery and evaluation; deterministic output paths are replaced |
| 4 | [`10_generate_final_report.ipynb`](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/10_generate_final_report.ipynb) | CPU is sufficient | `MODEL_ID` | Report tables and figures | Prints `RESULTS READY FOR REVIEW`; outputs appear in `$DRIVE_ROOT/reports/` | Rerun the report cell |
| 5 | [`11_sync_results_to_github.ipynb`](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/11_sync_results_to_github.ipynb) | CPU is sufficient | `MODEL_ID`; normally leave `RUN_ID` and `RESULT_BUNDLE_ID` blank | Bundle copying, Git push, and PR creation only after explicit opt-in | Valid bundle under `$DRIVE_ROOT/result_bundles/`; approved copy under `results/bundles/`; PR URL | Rerun discovery, validation, or dry-run. An existing valid bundle for the same run is reused |

Notebook 01 is optional analysis:
[open `01_dataset_analysis.ipynb`](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/01_dataset_analysis.ipynb).

## Model environments

Do not install all model families into one runtime.

### RT-DETRv2-L

- Recommended runtime: current hosted Colab with an NVIDIA GPU.
- Requirements: `requirements-rtdetr-colab.txt`.
- Exact configured checkpoint ID: `PekingU/rtdetr_v2_r101vd`.
- Validate before training:

```bash
python -c "import torch, transformers; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU', transformers.__version__)"
python -m scripts.verify_model_environments --environment rtdetr
```

If installation changes core packages, use **Runtime → Restart session**, then
rerun the notebook from the top. A Hugging Face download is cached by the
runtime; Drive checkpoints and experiment state remain persistent.

### Faster R-CNN ResNet-50 and Swin-T

The verified repository contract is:

- Python 3.10
- PyTorch 2.1.0 with CUDA 11.8
- torchvision 0.16.0
- NumPy 1.26.4
- MMCV 2.1.0
- MMEngine 0.10.7
- MMDetection 3.3.0

The current hosted Python 3.12/PyTorch 2.10 Colab runtime is not a verified
MMCV environment. Use a Colab local/custom runtime with the versions above.
In that runtime:

```bash
python -m pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
python -m pip install -r requirements-openmmlab-py310-cu118.txt
git clone --depth 1 --branch v3.3.0 \
  https://github.com/open-mmlab/mmdetection.git /content/mmdetection
export MMDET_ROOT="/content/mmdetection"
python -m scripts.verify_model_environments --environment openmmlab
python -c "import torch, mmcv, mmengine, mmdet; print(torch.__version__, torch.version.cuda, mmcv.__version__, mmengine.__version__, mmdet.__version__)"
```

Notebook 12 performs the version preflight and refuses an unsupported hosted
runtime instead of silently changing the benchmark environment.

### VMamba-T

VMamba needs the same OpenMMLab runtime plus:

- VMamba commit `2ed52ead062a51a64521ed3871d52914bf532876`;
- the compiled selective-scan extension;
- `VMAMBA_ROOT`;
- a verified original VMamba-T pretrained checkpoint at
  `$DRIVE_ROOT/pretrained/vmamba_t.pth`, exposed as
  `VMAMBA_T_PRETRAINED`.

```bash
git clone https://github.com/MzeroMiko/VMamba.git /content/VMamba
git -C /content/VMamba checkout 2ed52ead062a51a64521ed3871d52914bf532876
export VMAMBA_ROOT="/content/VMamba"
export VMAMBA_T_PRETRAINED="$DRIVE_ROOT/pretrained/vmamba_t.pth"
python -m pip install /content/VMamba/kernels/selective_scan --no-build-isolation
python -c "import selective_scan_cuda; print('selective_scan_cuda: OK')"
test -f "$VMAMBA_T_PRETRAINED"
python -m scripts.verify_model_environments --environment openmmlab
```

Notebook 12 uses the canonical Drive checkpoint path and does not ask you to
copy a checkpoint path between notebooks.

## Preflight and completion messages

Before search or final training, the notebook prints model, track, manifests,
image counts, full-train status, image/batch settings, LR, epoch budget, GPU,
runtime estimate, output path, resume status, and Git commit.

Do not start final training unless this exact line appears:

```text
FULL OFFICIAL TRAINING SPLIT VERIFIED: YES
```

The code compares final-train image IDs with the official train source and
also proves that official validation IDs are disjoint. It fails before model
construction on a mismatch.

Notebook 12 ends with `LR SEARCH COMPLETE`. Notebook 13 ends with
`FINAL TRAINING COMPLETE`. Notebooks 07 and 10 end with
`RESULTS READY FOR REVIEW`. Their summaries contain exact Drive paths and the
next notebook.

## Resume after a disconnect

### Search interrupted

Reopen notebook 12 and keep the same `MODEL_ID`, seed 42, and `DRIVE_ROOT`.
The search state is:

```text
$DRIVE_ROOT/checkpoints/lr_search/MODEL_ID/search_state.json
```

Candidate directories contain their own `last.pth` and epoch metrics. The
workflow reads completed rung decisions, promoted candidate IDs, and each
candidate's last completed epoch. It skips completed rungs and runs only
missing work. Do not rename candidate directories or change the LR grid.

Check before resuming:

```bash
python -m scripts.show_benchmark_status \
  --drive-root "$DRIVE_ROOT" \
  --model-id "$MODEL_ID"
```

### Final training interrupted

Notebook 13 automatically finds the latest incomplete run under:

```text
$DRIVE_ROOT/checkpoints/final/MODEL_ID/RUN_ID/
```

It resumes only if `last.pth` exists and `training_config.yaml` matches the
selected LR, model, 2-class track, 640 size, seed 42, 25-epoch scheduler
horizon, and final run kind. Otherwise it starts a new compatible run. The
resolved run ID is printed in the preflight; no manual `RESUME_RUN_ID` is
normally needed.

### Evaluation, reporting, or publishing interrupted

Reopen the relevant notebook and rerun its discovery and action cells.
Evaluation metrics, reports, validation, and dry-run export use deterministic
or versioned paths and are safe to rerun. Notebook 11 reuses an existing valid
bundle for the same run. Git publishing still requires explicit opt-in.

## Lightweight result bundle

One bundle contains one model, one 2-class final run, and one seed:

```text
$DRIVE_ROOT/result_bundles/MODEL_ID__2class__YYYYMMDD_HHMMSS/
├── bundle_manifest.json
├── README.md
├── configs/
│   ├── selected_lr.yaml
│   └── final_resolved_config.yaml
├── search/
│   ├── candidates.csv
│   ├── promotion_history.csv
│   └── search_summary.json
├── metrics/
│   ├── final_metrics.json
│   ├── per_class_metrics.csv
│   └── profiling_summary.json
├── reports/
│   ├── model_report.md
│   └── figures/
└── provenance/
    ├── environment_summary.json
    ├── dataset_hashes.json
    └── git_commit.txt
```

Included files are resolved/selected configurations, search candidates and
promotion history, measured final/per-class/performance metrics, parameter and
runtime summaries, environment and dataset hashes, commit provenance, small
figures, report, and manifest.

Excluded files include images and archives, converted datasets, checkpoints,
optimizer/scheduler states, raw predictions, TensorBoard logs, profiler traces,
credentials, tokens, Drive metadata, compiled extensions, and framework source
trees. The validator rejects forbidden extensions/directories, files over
20 MB, secrets, private absolute paths, inconsistent identity/LR, invalid
formats, unproved dataset identity, missing provenance, missing metrics, and
placeholder values.

Validate directly:

```bash
python -m scripts.validate_results --bundle-path "$BUNDLE_PATH"
```

## Dry-run and publishing

Notebook 11 defaults to:

```python
PUBLISH_RESULTS = False
DRY_RUN = True
```

Dry-run prints files copied/excluded, count, total size, target branch,
repository destinations, validation result, and projected diff. It does not
write repository files, stage, commit, push, or open a PR.

For publishing, authenticate without placing a token in a visible cell:

```bash
gh auth login --web
gh auth status
```

GitHub CLI device/browser login is preferred. A Colab Secret exposed as an
environment variable is acceptable; never print it or store it in Drive,
Git config, the repository, or a bundle.

After explicit opt-in, notebook 11:

1. requires a clean clone;
2. checks `gh auth status`;
3. fetches origin;
4. creates or refreshes `experiment-results`;
5. copies only the validated bundle under `results/bundles/`;
6. validates `results/`;
7. stages only `results/`;
8. shows the staged stat and names;
9. commits `results(MODEL_ID): add 2-class LR benchmark results`;
10. pushes `experiment-results`;
11. opens or reports the PR to `main`.

No checkpoint is copied into Git.

## Manual command fallback

Run these from the repository root in the correct model environment:

```bash
export DRIVE_ROOT="/content/drive/MyDrive/visdrone_architecture_benchmark"
export MODEL_ID="rtdetrv2_l"

python -m scripts.show_benchmark_status --drive-root "$DRIVE_ROOT"
python -m scripts.next_benchmark_step \
  --drive-root "$DRIVE_ROOT" --model-id "$MODEL_ID"

python scripts/lr_search.py \
  --drive-root "$DRIVE_ROOT" \
  --model-id "$MODEL_ID" \
  --batch-size 2 \
  --accumulation 4 \
  --start-expensive-stage

python scripts/full_dataset_finetune.py \
  --drive-root "$DRIVE_ROOT" \
  --model-id "$MODEL_ID" \
  --batch-size 2 \
  --accumulation 4 \
  --start-expensive-stage
```

The final-training command discovers the selected YAML automatically. Evaluate
a run ID printed by notebook 13 or the status command:

```bash
export RUN_ID="MODEL_ID__2class__640__YYYYMMDD_HHMMSS__seed42"

python scripts/evaluate.py \
  --drive-root "$DRIVE_ROOT" \
  --dataset-track 2class \
  --split val \
  --run-id "$RUN_ID" \
  --resolutions 640

python scripts/generate_report.py --drive-root "$DRIVE_ROOT"

export BUNDLE_ID="${MODEL_ID}__2class__YYYYMMDD_HHMMSS"
python scripts/create_results_manifest.py \
  --drive-root "$DRIVE_ROOT" \
  --dataset-track 2class \
  --model-id "$MODEL_ID" \
  --run-id "$RUN_ID" \
  --bundle-id "$BUNDLE_ID"

export BUNDLE_PATH="$DRIVE_ROOT/result_bundles/$BUNDLE_ID"
python -m scripts.validate_results --bundle-path "$BUNDLE_PATH"

python scripts/sync_results_to_repo.py \
  --drive-root "$DRIVE_ROOT" \
  --bundle-id "$BUNDLE_ID" \
  --repo-root . \
  --validate \
  --dry-run
```

After reviewing the dry-run, the advanced manual Git equivalent is:

```bash
gh auth status
git status --short
git fetch origin
git switch -C experiment-results origin/experiment-results
python scripts/sync_results_to_repo.py \
  --drive-root "$DRIVE_ROOT" \
  --bundle-id "$BUNDLE_ID" \
  --repo-root . \
  --validate
python -m scripts.validate_results --repo-results results
git add -- results
git diff --cached --stat
git diff --cached --name-status
git commit -m "results($MODEL_ID): add 2-class LR benchmark results"
git push -u origin experiment-results
gh pr create \
  --base main \
  --head experiment-results \
  --title "Results: $MODEL_ID VisDrone 2-class LR benchmark" \
  --body "Validated lightweight bundle: results/bundles/$BUNDLE_ID"
```

If `origin/experiment-results` does not exist yet, create it from main with:

```bash
git switch -C experiment-results origin/main
```

## Troubleshooting

| Problem | Symptom | Exact check | Safe correction |
|---|---|---|---|
| Drive not mounted | `/content/drive` is missing or writes fail | `ls /content/drive/MyDrive` | Rerun the mount cell and approve the prompt; do not change the persistent root |
| Dataset archive missing | Notebook 00 cannot find a valid ZIP | `find "$DRIVE_ROOT/datasets/VisDrone2019-DET/archives" -maxdepth 1 -type f` | Rerun acquisition or upload the official archive through notebook 00's manual upload path |
| Invalid COCO conversion | Dataset preflight reports category/image/annotation errors | `python -m scripts.prepare_data --drive-root "$DRIVE_ROOT" --tracks 2class --validate` | Rerun conversion from unchanged verified raw data; preserve the raw directory |
| GPU unavailable | Preflight prints `NOT DETECTED` | `python -c "import torch; print(torch.cuda.is_available())"` | Select an NVIDIA GPU runtime or reconnect the custom GPU runtime |
| CUDA out of memory | Candidate/run records `FAILED_OOM` | `nvidia-smi` | Lower per-device batch and increase accumulation so effective batch remains exactly 8; rerun the same notebook |
| MMCV import failure | `mmcv`/custom-op import error | `python -c "import torch, mmcv; print(torch.__version__, torch.version.cuda, mmcv.__version__)"` | Use Python 3.10, Torch 2.1+cu118, and the matching MMCV 2.1 wheel; restart once |
| VMamba selective scan failure | `import selective_scan_cuda` fails | `python -c "import selective_scan_cuda"` | Rebuild from the pinned VMamba commit with `--no-build-isolation`; verify CUDA compiler compatibility |
| Hugging Face download failure | RT-DETR checkpoint download times out or is unauthorized | `python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('PekingU/rtdetr_v2_r101vd'))"` | Reconnect, verify Hugging Face access, and rerun; do not substitute another checkpoint ID |
| Runtime restart | Imports disappear after pip asks for restart | `python -c "import sys; print(sys.version)"` | Restart once, remount Drive, and rerun notebook cells from the top |
| Search resume mismatch | A new search starts or state is rejected | `python -m scripts.show_benchmark_status --drive-root "$DRIVE_ROOT" --model-id "$MODEL_ID"` | Restore the same model, seed 42, Drive root, and fixed controls; do not rename search directories |
| Final run is not full train | Required verification line is absent | `python -c "from src.training.lr_search import assert_final_training_uses_official_train; assert_final_training_uses_official_train('$DRIVE_ROOT/datasets/VisDrone2019-DET/manifests/lr_search')"` | Stop; regenerate/validate manifests through notebook 12 before training |
| Selected LR YAML missing | Notebook 13 says to finish notebook 12 | `test -f "$DRIVE_ROOT/lr_search_configs/${MODEL_ID}_2class_selected.yaml"` | Resume notebook 12 until `LR SEARCH COMPLETE` |
| Registry missing run | Evaluation finds no compatible completed run | `python -m scripts.show_benchmark_status --drive-root "$DRIVE_ROOT" --model-id "$MODEL_ID"` | Finish/resume notebook 13; do not hand-edit the registry |
| Evaluation finds no checkpoint | Registry run exists but checkpoint path is absent | Inspect `checkpoint_best_map` in `$DRIVE_ROOT/experiment_registry/checkpoint_registry.json` | Remount the same Drive and resume final training; do not point evaluation at a search checkpoint |
| Bundle validation failure | Validator exits nonzero with a list | `python -m scripts.validate_results --bundle-path "$BUNDLE_PATH"` | Correct only the listed source artifact and recreate a new versioned bundle; never bypass validation |
| Git authentication failure | `gh auth status` fails | `gh auth status` | Run `gh auth login --web` or reconnect the protected secret; never paste/print a token |
| Remote branch changed | Push is rejected as non-fast-forward | `git fetch origin && git log --oneline --left-right HEAD...origin/experiment-results` | Preserve local work, update from the remote branch, revalidate/stage, and push; never force-push shared results |

## Before starting training

- [ ] Dataset setup completed
- [ ] Correct runtime selected
- [ ] GPU detected
- [ ] Model environment smoke test passed
- [ ] Drive is writable
- [ ] Search manifests validated
- [ ] Workload estimate reviewed

## Before final fine-tuning

- [ ] LR search completed
- [ ] Selected YAML exists
- [ ] Selected LR is recorded
- [ ] Full official train manifest verified
- [ ] Official validation is excluded from training
- [ ] Original pretrained checkpoint is available

## Before uploading results

- [ ] Final run completed
- [ ] Evaluation completed
- [ ] Report generated
- [ ] Bundle validated
- [ ] No checkpoints or datasets included
- [ ] No secrets detected
- [ ] Dry-run diff reviewed
- [ ] Branch is `experiment-results`
