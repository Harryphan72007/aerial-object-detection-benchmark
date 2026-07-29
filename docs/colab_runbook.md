# Google Colab runbook

## 1. Runtime to choose

For dataset setup and analysis, choose the current hosted Colab runtime. A GPU
is optional. For RT-DETRv2 training, choose an NVIDIA GPU runtime. For
MMDetection or VMamba, use a Colab local/custom runtime with Python 3.10,
PyTorch 2.1.0+cu118, and the versions in
`requirements-openmmlab-py310-cu118.txt`; the current hosted Python
3.12/PyTorch 2.10 runtime is not a verified MMCV environment.

Do not install all model families into one runtime.

## 2. Notebook to open first

Open `notebooks/00_visdrone_colab_setup.ipynb` and run every cell in order.
The older `00_colab_repository_setup.ipynb` and
`00_environment_and_data_setup.ipynb` are compatibility preflights, not the
primary setup.

## 3. Cells requiring interaction

1. **User configuration:** choose repository branch, Drive/local storage,
   source (`auto`, `manual`, or `kaggle`), redownload behavior, tracks, smoke
   size, and seed.
2. **Google Drive mount:** approve the normal Colab Drive prompt when
   `USE_GOOGLE_DRIVE=True`.
3. **Manual archive upload:** appears only when the validated cache and
   token-free source fail, or when `DATASET_SOURCE="manual"`.
4. **Kaggle:** use only when you explicitly selected a verified handle. Set
   `KAGGLE_DATASET_HANDLE`; configure authentication outside the notebook. Never
   paste or print credentials.

## 4. Persistent outputs

```text
/content/drive/MyDrive/visdrone_architecture_benchmark/
├── datasets/VisDrone2019-DET/
│   ├── archives/
│   ├── raw/
│   ├── processed/
│   │   ├── coco_2class/
│   │   └── coco_10class/
│   └── manifests/
├── checkpoints/lr_search/MODEL_ID/CANDIDATE_ID/
├── checkpoints/final/MODEL_ID/RUN_ID/
├── experiment_registry/checkpoint_registry.json
├── experiment_registry/runs.csv
├── lr_search_configs/
├── evaluation/
├── reports/
└── runs/
```

Raw archives, extracted data, checkpoints, tokens, and generated caches are
ignored by Git.

## 5. Resume after a disconnect

Rerun notebook 00. It reuses a valid archive only when ZIP structure/CRC and
the recorded SHA-256 agree, and it skips valid extracted split directories.
Conversion is deterministic and safely overwrites generated JSON; raw files
remain unchanged.

For training, reopen the same model notebook and set `RESUME_RUN_ID` to the run
ID recorded in `experiment_registry/checkpoint_registry.json`. Run IDs retain:

```text
MODEL_ID__DATASET_TRACK__RESOLUTION__TIMESTAMP__SEED
```

## 6. Small smoke test

In notebook 00's configuration cell set:

```python
SMOKE_TEST = True
SMOKE_TEST_SUBSET_SIZE = 8
TRACKS = ("2class", "10class")
```

For repository verification, the automated equivalent is:

```bash
SMOKE_TEST=1 VISDRONE_DRIVE_ROOT=.notebook-smoke \
python -m scripts.run_notebook_smoke --kernel python3
```

Training notebooks validate their command, dataset track, disk, and registry in
smoke mode, then stop before model construction or expensive execution.

## 7. Full run later

1. Set `SMOKE_TEST=False` and run notebook 00 to completion.
2. Run `01_dataset_analysis.ipynb`.
3. Open exactly one of notebooks 02–05 in its compatible environment and run
   the environment/adapter smoke checks.
4. Open notebook 12. Keep the 2-class track, 640-pixel input, seed 42, AMP,
   effective batch 8, nine candidates, and 2/5/10/15 rungs fixed.
5. Review the measured one-day workload estimate. Explicitly opt in if it
   exceeds 24 hours, then run or resume the LR-only search.
6. Open notebook 13. It proves complete-train identity, reloads pretrained
   weights, trains 25 epochs, and evaluates once on complete official validation.
7. Run 07, 08, 09, and 10 after final checkpoints exist.
8. Run 11 only after reviewing a lightweight result bundle and Git diff.

Never compare Track A two-class mAP with published Track B ten-class mAP.
