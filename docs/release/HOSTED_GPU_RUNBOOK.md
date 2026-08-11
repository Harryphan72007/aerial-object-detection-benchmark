# Hosted GPU runbook (Colab / Kaggle)

The order below is mandatory. Each step's output is the next step's precondition,
and the adapter gate is enforced in code, not by convention.

```text
1. prepare dataset          notebook 00
2. provision model runtime  notebook 10-13 / 20-23 cell, or the smoke driver
3. GPU adapter smoke        python -m scripts.gpu_adapter_smoke
4. confirm READY record     <artifact_root>/adapter_smoke/<model>__<track>__smoke.json
5. HPO                      notebook 10-13, START_HPO = True
6. final training           notebook 20-23, START_FINETUNING = True
```

Steps 5 and 6 refuse to start without the step-4 record.

## 0. Runtime requirements

| | Colab | Kaggle |
|---|---|---|
| Accelerator | any CUDA GPU (T4/L4/A100) | GPU T4 ×1 or P100 |
| Artifact root (persists) | `/content/drive/MyDrive/visdrone_architecture_benchmark` | `/kaggle/working/visdrone_architecture_benchmark` |
| Model runtimes (rebuildable) | `/content/visdrone_model_envs` | `/kaggle/temp/visdrone_model_envs` |
| Framework checkouts (rebuildable) | `/content/visdrone_frameworks` | `/kaggle/temp/visdrone_frameworks` |
| HPO trial scratch | `/content/visdrone_hpo_trials` | `/kaggle/temp/visdrone_hpo_trials` |

Kaggle caps `/kaggle/working` at **20 GB** and saves it as the notebook output.
Only real artifacts live there: the prepared dataset (~5.5 GB), the Optuna study,
and checkpoints. Everything rebuildable goes to `/kaggle/temp`, which is scratch
on the same larger disk and outside that quota - one OpenMMLab runtime is 5-7 GB
once torch+cu118, the NVIDIA libraries, and MMCV/MMDetection are installed, and
would otherwise exhaust the budget before the first checkpoint is written.

### Kaggle session settings (all three are required)

| Setting | Value | Why |
|---|---|---|
| Accelerator | `GPU T4 x2` or `GPU P100` | the adapter gate and every run need CUDA |
| Internet | **On** (needs phone verification) | off by default; without it `git clone`, `pip install`, and the dataset download all fail |
| Persistence | **Files only** (or Variables and Files) | otherwise `/kaggle/working` is empty on the next session and the dataset is re-downloaded |

Kaggle GPU quota is ~30 h/week with a 12 h session cap. One model's HPO plus its
final run fits that if you keep persistence on and let the workflow resume.

Google Drive is never used on Kaggle - `USE_GOOGLE_DRIVE` is ignored there, and
`/kaggle/working` is the persistent root.

Model environments are isolated `uv` virtualenvs, so the notebook kernel's own
packages are never replaced and no kernel restart is normally required. If the
shared dependency install does replace an already-imported compiled package, the
bootstrap stops with `RESTART REQUIRED`, names the package, and asks for exactly
one restart; rerun all cells afterwards.

## 1. Prepare the dataset

Open `notebooks/00_prepare_visdrone.ipynb`, run all cells, and wait for
`DATA CONTRACT VERIFIED: YES`. Set `PREPARE_10CLASS_TRACK = True` first if you
intend to run the 10-class track.

## 2–4. Provision and gate each model

Run these in a GPU notebook cell (or a terminal in the repository root). The
driver provisions the model runtime, executes the checks inside it, and writes the
record.

```bash
python -m scripts.gpu_adapter_smoke \
  --repo-root . \
  --drive-root /content/drive/MyDrive/visdrone_architecture_benchmark \
  --dataset-track 2class \
  --model-id faster_rcnn_resnet50
```

Repeat with `--model-id faster_rcnn_swin_t`, `--model-id faster_rcnn_vmamba_t`,
and `--model-id rtdetrv2_l`, or omit `--model-id` to gate all four in one run.

Expected final line per model:

```text
faster_rcnn_resnet50: READY
  record: /content/drive/.../adapter_smoke/faster_rcnn_resnet50__2class__smoke.json
```

Anything other than `READY` means the record's `failure` and `checks` fields name
the failed check, exception type, and traceback. Fix that before continuing; do
not delete the record to "unblock" a run — the gate reads it.

### ResNet-50 — `faster_rcnn_resnet50`

OpenMMLab runtime: Python 3.10.16, torch 2.1.0+cu118, MMCV 2.1.0, MMDetection
`44ebd17…`. The backbone checkpoint is the torchvision ResNet-50 referenced by
the pinned MMDetection config and is fetched into the Torch hub cache.

### Swin-T — `faster_rcnn_swin_t`

Same OpenMMLab runtime. The base config is a Mask R-CNN config converted to
bbox-only Faster R-CNN at construction time, exactly as the training backend does
it. MMDetection's `swin_converter` renames and reorders some patch-merging
tensors, so the pretrained-load check uses a value-coverage threshold of 0.85
rather than key equality; the measured coverage is written into the record.

### VMamba-T — `faster_rcnn_vmamba_t`

Additionally needs the selective-scan CUDA extension and the pinned checkpoint.
Provisioning prints this diagnostic block before compiling:

```text
PyTorch version: 2.1.0+cu118
PyTorch CUDA version: 11.8
Host CUDA toolkit version: 12.4          <- Colab's default nvcc
Selected CUDA_HOME: /usr/local/cuda-11.8
nvcc path: /usr/local/cuda-11.8/bin/nvcc
nvcc version: 11.8
Host C++ compiler: /usr/bin/g++-11 (11.4.0)
Host C compiler: /usr/bin/gcc-11
VMamba pretrained checkpoint: reused|downloaded <path> (sha256 dbc0cc4f5ec0..., 91649482 bytes)
selective_scan build status: READY
```

**Why a second toolkit is needed.** `selective_scan_cuda_oflex` is a Torch CUDA
extension, and `torch.utils.cpp_extension` refuses to build when the toolkit's
CUDA *major* version differs from PyTorch's. The cu118 pin is not a preference:
the only published MMCV 2.1.0 wheels are `cu118/torch2.1`. Hosted images ship
CUDA 12.x, so an 11.x toolkit must be present.

Resolution order:

1. `VISDRONE_CUDA_HOME`, then `CUDA_HOME`/`CUDA_PATH`;
2. `/usr/local/cuda-11.8`, `/usr/local/cuda-11`, `/opt/cuda-11.8`;
3. whatever `nvcc` is on `PATH`.

If none matches, and the platform is Colab or Kaggle, provisioning installs the
minimal pinned package set (`cuda-nvcc-11-8`, `cuda-cudart-dev-11-8`,
`cuda-cccl-11-8`, `libcusparse-dev-11-8`, `libcublas-dev-11-8`, `g++-11`) and
re-resolves. Set `VISDRONE_INSTALL_CUDA_TOOLKIT=0` to forbid that; the run then
blocks with the exact `apt-get` command instead. The selected toolkit and
compiler are exported into the build subprocess only — `/usr/local/cuda` is never
repointed.

If you prefer to install it yourself first:

```bash
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  cuda-nvcc-11-8 cuda-cudart-dev-11-8 cuda-cccl-11-8 \
  libcusparse-dev-11-8 libcublas-dev-11-8 g++-11
```

**Checkpoint.** VMamba-T will not train from scratch. Provisioning downloads,
verifies, and atomically installs:

| | |
|---|---|
| Stored as | `<artifact_root>/pretrained/vmamba_tiny_e292.pth` |
| Upstream name | `vssmtiny_dp01_ckpt_epoch_292.pth` |
| Source | `https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmtiny_dp01_ckpt_epoch_292.pth` |
| SHA-256 | `dbc0cc4f5ec0e45db5fba7c939d2c7d9b617e891ac10766912a8d604c37c5e47` |
| Size | 91,649,482 bytes |

This is the file the pinned detector config
(`detection/configs/vssm/mask_rcnn_vssm_fpn_coco_tiny.py` at revision
`2ed52ead…`) names, and it matches that config's v0 backbone shape
(`dims=96, depths=(2,2,9,2), ssm_ratio=2.0, mlp_ratio=0.0`). The VMamba **v2**
tiny checkpoints (`vssm1_tiny_0230s`) are a different architecture and must not
be substituted. To stage it in advance, or on an offline host:

```bash
python -m scripts.fetch_pretrained_checkpoints --drive-root <artifact_root> --family vmamba
python -m scripts.fetch_pretrained_checkpoints --drive-root <artifact_root> --verify-only
```

Set `VISDRONE_ALLOW_CHECKPOINT_DOWNLOAD=0` to forbid automatic downloads; the run
then blocks and names the fetch command.

### RT-DETRv2-L — `rtdetrv2_l`

Separate runtime: Python 3.11.13, torch 2.7.1+cu128, Transformers 4.52.4. Weights
come from `PekingU/rtdetr_v2_r50vd` at revision `282494…`. The class head is
resized for the track's class count, so Transformers reports those tensors as
`mismatched_keys`; that is expected and recorded. `missing_keys` and
`unexpected_keys` must both be empty.

## 5. HPO

Open `notebooks/1{0,1,2,3}_*.ipynb`, keep `DATASET_TRACK` matching the gated
record, review the preview with `START_HPO = False`, then set `START_HPO = True`
and run all cells. Reruns load the persisted study and only run missing finite
trials.

## 6. Final training

Open the matching `notebooks/2{0,1,2,3}_*.ipynb`. `FULL_MATRIX = False` (default)
runs the tuned recipe at seed 42. `FULL_MATRIX = True` runs the opt-in
`baseline`+`tuned` × seeds 17/42/3407 matrix — six runs, a multi-session job,
reported separately.

## 7. Runtime budget

Before quoting GPU hours:

```bash
python -m scripts.measure_throughput --drive-root <artifact_root>
```

It measures every controlled-track model in one pass — there is no per-model
flag — so run it from a runtime where each family's environment is importable,
or run it once per family after that family's provisioning cell.

`docs/reference/runtime_budget.md` states `t_iter` is unmeasured and its hour
columns are `null` until this has run on the target hardware.

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `EnvironmentNotProvisionedError` | no isolated runtime selected | run the notebook's `ensure_model_environment` cell, or the smoke driver without `--skip-provisioning` |
| `AdapterGateError: no adapter smoke record exists` | step 3 not done for this model/track | run the gate command above |
| `AdapterGateError` listing `git_commit`/`gpu` differences | record made on another commit or GPU | rerun the gate here |
| `DatasetTrackNotPreparedError` | `10class` selected but not prepared | rerun notebook 00 with `PREPARE_10CLASS_TRACK = True` |
| `CheckpointVerificationError` | truncated or wrong VMamba checkpoint | delete it and rerun, or use `scripts.fetch_pretrained_checkpoints` |
| `EnvironmentProvisioningError` naming CUDA | no 11.x toolkit | run the `apt-get` block above, or set `VISDRONE_CUDA_HOME` |
| `KernelRestartRequired` | a compiled package was replaced under the kernel | restart the runtime once and rerun all cells |
| `DriveUnavailableError` / `NotImplementedError: Mounting drive is unsupported` | a Colab draft, preview, restricted, enterprise, or local runtime — `drive.mount` needs `/var/colab/hostname` | connect a standard hosted runtime; or point `VISDRONE_DRIVE_ROOT` at a directory you mounted yourself; or set `USE_GOOGLE_DRIVE = False` for a **smoke run only** — session storage is deleted at disconnect |
| `WARNING: EPHEMERAL ARTIFACT ROOT` | running without Drive | expected only for smoke runs; never start HPO or final training from a session-storage root |
