# Model environments

Hosted Colab and Kaggle are supported through isolated subprocess environments:

| Family | Python | PyTorch / CUDA | Pinned source |
|---|---|---|---|
| RT-DETRv2 | 3.11.13 | 2.7.1+cu128 | RT-DETR `a21d516…`; R101 weights `a558a479…` |
| ResNet-50 / Swin-T | 3.10.16 | 2.1.0+cu118 | MMDetection `44ebd17…`, MMCV 2.1.0 |
| VMamba-T | 3.10.16 | 2.1.0+cu118 | MMDetection `44ebd17…`, VMamba `2ed52ea…` |

See [`configs/runtime_environments.yaml`](../configs/runtime_environments.yaml)
for exact revisions, versions, sources, and licenses. Content-addressed
environments live under `/content/visdrone_model_envs` in Colab or
`/kaggle/working/visdrone_model_envs` in Kaggle; manifests are saved under the
selected artifact root's `environment_manifests` directory.

Import success is not compatibility. Before HPO, the real adapter must construct,
run forward/backward/optimizer on one small batch, predict, save a checkpoint,
and reload it. Gate fingerprints include source commit, model/framework,
Python/PyTorch/CUDA/GPU, dependency-lock hash, and schema version.

A compatible `READY` gate is reused. A failed gate retries after source or
environment change without deleting any search or final checkpoint.

VMamba additionally requires the selective-scan extension and
`$DRIVE_ROOT/pretrained/vmamba_tiny_e292.pth`. Missing requirements produce an
exact blocker; no substitute model is used and scratch training is disabled.

The isolated design normally needs no kernel restart. If Colab itself requests
one after shared HPO dependency installation, restart once and rerun all cells;
persistent artifact state is unchanged.

## Local immutable framework checkouts

MMDetection and VMamba source are active, mutable Git repositories only while a
new pinned checkout is being built. They are never checked out or reset directly
under the persistent artifact root. The local roots are:

| Runtime | Framework root |
|---|---|
| Colab | `/content/visdrone_frameworks` |
| Kaggle | `/kaggle/working/visdrone_frameworks` |
| Local Jupyter | `<model-runtime-root>/visdrone_frameworks` |

Each checkout path contains the framework name, requested revision, and a short
repository-URL identity. For example:

```text
/content/visdrone_frameworks/VMamba/
  2ed52ead062a51a64521ed3871d52914bf532876-<url-id>/
```

Provisioning holds a framework-and-revision-specific filesystem lock before any
clone, fetch, checkout, reset, clean, submodule, or rename operation. The lock has
a configurable timeout (`VISDRONE_FRAMEWORK_LOCK_TIMEOUT_SECONDS`, 600 seconds by
default) and records its PID and timestamps in adjacent JSON metadata. Kernel-held
lock state is released automatically if a process exits. Stale metadata is
recovered only after the filesystem lock can be acquired; an active owner is
never removed.

New checkouts are created in a unique `.building-<pid>-<id>` directory. The
provisioner fetches the configured revision, creates a detached checkout, resets
tracked and staged changes, removes untracked/ignored files, initializes configured
submodules, and verifies both the resolved commit and an empty
`git status --porcelain`. It then writes `.provisioning_complete.json` and atomically
renames the directory to its final revision-keyed path. An incomplete directory is
never reused. A cached checkout is reused only when its sentinel, repository URL,
revision, HEAD, required paths, submodules, index-lock state, and clean status all
validate. Invalid local caches are quarantined and rebuilt.

VMamba's selective-scan extension is compiled from a disposable copy under the
hashed Python environment, not from the completed VMamba checkout, so build
products cannot dirty the immutable source cache.

If a repository-specific `.git/index.lock` remains after an interrupted Git
operation, it is considered for removal only while the framework provisioning
lock is held and only after process inspection finds no Git process using that
repository. Provisioning never kills Git processes or removes unrelated lock
files.

## Legacy cache migration and environment rebuilds

Older directories such as `$DRIVE_ROOT/frameworks/mmdetection` and
`$DRIVE_ROOT/frameworks/VMamba` are legacy caches. The provisioner prints a warning,
ignores them, and does not modify or delete them. Dataset archives, Optuna studies,
metrics, model checkpoints, pretrained checkpoints, and completed manifests remain
in the artifact root and are unaffected.

The model-environment hash covers the environment specification, source revisions,
provisioning version, requirements contents, verifier, and provisioning code. The
addition of `psutil==7.0.0` to the OpenMMLab requirements and this framework-cache
version therefore cause one new clean OpenMMLab/VMamba environment build. A failed
or interrupted hashed environment is not reused: diagnostics are retained under
`$DRIVE_ROOT/environment_failures`, only that local environment directory is
removed on retry, and the same notebook cell rebuilds it. A matching READY
environment still passes a quick probe before reuse.
