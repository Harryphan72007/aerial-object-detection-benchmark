# Model environments

Hosted Colab is supported through isolated subprocess environments:

| Family | Python | PyTorch / CUDA | Pinned source |
|---|---|---|---|
| RT-DETRv2 | 3.11.13 | 2.7.1+cu128 | RT-DETR `a21d516…`; R101 weights `a558a479…` |
| ResNet-50 / Swin-T | 3.10.16 | 2.1.0+cu118 | MMDetection `44ebd17…`, MMCV 2.1.0 |
| VMamba-T | 3.10.16 | 2.1.0+cu118 | MMDetection `44ebd17…`, VMamba `2ed52ea…` |

See [`configs/runtime_environments.yaml`](../configs/runtime_environments.yaml)
for exact revisions, versions, sources, and licenses. Content-addressed
environments live under `/content/visdrone_model_envs`; manifests are saved
under `$DRIVE_ROOT/environment_manifests`.

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
persistent Drive state is unchanged.
