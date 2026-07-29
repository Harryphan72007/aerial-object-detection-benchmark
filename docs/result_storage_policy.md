# Result storage policy

The project has three storage layers.

| Layer | Contents |
|---|---|
| GitHub | Source, configs, cleared notebooks, tests, lightweight CSV/JSON summaries, final PNG figures, Markdown/HTML reports, manifests, and small qualitative samples |
| Google Drive | VisDrone data, converted COCO annotations, pretrained files, resumable LR-search/final checkpoints, optimizer/scheduler/AMP/RNG state, complete histories/logs, raw predictions, profiling, LR-search artifacts, ONNX/TensorRT exports, complete reports, and result bundles |
| Colab `/content` | Cloned source, framework trees, caches, compiled extensions, and temporary intermediates |

The single Drive root is `/content/drive/MyDrive/visdrone_architecture_benchmark`.
VisDrone lives under
`datasets/VisDrone2019-DET/{archives,raw,processed,manifests}`.
`src.paths.ProjectPaths` remains the canonical path builder. Command-line
`--drive-root` overrides `project_config.yaml`.

Never commit `*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`, `*.onnx`, `*.engine`, datasets, raw predictions, profiling traces, TensorBoard logs, credentials, or private absolute paths. Selected release checkpoints belong in a GitHub Release, Hugging Face, or Zenodo artifact, not normal Git history.

Each publication is copied intact to
`results/bundles/<RESULT_BUNDLE_ID>/`; the latest pointer is
`results/manifests/latest_result_manifest.json`. Export never stages or commits
automatically.

## Publication gates

`sync_results_to_repo.py` validates the bundle before copying. It rejects mixed
models/tracks/runs, selected-LR mismatches, unproved official train/validation
identity, incompatible class mappings, missing registry runs, checkpoint or
annotation hash mismatches, NaN/infinite/impossible metrics, placeholders,
secret-like content, private absolute paths, excluded extensions/directories,
and files over the configured 20 MB limit. It copies only approved
CSV/JSON/YAML summaries, Markdown/HTML reports, final PNG/JPEG figures, and
small text files into `results/`.

When `--clean-target` replaces an existing bundle with the same ID, that exact
bundle is moved to `results/archive/<RESULT_BUNDLE_ID>__replaced/`; unrelated
results are untouched. Run
`python -m scripts.validate_results --repo-results results` immediately before
staging and in pull-request CI.
