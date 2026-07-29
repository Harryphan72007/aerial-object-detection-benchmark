# Result storage policy

The project has three storage layers.

| Layer | Contents |
|---|---|
| GitHub | Source, configs, cleared notebooks, tests, lightweight CSV/JSON summaries, final PNG figures, Markdown/HTML reports, manifests, and small qualitative samples |
| Google Drive | VisDrone data, converted COCO annotations, pretrained files, resumable checkpoints, optimizer/scheduler/AMP state, complete histories/logs, raw predictions, profiling, Optuna, ONNX/TensorRT exports, complete reports, and result bundles |
| Colab `/content` | Cloned source, framework trees, caches, compiled extensions, and temporary intermediates |

The single Drive root is `/content/drive/MyDrive/visdrone_architecture_benchmark`. `src.paths.ProjectPaths` remains the canonical path builder. Command-line `--drive-root` overrides `project_config.yaml`.

Never commit `*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`, `*.onnx`, `*.engine`, datasets, raw predictions, profiling traces, TensorBoard logs, credentials, or private absolute paths. Selected release checkpoints belong in a GitHub Release, Hugging Face, or Zenodo artifact, not normal Git history.

Before replacing a publication, the exporter archives the previous validated set under `results/archive/<PREVIOUS_RESULT_BUNDLE_ID>/`. Export never stages or commits automatically.

## Publication gates

`sync_results_to_repo.py` validates the bundle before copying. It rejects mixed `2class`/`10class` tracks, incompatible class mappings, missing registry runs, checkpoint or annotation hash mismatches, NaN/infinite/impossible metrics, secret-like content, private absolute paths, excluded extensions, and files over the configured 20 MB limit. It copies only CSV/JSON summaries, Markdown/HTML reports, final PNG/JPEG figures, and small text/sample files into `results/`.

The previous validated publication is moved to `results/archive/<PREVIOUS_RESULT_BUNDLE_ID>/` when `--clean-target` is used. Run `scripts/validate_results.py --repo-results results/` again immediately before staging and in pull-request CI.
