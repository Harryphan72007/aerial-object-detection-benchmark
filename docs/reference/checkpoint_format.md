# Checkpoint and manifest format

Each final run lives under `checkpoints/final/MODEL_ID/RUN_ID/`. While active,
`last.pth` is atomically overwritten and contains all resume state plus the selected
validation-mAP model state. A completed run contains exactly one model file:
`best.pth`. Its v2 identity records run ID, model ID, seed, configuration hash,
selected epoch/metric, and the raw or EMA weight variant.

Legacy directories are resolved read-only in this order: `best.pth`,
`best_map.pth`, `best_raw.pth`, then explicitly enabled compatibility aliases.

The required files are listed in the README. `run_manifest.json` is the discovery contract. Paths in it are absolute so Colab reconnects can resume without guessing. The JSON registry is authoritative; `runs.csv` is atomically regenerated for analysis.

For RT-DETRv2, `.pth` stores model/optimizer/scheduler states plus the exact upstream model ID and label mapping. For MMDetection, `.pth` is the framework checkpoint and `runtime_config.py` is saved beside it.

`validate_checkpoint_manifest` checks required fields and optional file existence. Rebuilding the registry scans manifests rather than inferring run metadata from folder names.
