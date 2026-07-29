# Checkpoint and manifest format

Each run lives under `checkpoints/MODEL_ID/RUN_ID/`. `last.pth` is updated every epoch; `best_map.pth` and `best_aptiny.pth` are immutable aliases to the best observed epochs. Framework-native files may also remain in the run directory.

The required files are listed in the README. `run_manifest.json` is the discovery contract. Paths in it are absolute so Colab reconnects can resume without guessing. The JSON registry is authoritative; `runs.csv` is atomically regenerated for analysis.

For RT-DETRv2, `.pth` stores model/optimizer/scheduler states plus the exact upstream model ID and label mapping. For MMDetection, `.pth` is the framework checkpoint and `runtime_config.py` is saved beside it.

`validate_checkpoint_manifest` checks required fields and optional file existence. Rebuilding the registry scans manifests rather than inferring run metadata from folder names.
