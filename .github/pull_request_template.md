## Benchmark result publication

- Result bundle:
- Dataset track (`2class` or `10class`):
- Class definition:
- Selected run IDs:
- Evaluated models:
- Primary resolution:
- Seed count/status:
- Training source commits:
- Known failures or exclusions:

### Reproduction

```text
python scripts/evaluate.py --drive-root "<DRIVE_ROOT>" --dataset-track <TRACK> --best-per-model
python scripts/create_results_manifest.py --drive-root "<DRIVE_ROOT>" --dataset-track <TRACK>
python scripts/sync_results_to_repo.py --drive-root "<DRIVE_ROOT>" --bundle-id "<BUNDLE_ID>" --repo-root . --validate --dry-run
```

Checkpoints, raw predictions, datasets, logs, and private Drive paths remain outside Git.
