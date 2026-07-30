# Result storage and publishing

Runtime artifacts remain under `$DRIVE_ROOT`: datasets, HPO studies, checkpoints,
registry files, predictions, evaluation, profiling, reports, and result bundles.

Git publication permits only:

```text
results/bundles/<bundle-id>/**
results/manifests/latest_result_manifest.json
```

Before commit, the publisher rejects secrets, private paths, oversized files,
datasets, checkpoints, raw predictions, archives, credentials, and unexpected
staged paths. It validates the complete staged result tree and displays staged
filenames and diff statistics.

Publishing requires a clean `main` source checkout, `GH_TOKEN` authentication,
and verified push permission. Work happens in a separately configured temporary
clone. A missing `experiment-results` branch starts safely from `origin/main`;
an existing branch is fetched and fast-forwarded. Force-push is never used.

Dry-run needs no authentication and leaves Git unchanged:

```bash
python -m scripts.benchmark publish --model-id rtdetrv2_l --dry-run
python -m scripts.validate_results --repo-results results/
```
