# Result storage and publishing

Persistent runtime artifacts live under the configured Drive root:

```text
datasets/             raw and converted data
checkpoints/lr_search candidate state and checkpoints
checkpoints/final/    resumable final runs
predictions/          raw COCO predictions
evaluation/           measured metrics and profiling
reports/              generated reports and figures
result_bundles/       lightweight validated bundles
```

None of these runtime directories belongs in Git.

The only approved repository result path is:

```text
results/bundles/<bundle-id>/
```

Bundles contain selected/final configs, search rankings and promotions, measured
metrics, small figures, environment and dataset hashes, and Git provenance.
They reject datasets, checkpoints, optimizer/scheduler states, raw predictions,
logs, credentials, private paths, secret-like content, and oversized files.

Notebook 02 defaults to `PUBLISH_RESULTS=False` and `DRY_RUN=True`. The dry-run
may create a persistent bundle but must leave Git byte-for-byte unchanged.
Explicit publishing switches to `experiment-results`, exports and stages only the
approved bundle path, shows staged names/statistics, commits, pushes, and opens or
reports a pull request to `main`.

Validate repository bundles with:

```bash
python -m scripts.validate_results --repo-results results/
```
