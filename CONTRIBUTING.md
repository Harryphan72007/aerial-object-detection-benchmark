# Contributing

Create one focused branch and pull request per logical change. Preserve legacy
artifact readers and notebook behavior unless the change explicitly migrates a
versioned contract. Add the smallest regression test and document the exact
rollback.

Before committing:

```bash
python scripts/clean_notebooks.py notebooks
python -m scripts.validation.check_prohibited_files
make verify
```

Never commit datasets, generated tiles, checkpoints, weights, raw predictions,
Optuna databases, credentials, Drive tokens, runtime caches, or unverified
benchmark numbers. Store large runtime artifacts on Google Drive. Only small,
reviewed fixtures under `tests/fixtures` and validated lightweight result bundles
under `results/bundles` may cross that boundary.

Document the exact source, revision, license, checksum, preprocessing, hardware,
seed, dataset track, mode, and evaluation protocol for new models or results.
Controlled and performance experiments must use separate configs and output
namespaces.

Keep framework-specific code behind adapters and reusable logic outside
notebooks. Treat `.ipynb` files as canonical: strip outputs and execution counts,
do not manually upload replacement notebooks, and do not claim Colab/GPU
validation unless it was executed.

See `SECURITY.md` for private vulnerability and credential reporting.
