## Benchmark result publication

- Result bundle:
- Dataset track (`2class` controlled workflow):
- Class definition:
- Selected run IDs:
- Evaluated models:
- Primary resolution:
- Seed count/status:
- Training source commits:
- Known failures or exclusions:

### Reproduction

```text
python -m scripts.benchmark status
python -m scripts.benchmark publish --model-id <MODEL_ID> --dry-run
python -m scripts.validate_results --repo-results results/
```

Checkpoints, raw predictions, datasets, logs, and private Drive paths remain outside Git.
