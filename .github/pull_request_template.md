## Migration validation

- [ ] Static and schema checks pass
- [ ] CPU tests pass
- [ ] Notebook validation, cleaning, smoke, and inventory checks pass
- [ ] Security, prohibited-artifact, and result-bundle checks pass
- [ ] No dataset, checkpoint, raw prediction, token, or private path is committed
- [ ] GPU validation is either recorded below or explicitly deferred to Colab

GPU validation evidence/defer reason:

## Benchmark result publication (when applicable)

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
