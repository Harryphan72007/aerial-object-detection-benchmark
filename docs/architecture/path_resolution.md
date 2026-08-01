# Legacy-compatible path resolution

PR 6 keeps `src.paths.ProjectPaths` authoritative for every existing notebook,
trainer, evaluator, registry, and Drive artifact. The compatibility function
`src.pathing.resolve_legacy_paths` returns that same type with identical paths.

New code may construct `RunPathIdentity` and use `ArtifactPathResolver` for
isolated namespaces with this identity:

```text
artifact / track / mode / model / experiment / seed / run_id
```

Allowed combinations are smoke/smoke, controlled/full, and
performance/{full,sliced,ensemble}. The resolver is read-only: it does not create
directories. PR 6 exposes these paths but no existing producer writes to them.

The layout is configured in `configs/shared/paths.yaml`. Changing legacy names
requires an explicit compatibility adapter and existing-artifact evaluation.
