# Archived notebook policy

The former monolithic notebooks 00–13 are deprecated and are intentionally not copied
into the runnable notebook tree. Their exact source remains available in Git history at
the migration baseline recorded in `docs/migration/MIGRATION_LOG.md`.

Archived notebooks are historical evidence only: CI does not execute them, documentation
must not link to them as entry points, and fixes belong in `src/` plus the canonical thin
notebooks. Do not restore notebook-local model, training, checkpoint, artifact, resume, or
evaluator implementations here.
