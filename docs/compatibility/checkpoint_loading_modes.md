# Checkpoint loading modes

Checkpoint v2 metadata is classified before tensor deserialization.

- `full_resume` requires exact model, config, dataset, optimizer, scheduler,
  accumulation, and seed signatures plus model/optimizer/scheduler/training state.
- `weights_only` requires a matching model signature and model state, but starts a
  new run when any optimization or data signature differs.
- `evaluation_only` is the safe default for legacy checkpoints without v2 metadata.
- `incompatible` means even model weights cannot be safely applied.

Callers must request the classified mode explicitly. A legacy checkpoint is never
silently promoted to full resume, and no checkpoint is rewritten in place.
