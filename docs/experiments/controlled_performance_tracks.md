# Controlled and performance tracks

> **Status.** Only the **controlled** track (Track 1) is implemented. The
> **performance** track (Track 2) is **absent**: `configs/performance/*.yaml` and
> the namespace plumbing exist as scaffolding, but there is no HPO or final
> workflow entry point that runs it. Within the controlled track, `baseline`
> (default learning rate) and `tuned` (searched learning rate) are **both Track 1**
> runs — the baseline is a default-LR diagnostic, not Track 2. Do not describe the
> `baseline`/`tuned` split as controlled-vs-performance; they differ only by
> learning rate.

The controlled track preserves the benchmark protocol and accepts only common
runtime options. The performance track *would* have a different output namespace
and enable EMA, tiled training, sliced inference, or label-granularity
ablations — but it is not yet wired to any runnable workflow.

No comparison table may contain rows from both namespaces: the comparison writers
(`src/workflows/comparison.py`, `src/workflows/versioned_comparison.py`) reject a
table that mixes a controlled and a performance run.

Artifacts must declare `benchmark_track` and `output_namespace`. Legacy artifacts
without these fields are treated as controlled for compatibility. An explicitly
performance-tagged artifact is rejected from controlled summaries, even when its
metrics otherwise satisfy the controlled evaluator contract.

Output roots are `$DRIVE_ROOT/experiments/controlled` and
`$DRIVE_ROOT/experiments/performance`; comparison tables are respectively
`controlled_summary` and `performance_summary`.
