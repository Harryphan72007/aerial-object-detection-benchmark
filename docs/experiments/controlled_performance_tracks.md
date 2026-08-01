# Controlled and performance tracks

The controlled track preserves the benchmark protocol and accepts only common
runtime options. The performance track has a different output namespace and may
enable EMA, tiled training, sliced inference, or label-granularity ablations.

Artifacts must declare `benchmark_track` and `output_namespace`. Legacy artifacts
without these fields are treated as controlled for compatibility. An explicitly
performance-tagged artifact is rejected from controlled summaries, even when its
metrics otherwise satisfy the controlled evaluator contract.

Output roots are `$DRIVE_ROOT/experiments/controlled` and
`$DRIVE_ROOT/experiments/performance`; comparison tables are respectively
`controlled_summary` and `performance_summary`.
