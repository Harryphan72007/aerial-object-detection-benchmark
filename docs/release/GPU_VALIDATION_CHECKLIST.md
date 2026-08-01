# GPU validation checklist

GPU checks are intentionally manual because GitHub Actions does not receive dataset,
Drive, Colab, or GPU credentials. Complete this checklist in a trusted Colab runtime.

- Record the commit SHA, Python, CUDA, GPU, framework, and driver versions.
- Run one guarded smoke workflow for every affected model adapter.
- Confirm checkpoint save, resume, and best/last selection behavior.
- Confirm legacy and versioned artifact reads produce equivalent evaluator inputs.
- Record peak GPU memory, training/inference latency, and evaluator version.
- Upload only lightweight reports and manifests; never upload datasets or checkpoints.
- Link the evidence bundle from the pull request and note every deferred item.

CI must remain fully green without GPU secrets. A GPU-required failure is release-blocking
only when the release changes model execution, checkpointing, or numerical evaluation.
