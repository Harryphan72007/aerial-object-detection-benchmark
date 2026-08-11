# First Colab session

There is no separate bootstrap notebook. Every canonical notebook carries the
same bootstrap cell, so the first one you open does the setup:

1. Open the notebook you want from GitHub in Colab.
2. Leave the parameter cell at its shipped defaults for the first pass — every
   expensive stage is gated behind a flag that ships `False`.
3. Run all cells. The bootstrap mounts Drive, resolves the repository checkout,
   detects the platform, initializes the artifact layout, and prints the
   resolved commit, artifact root, and dependency policy.
4. Confirm the reported commit is the intended revision and `dirty` is false.
5. Review the printed stage contract, then set the start flag and run all cells
   again to begin the real run.

The bootstrap does not install a detector family or start training. Model-family
environments are provisioned only after a model is selected. Do not manually
upload a replacement notebook or copy checkpoints into the Git checkout.

Static/CPU verification does not prove Google authentication, Drive writes,
hosted package installation, CUDA, or GPU behavior. Those require an executed
Colab record.
