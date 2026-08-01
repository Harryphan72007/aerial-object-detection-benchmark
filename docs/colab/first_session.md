# First Colab session

1. Open `notebooks/00_bootstrap_colab.ipynb` from GitHub in Colab.
2. Select `REFERENCE_TYPE` (`branch`, `tag`, or `commit`) and `REFERENCE`.
   Exact commits or immutable tags are preferred for official experiments.
3. Run all cells. The notebook mounts Drive, clones the disposable checkout,
   selects the requested reference, installs the shared stack, initializes the
   Drive layout, and prints read-only diagnostics.
4. Confirm the reported commit is the intended revision and `dirty` is false.
5. Continue with `00_prepare_visdrone.ipynb`; select smoke mode before any full
   dataset or model workflow.

The bootstrap does not install a detector family or start training. Model-family
environments are provisioned only after a model is selected. Do not manually
upload a replacement notebook or copy checkpoints into the Git checkout.

PR 5 static/CPU verification does not prove Google authentication, Drive writes,
hosted package installation, CUDA, or GPU behavior. Those require an executed
Colab G0 record.
