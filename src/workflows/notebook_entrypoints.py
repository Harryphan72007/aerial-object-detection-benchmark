"""Small presentation helpers shared by canonical notebooks.

Domain work belongs in package workflows. These helpers keep user-facing notebook
messages consistent without reintroducing notebook-local functions.
"""
from __future__ import annotations

from src.data.contract import DataContractReport


DATA_SETUP_NOTEBOOK_URL = (
    "https://colab.research.google.com/github/Harryphan72007/"
    "aerial-object-detection-benchmark/blob/main/notebooks/00_prepare_visdrone.ipynb"
)


def require_verified_data(report: DataContractReport) -> None:
    """Stop a model notebook safely when dataset preparation is incomplete."""

    if report.verified:
        return
    print("DATA CONTRACT VERIFIED: NO")
    print("Dataset setup is incomplete or was interrupted.")
    if any("staging directory" in error for error in report.errors):
        print(
            "A stale extraction staging directory was detected; notebook 00 will "
            "recover or rebuild it from the verified Drive ZIP."
        )
    print(
        "Run every cell in 00_prepare_visdrone.ipynb, wait for DATA CONTRACT "
        "VERIFIED: YES, then rerun notebook 01."
    )
    print(DATA_SETUP_NOTEBOOK_URL)
    raise RuntimeError(
        "Notebook 01 stopped safely before caching or training because dataset "
        "preparation is incomplete."
    )
