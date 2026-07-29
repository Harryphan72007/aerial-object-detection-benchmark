"""Dataset download guidance. VisDrone is intentionally not mirrored."""
from __future__ import annotations

from pathlib import Path


def ensure_visdrone_layout(raw_root: str | Path) -> None:
    root = Path(raw_root)
    expected = [
        root / "VisDrone2019-DET-train" / "images",
        root / "VisDrone2019-DET-train" / "annotations",
        root / "VisDrone2019-DET-val" / "images",
        root / "VisDrone2019-DET-val" / "annotations",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Download VisDrone2019-DET from its official source and create this layout. "
            "Missing:\n" + "\n".join(missing)
        )
