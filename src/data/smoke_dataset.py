"""Create a tiny, deterministic official-layout dataset for CPU notebook smoke tests."""
from __future__ import annotations

import zipfile
from pathlib import Path

from PIL import Image

from src.data.download import VISDRONE_ARCHIVES


def create_smoke_archives(archive_dir: str | Path, images_per_split: int = 12) -> list[Path]:
    if images_per_split < 10:
        raise ValueError(
            "images_per_split must be at least 10 so both LR-search subsets are nonempty"
        )
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for split, spec in VISDRONE_ARCHIVES.items():
        folder = str(spec["folder"])
        archive_path = archive_dir / str(spec["filename"])
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index in range(images_per_split):
                image_name = f"{split}_{index + 1:07d}.jpg"
                image_path = archive_dir / f".{split}-{image_name}"
                width, height = 160, 96
                Image.new(
                    "RGB",
                    (width, height),
                    color=((index * 31) % 255, (index * 59) % 255, 90),
                ).save(image_path)
                archive.write(image_path, f"{folder}/images/{image_name}")
                image_path.unlink()
                rows = [
                    "2,2,6,8,1,1,0,0",
                    "12,4,8,10,1,2,0,1",
                    f"{30 + index},20,12,10,1,{3 + (index % 8)},1,2",
                    "60,30,20,16,1,4,0,0",
                    "0,0,15,15,0,0,0,0",
                ]
                archive.writestr(
                    f"{folder}/annotations/{Path(image_name).stem}.txt",
                    "\n".join(rows) + "\n",
                )
        outputs.append(archive_path)
    return outputs
