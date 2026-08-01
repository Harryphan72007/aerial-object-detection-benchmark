from __future__ import annotations

import copy
from pathlib import Path

from src.data.tiling import canonical_dataset_hash, tile_coco_dataset, tile_windows
from src.utils.serialization import read_json

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "tiling" / "boundary_case.json"


def test_tile_windows_cover_boundaries_without_duplicate_origins() -> None:
    windows = tile_windows(100, 100, tile_size=60, overlap=20)
    assert [(tile.x, tile.y) for tile in windows] == [(0, 0), (40, 0), (0, 40), (40, 40)]


def test_boundary_clipping_ignore_regions_empty_tiles_and_source_immutability() -> None:
    source = read_json(FIXTURE)
    original = copy.deepcopy(source)
    tiled, manifest = tile_coco_dataset(
        source,
        tile_size=60,
        overlap=20,
        minimum_visible_fraction=0.25,
    )
    assert source == original
    assert manifest["source_dataset_hash"] == canonical_dataset_hash(original)
    first = manifest["tiles"][0]
    decisions = {row["source_annotation_id"]: row for row in first["annotation_decisions"]}
    assert decisions[1]["keep"] is True
    assert decisions[1]["visible_fraction"] == 1 / 3
    assert decisions[2]["reason"] == "below_visible_fraction"
    ignored = [row for row in tiled["annotations"] if row.get("source_annotation_id") == 3]
    assert ignored and all(row["ignore"] == 1 for row in ignored)
    assert any(tile["empty"] for tile in manifest["tiles"])
    assert len({image["id"] for image in tiled["images"]}) == len(tiled["images"])


def test_tiling_is_deterministic() -> None:
    source = read_json(FIXTURE)
    first = tile_coco_dataset(source, tile_size=60, overlap=20, minimum_visible_fraction=0.25)
    second = tile_coco_dataset(source, tile_size=60, overlap=20, minimum_visible_fraction=0.25)
    assert first == second
