from __future__ import annotations

from pathlib import Path

from src.training.checkpoint_selection import (
    BestCheckpointState,
    materialize_best_checkpoint,
    resolve_best_checkpoint,
)
from src.training.early_stopping import EarlyStopping


def test_early_peak_is_preserved_and_patience_round_trips() -> None:
    stopping = EarlyStopping(patience=2, min_delta=0.001)
    selector = BestCheckpointState()
    history = [0.1, 0.5, 0.4]
    for epoch, metric in enumerate(history, 1):
        selector.update(epoch, raw_metric=metric)
        stopping.update(epoch, metric)
    state = stopping.state_dict()
    resumed = EarlyStopping(patience=2, min_delta=0.001)
    resumed.load_state_dict(state)
    assert resumed.update(4, 0.3) is True
    assert resumed.best_epoch == selector.best_raw_epoch == 2
    assert selector.best_raw == 0.5


def test_raw_ema_selection_remain_independent() -> None:
    selector = BestCheckpointState()
    assert selector.update(1, raw_metric=0.4, ema_metric=0.3) == {
        "best_raw": True,
        "best_ema": True,
    }
    assert selector.update(2, raw_metric=0.35, ema_metric=0.5) == {
        "best_raw": False,
        "best_ema": True,
    }
    restored = BestCheckpointState()
    restored.load_state_dict(selector.state_dict())
    assert restored.state_dict() == selector.state_dict()


def test_best_raw_and_legacy_alias_resolve_without_replacing_last(tmp_path: Path) -> None:
    last = tmp_path / "last.pth"
    last.write_bytes(b"best-at-early-peak")
    materialize_best_checkpoint(last, tmp_path, weight_variant="raw")
    last.write_bytes(b"resumable-last")
    assert (tmp_path / "best_raw.pth").read_bytes() == b"best-at-early-peak"
    assert (tmp_path / "best.pt").read_bytes() == b"best-at-early-peak"
    assert resolve_best_checkpoint(tmp_path).name == "best.pt"
    assert last.read_bytes() == b"resumable-last"
