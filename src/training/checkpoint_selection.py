"""Best-metric state and canonical/legacy checkpoint resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from src.training.checkpointing import materialize_checkpoint_alias


@dataclass
class BestCheckpointState:
    best_raw: float = float("-inf")
    best_raw_epoch: int = 0
    best_ema: float = float("-inf")
    best_ema_epoch: int = 0

    def update(
        self, epoch: int, *, raw_metric: float, ema_metric: float | None = None
    ) -> dict[str, bool]:
        selected = {"best_raw": False, "best_ema": False}
        if float(raw_metric) > self.best_raw:
            self.best_raw = float(raw_metric)
            self.best_raw_epoch = int(epoch)
            selected["best_raw"] = True
        if ema_metric is not None and float(ema_metric) > self.best_ema:
            self.best_ema = float(ema_metric)
            self.best_ema_epoch = int(epoch)
            selected["best_ema"] = True
        return selected

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        expected = {"best_raw", "best_raw_epoch", "best_ema", "best_ema_epoch"}
        if set(value) != expected:
            raise ValueError("checkpoint-selection state fields do not match")
        self.best_raw = float(value["best_raw"])
        self.best_raw_epoch = int(value["best_raw_epoch"])
        self.best_ema = float(value["best_ema"])
        self.best_ema_epoch = int(value["best_ema_epoch"])


def materialize_best_checkpoint(
    last_checkpoint: str | Path,
    run_dir: str | Path,
    *,
    weight_variant: str,
) -> Path:
    if weight_variant not in {"raw", "ema"}:
        raise ValueError("weight_variant must be raw or ema")
    root = Path(run_dir)
    selected = root / "best.pth"
    materialize_checkpoint_alias(last_checkpoint, selected)
    return selected


def resolve_best_checkpoint(run_dir: str | Path, *, prefer_ema: bool = False) -> Path:
    root = Path(run_dir)
    candidates = [root / "best.pth", root / "best_map.pth", root / "best_raw.pth"]
    if prefer_ema:
        candidates.append(root / "best_ema.pth")
    candidates.extend((root / "best.pt", root / "best_aptiny.pth"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no best checkpoint found under {root}")
