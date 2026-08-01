"""Validated run identity shared by new artifact namespaces."""
from __future__ import annotations

import re
from dataclasses import dataclass


SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TRACK_MODES = {
    "smoke": {"smoke"},
    "controlled": {"full"},
    "performance": {"full", "sliced", "ensemble"},
}


@dataclass(frozen=True)
class RunPathIdentity:
    track: str
    mode: str
    model_id: str
    experiment: str
    seed: int
    run_id: str

    def __post_init__(self) -> None:
        if self.track not in TRACK_MODES:
            raise ValueError(f"unsupported track: {self.track}")
        if self.mode not in TRACK_MODES[self.track]:
            raise ValueError(
                f"mode {self.mode!r} is incompatible with track {self.track!r}"
            )
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        for name, value in (
            ("model_id", self.model_id),
            ("experiment", self.experiment),
            ("run_id", self.run_id),
        ):
            if not SAFE_SEGMENT.fullmatch(value):
                raise ValueError(f"unsafe {name}: {value!r}")
