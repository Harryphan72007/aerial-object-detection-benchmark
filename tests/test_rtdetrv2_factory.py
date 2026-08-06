from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.models.rtdetrv2 import RTDETR_FACTORY_MODEL_ID, RTDetrV2Factory

ROOT = Path(__file__).resolve().parents[1]


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _Torch:
    cuda = _Cuda()


class _Processor:
    calls: list[tuple[str, dict]] = []

    @classmethod
    def from_pretrained(cls, source: str, **kwargs: object) -> "_Processor":
        cls.calls.append((source, kwargs))
        return cls()


class _Model:
    calls: list[tuple[str, dict]] = []

    def __init__(self) -> None:
        self.loaded: tuple[object, bool] | None = None
        self.device = None
        self.training = True
        self.config = SimpleNamespace(id2label={0: "person", 1: "vehicle"})

    @classmethod
    def from_pretrained(cls, source: str, **kwargs: object) -> "_Model":
        cls.calls.append((source, kwargs))
        return cls()

    def load_state_dict(self, state: object, strict: bool) -> SimpleNamespace:
        self.loaded = (state, strict)
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def to(self, device: str) -> "_Model":
        self.device = device
        return self

    def eval(self) -> "_Model":
        self.training = False
        return self

    def __call__(self, **inputs: object) -> dict[str, object]:
        return {"outputs": inputs}


def _config() -> dict[str, object]:
    return {
        "model_id": RTDETR_FACTORY_MODEL_ID,
        "framework": "transformers",
        "pretrained_model_name_or_path": "PekingU/rtdetr_v2_r50vd",
        "pretrained_revision": "282494075698cab9faa1096ae26856890030c817",
        "input_resolution": 640,
    }


def test_factory_loads_legacy_evaluation_checkpoint_and_runs_forward(
    tmp_path: Path,
) -> None:
    _Processor.calls.clear()
    _Model.calls.clear()
    checkpoint = tmp_path / "best_map.pth"
    checkpoint.write_bytes(b"legacy fixture")
    state = {"model": {"weight": 1}, "id2label": {0: "person", 1: "vehicle"}}
    factory = RTDetrV2Factory(ROOT, config=_config())
    result = factory.build(
        checkpoint,
        torch_module=_Torch,
        processor_class=_Processor,
        model_class=_Model,
        state_loader=lambda path: state if path == checkpoint else {},
    )
    assert result.model.loaded == (state["model"], True)
    assert result.model.device == "cpu"
    assert result.model.training is False
    assert result.report["legacy_checkpoint_loaded"] is True
    assert result.report["id2label"] == {0: "person", 1: "vehicle"}
    assert _Processor.calls[0][1]["size"] == {"height": 640, "width": 640}
    assert _Model.calls[0][1]["ignore_mismatched_sizes"] is True
    assert factory.forward(result.model, {"pixel_values": "sample"}) == {
        "outputs": {"pixel_values": "sample"}
    }


def test_factory_remote_construction_keeps_pinned_revision() -> None:
    _Processor.calls.clear()
    _Model.calls.clear()
    result = RTDetrV2Factory(ROOT, config=_config()).build(
        None,
        device="cpu",
        torch_module=_Torch,
        processor_class=_Processor,
        model_class=_Model,
    )
    assert result.report["legacy_checkpoint_loaded"] is False
    assert _Model.calls[0][1]["revision"] == _config()["pretrained_revision"]


def test_factory_rejects_incompatible_legacy_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.pth"
    checkpoint.write_bytes(b"legacy fixture")

    class IncompatibleModel(_Model):
        def load_state_dict(self, state: object, strict: bool) -> SimpleNamespace:
            return SimpleNamespace(missing_keys=["head.weight"], unexpected_keys=[])

    with pytest.raises(RuntimeError, match=r"missing=\['head.weight'\]"):
        RTDetrV2Factory(ROOT, config=_config()).build(
            checkpoint,
            torch_module=_Torch,
            processor_class=_Processor,
            model_class=IncompatibleModel,
            state_loader=lambda _: {"model": {}, "id2label": {0: "person"}},
        )


def test_factory_source_does_not_define_optimizer_or_scheduler() -> None:
    source = (ROOT / "src" / "models" / "rtdetrv2" / "factory.py").read_text()
    assert "torch.optim" not in source
    assert "lr_scheduler" not in source


def test_configured_checkpoint_matches_declared_l_variant() -> None:
    """The v2 S/M/L/X names map to R18/R34/R50/R101; L must be r50vd.

    Guards against the historical mislabel where `rtdetrv2_l` was pinned to the
    R101 (X) checkpoint while advertising the L variant.
    """
    from src.utils.serialization import read_yaml

    config = read_yaml(ROOT / "configs" / RTDETR_FACTORY_MODEL_ID / "model.yaml")
    checkpoint = str(config["pretrained_model_name_or_path"])

    assert config["variant"] == "L"
    assert config["backbone"] == "r50vd"
    # L is the r50vd backbone; r101vd is the X variant and must not appear here.
    assert "r50vd" in checkpoint
    assert "r101vd" not in checkpoint

    # Declared parameter band must be consistent with L (~42M), not X (~77M).
    params_m = int(config["approx_parameters_millions"])
    assert 35 <= params_m <= 55
