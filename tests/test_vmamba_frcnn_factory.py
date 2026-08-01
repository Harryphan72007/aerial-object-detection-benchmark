from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.models.vmamba_frcnn import (
    VMAMBA_FACTORY_MODEL_ID,
    VMambaFRCNNFactory,
    VMambaFeatureValidator,
    detect_selective_scan_backend,
)

ROOT = Path(__file__).resolve().parents[1]
REVISION = "2ed52ead062a51a64521ed3871d52914bf532876"


class _Tensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def _upstream(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "VMamba"
    detection = root / "detection"
    config = detection / "configs" / "vssm" / "mask_rcnn_vssm_fpn_coco_tiny.py"
    config.parent.mkdir(parents=True)
    config.write_text("# fixture\n")
    (detection / "model.py").write_text("REGISTERED = 'MM_VSSM'\n")
    pretrained = tmp_path / "vmamba_tiny.pth"
    pretrained.write_bytes(b"fixture")
    return root, pretrained


def _config() -> dict:
    return {
        "model": {
            "type": "MaskRCNN",
            "backbone": {"pretrained": None, "init_cfg": {"type": "Pretrained"}},
            "roi_head": {
                "bbox_head": {"num_classes": 80},
                "mask_roi_extractor": {},
                "mask_head": {"num_classes": 80},
            },
        }
    }


def test_clean_process_registration(tmp_path: Path) -> None:
    upstream, _ = _upstream(tmp_path)
    code = (
        "import json; from src.models.vmamba_frcnn.importer import "
        "register_vmamba_detection; "
        f"m=register_vmamba_detection({str(upstream)!r}); "
        "print(json.dumps({'name':m.__name__,'registered':m.REGISTERED}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {"name": "model", "registered": "MM_VSSM"}


def test_backend_report_prefers_approved_optimized_extension() -> None:
    report = detect_selective_scan_backend(
        lambda name: object() if name == "selective_scan_cuda" else None
    )
    assert report == {
        "module": "selective_scan_cuda",
        "kind": "optimized_cuda",
        "optimized": True,
        "approved": True,
    }


def test_feature_contract_accepts_nchw_and_rejects_nhwc() -> None:
    valid = [
        _Tensor((1, channels, size, size))
        for channels, size in zip((96, 192, 384, 768), (32, 16, 8, 4))
    ]
    assert VMambaFeatureValidator().validate(valid) == tuple(valid)
    invalid = [_Tensor((1, 32, 32, 96)), *valid[1:]]
    with pytest.raises(ValueError, match="must be NCHW"):
        VMambaFeatureValidator().validate(invalid)


def test_factory_enforces_revision_registration_backend_and_pretraining(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    upstream, pretrained = _upstream(tmp_path)
    monkeypatch.setenv("VMAMBA_ROOT", str(upstream))
    monkeypatch.setenv("VMAMBA_T_PRETRAINED", str(pretrained))
    result = VMambaFRCNNFactory(ROOT).build(
        num_classes=2,
        revision_reader=lambda _: REVISION,
        backend_finder=lambda name: object() if name == "selective_scan_cuda" else None,
        config_loader=lambda _: _config(),
        model_builder=lambda config, checkpoint, device: {
            "config": config,
            "checkpoint": checkpoint,
            "device": device,
        },
    )
    model = result.model["config"]["model"]
    assert model["type"] == "FasterRCNN"
    assert model["roi_head"]["bbox_head"]["num_classes"] == 2
    assert "mask_head" not in model["roi_head"]
    assert model["backbone"]["pretrained"] == str(pretrained.resolve())
    assert result.report["revision"] == REVISION
    assert result.report["selective_scan"]["approved"] is True
    assert VMAMBA_FACTORY_MODEL_ID == "faster_rcnn_vmamba_t"


def test_factory_rejects_wrong_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    upstream, pretrained = _upstream(tmp_path)
    monkeypatch.setenv("VMAMBA_ROOT", str(upstream))
    monkeypatch.setenv("VMAMBA_T_PRETRAINED", str(pretrained))
    with pytest.raises(RuntimeError, match="revision mismatch"):
        VMambaFRCNNFactory(ROOT).build(
            num_classes=2,
            revision_reader=lambda _: "0" * 40,
            backend_finder=lambda _: object(),
            config_loader=lambda _: _config(),
            model_builder=lambda *_: object(),
        )
