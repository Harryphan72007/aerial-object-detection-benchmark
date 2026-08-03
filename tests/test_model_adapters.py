import sys
import types
import json
import os
import subprocess
from pathlib import Path

import pytest
from src.models.mmdetection_adapter import MMDetectionAdapter
from src.models.registry import create_adapter,load_model_config
from src.models.rtdetr_adapter import RTDetrV2Adapter

def test_registry_constructs_without_heavy_imports():
    assert isinstance(create_adapter("faster_rcnn_resnet50","cpu"),MMDetectionAdapter)
    assert isinstance(create_adapter("rtdetrv2_l","cpu"),RTDetrV2Adapter)

def test_model_config_resolution():
    cfg=load_model_config("rtdetrv2_l")
    assert cfg["framework"]=="transformers"

def test_unknown_model():
    with pytest.raises(KeyError):create_adapter("unknown")


def test_vmamba_adapter_registers_external_modules_before_init_detector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, object]] = []
    apis = types.ModuleType("mmdet.apis")

    def init_detector(config: str, checkpoint: str, *, device: str):
        calls.append(("init", (config, checkpoint, device)))
        return object()

    apis.init_detector = init_detector
    mmdet = types.ModuleType("mmdet")
    mmdet.apis = apis
    monkeypatch.setitem(sys.modules, "mmdet", mmdet)
    monkeypatch.setitem(sys.modules, "mmdet.apis", apis)
    monkeypatch.setattr(
        "src.models.vmamba_frcnn.importer.register_vmamba_detection",
        lambda root: calls.append(("register", root)),
    )
    external = tmp_path / "VMamba"
    adapter = MMDetectionAdapter("faster_rcnn_vmamba_t", "cpu")

    adapter.load_model(
        tmp_path / "checkpoint.pth",
        {
            "resolved_framework_config": str(tmp_path / "config.py"),
            "registration_import": "model",
            "external_root": str(external),
        },
    )

    assert calls[0] == ("register", external)
    assert calls[1][0] == "init"


def test_vmamba_adapter_registration_works_in_fresh_process_without_pythonpath(
    tmp_path: Path,
) -> None:
    external = tmp_path / "VMamba"
    detection = external / "detection"
    detection.mkdir(parents=True)
    (detection / "model.py").write_text("REGISTERED = 'MM_VSSM'\n", encoding="utf-8")
    code = f"""
import json, sys, types
apis = types.ModuleType('mmdet.apis')
def init_detector(config, checkpoint, device):
    import model
    return {{'registered': model.REGISTERED, 'device': device}}
apis.init_detector = init_detector
mmdet = types.ModuleType('mmdet')
mmdet.apis = apis
sys.modules['mmdet'] = mmdet
sys.modules['mmdet.apis'] = apis
from src.models.mmdetection_adapter import MMDetectionAdapter
adapter = MMDetectionAdapter('faster_rcnn_vmamba_t', 'cpu')
result = adapter.load_model('checkpoint.pth', {{
    'resolved_framework_config': 'config.py',
    'registration_import': 'model',
    'external_root': {str(external)!r},
}})
print(json.dumps(result))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
    )
    assert json.loads(completed.stdout) == {"registered": "MM_VSSM", "device": "cpu"}
