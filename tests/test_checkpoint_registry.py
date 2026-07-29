from datetime import datetime,timezone
from pathlib import Path
from src.paths import ProjectPaths
from src.training.checkpointing import RunRegistry,make_run_id,validate_manifest_dict
from src.utils.serialization import write_json

def manifest(run_id,run_dir):
    for name in ("best_map.pth","best_aptiny.pth","last.pth"):(run_dir/name).write_bytes(b"x")
    return {"run_id":run_id,"model_id":"faster_rcnn_resnet50","architecture_family":"CNN","dataset_track":"2class","class_names":["person","vehicle"],"seed":42,"input_resolution":1024,"checkpoint_best_map":str(run_dir/"best_map.pth"),"checkpoint_best_aptiny":str(run_dir/"best_aptiny.pth"),"checkpoint_last":str(run_dir/"last.pth"),"config_path":str(run_dir/"training_config.yaml"),"created_at":datetime.now(timezone.utc).isoformat(),"framework":"mmdetection","framework_version":"3.3.0","pytorch_version":"test","cuda_version":None,"gpu_name":"CPU","total_parameters":10,"trainable_parameters":8,"frozen_parameters":2,"best_validation_map":.2,"best_validation_aptiny":.1,"best_epoch":1,"total_training_seconds":2.0,"status":"completed"}

def test_run_id_format():
    value=make_run_id("m","2class",640,17,datetime(2026,7,25,15,30,0,tzinfo=timezone.utc))
    assert value=="m__2class__640__20260725_153000__seed17"

def test_atomic_registry(tmp_path):
    paths=ProjectPaths.from_value(tmp_path).create();run_id="r";run_dir=paths.run_dir("faster_rcnn_resnet50",run_id);run_dir.mkdir(parents=True);m=manifest(run_id,run_dir);mp=run_dir/"run_manifest.json";write_json(mp,m)
    reg=RunRegistry(paths);reg.register_run(mp);assert reg.get_best_run("faster_rcnn_resnet50")["run_id"]==run_id;assert reg.load_checkpoint_from_registry(run_id).exists();assert not reg.validate_checkpoint_manifest(run_id)

def test_manifest_missing_fields():
    assert validate_manifest_dict({})
