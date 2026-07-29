import json

from PIL import Image

from src.data.collapse_classes import ClassMapping
from src.data.convert_visdrone import convert_split, parse_visdrone_line
from src.data.validate_annotations import validate_coco


def test_parse_line():
    assert parse_visdrone_line("1,2,3,4,1,4,0,1") == (1, 2, 3, 4, 1, 4, 0, 1)


def test_convert_and_validate(tmp_path):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    Image.new("RGB", (20, 20)).save(images / "0001.jpg")
    (annotations / "0001.txt").write_text(
        "1,2,5,6,1,1,0,0\n"
        "10,10,8,8,1,4,1,2\n"
        "0,0,2,2,1,0,0,0\n",
        encoding="utf-8",
    )
    output = tmp_path / "instances.json"
    summary = convert_split(
        images, annotations, output, ClassMapping("2class")
    )
    assert summary.images == 1
    assert summary.annotations == 2
    assert summary.ignored_regions == 1
    data = json.loads(output.read_text())
    assert [annotation["category_id"] for annotation in data["annotations"]] == [
        1,
        2,
    ]
    report = validate_coco(output, images)
    assert report.valid, report.errors
