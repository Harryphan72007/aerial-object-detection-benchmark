import json

from PIL import Image

from src.data.collapse_classes import ClassMapping
from src.data.convert_visdrone import convert_split, parse_visdrone_line, validate_box
from src.data.validate_annotations import validate_coco


def test_parse_line():
    assert parse_visdrone_line("1,2,3,4,1,4,0,1") == (1, 2, 3, 4, 1, 4, 0, 1)
    assert parse_visdrone_line("1,2,3,4,1,4,0,1,") == (1, 2, 3, 4, 1, 4, 0, 1)


def test_parse_line_rejects_malformed_rows():
    import pytest

    for row in ("1,2,3", "1,2,3,4,1,4,0,nope", "1,2,3.5,4,1,4,0,1"):
        with pytest.raises(ValueError):
            parse_visdrone_line(row)


def test_box_validation_reports_every_defect():
    assert validate_box(1, 2, 3, 4, 20, 20) == []
    assert validate_box(-1, -2, 0, 30, 20, 20) == [
        "zero_area",
        "negative_coordinates",
        "out_of_bounds",
    ]


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


def test_conversion_audits_invalid_rows_and_is_deterministic(tmp_path):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    Image.new("RGB", (20, 20)).save(images / "0001.jpg")
    (annotations / "0001.txt").write_text(
        "1,2,5,6,1,1,0,0\n"
        "-1,2,5,6,1,4,0,0\n"
        "10,10,0,8,1,4,0,0\n"
        "18,18,5,5,1,4,0,0\n"
        "1,2,5,6,1,99,0,0\n"
        "malformed\n",
        encoding="utf-8",
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    report_path = tmp_path / "audit.json"
    summary = convert_split(
        images,
        annotations,
        first,
        ClassMapping("10class"),
        report_json=report_path,
    )
    convert_split(images, annotations, second, ClassMapping("10class"))
    assert first.read_bytes() == second.read_bytes()
    assert summary.annotations == 1
    assert summary.negative_coordinates == 1
    assert summary.zero_area_boxes == 1
    assert summary.out_of_bounds_boxes == 1
    assert summary.unknown_category_ids == 1
    assert summary.malformed_rows == 1
    assert report_path.is_file()


def test_ten_class_and_two_class_outputs(tmp_path):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    Image.new("RGB", (40, 40)).save(images / "0001.jpg")
    rows = [f"1,1,2,2,1,{category_id},0,0" for category_id in range(0, 12)]
    (annotations / "0001.txt").write_text("\n".join(rows), encoding="utf-8")
    ten = tmp_path / "ten.json"
    two = tmp_path / "two.json"
    ten_summary = convert_split(images, annotations, ten, ClassMapping("10class"))
    two_summary = convert_split(images, annotations, two, ClassMapping("2class"))
    ten_payload = json.loads(ten.read_text())
    two_payload = json.loads(two.read_text())
    assert [item["category_id"] for item in ten_payload["annotations"]] == list(
        range(1, 11)
    )
    assert [item["category_id"] for item in two_payload["annotations"]] == [
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
    ]
    assert ten_summary.ignored_regions == two_summary.ignored_regions == 2
