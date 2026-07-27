import pytest

from aerial_benchmark.visdrone import parse_annotation_line


def test_parse_visdrone_annotation() -> None:
    assert parse_annotation_line("10,20,30,40,1,4,0,2") == {
        "x": 10,
        "y": 20,
        "width": 30,
        "height": 40,
        "score": 1,
        "class_id": 4,
        "truncation": 0,
        "occlusion": 2,
    }


def test_reject_short_annotation() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        parse_annotation_line("1,2,3")
