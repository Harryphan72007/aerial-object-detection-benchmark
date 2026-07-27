import json

import pytest

from aerial_benchmark.results import append_result


def test_result_writer_rejects_invented_or_missing_metrics(tmp_path) -> None:
    with pytest.raises(ValueError, match="Missing evaluation metrics"):
        append_result(
            tmp_path / "runs.jsonl",
            {
                "model": "demo",
                "seed": 17,
                "metrics": {"map": 0.5},
                "provenance": {},
                "prediction_sha256": "abc",
            },
        )


def test_result_writer_appends_valid_record(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    metrics = {
        "map": 0.1,
        "map_50": 0.2,
        "map_75": 0.05,
        "map_small": 0.01,
        "map_medium": 0.1,
        "map_large": 0.2,
    }
    append_result(
        path,
        {
            "model": "demo",
            "seed": 17,
            "metrics": metrics,
            "provenance": {"git_revision": "abc"},
            "prediction_sha256": "def",
        },
    )
    assert json.loads(path.read_text())["metrics"] == metrics
