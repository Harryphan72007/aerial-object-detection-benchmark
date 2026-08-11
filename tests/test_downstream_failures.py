from pathlib import Path

import pytest

from scripts.evaluate import require_successful_evaluation
from scripts.profile_model import required_profile_succeeded
from src.data.image_files import first_supported_image, supported_image_files


def test_evaluation_failures_produce_nonzero_error() -> None:
    failures = [
        {
            "run_id": "run-1",
            "exception_type": "ModuleNotFoundError",
            "message": "missing backend",
        }
    ]
    with pytest.raises(RuntimeError, match="run-1"):
        require_successful_evaluation(failures)
    require_successful_evaluation([])


@pytest.mark.parametrize(
    ("profiles", "expected"),
    [
        ([{"batch_size": 1, "status": "completed"}], True),
        ([{"batch_size": 1, "status": "failed"}], False),
        ([{"batch_size": 4, "status": "completed"}], False),
        ([], False),
    ],
)
def test_required_batch_one_profile_controls_completion(
    profiles: list[dict], expected: bool
) -> None:
    assert required_profile_succeeded(profiles) is expected


def test_image_discovery_filters_extensions_and_is_deterministic(
    tmp_path: Path,
) -> None:
    (tmp_path / "00-not-an-image.txt").write_text("text", encoding="utf-8")
    (tmp_path / "b.JPG").write_bytes(b"image")
    (tmp_path / "a.png").write_bytes(b"image")
    (tmp_path / "directory.jpg").mkdir()

    images = supported_image_files(tmp_path)

    assert [path.name for path in images] == ["a.png", "b.JPG"]
    assert first_supported_image(tmp_path).name == "a.png"


def test_image_discovery_fails_clearly_without_supported_inputs(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="No supported image"):
        first_supported_image(tmp_path)
