from __future__ import annotations

from pathlib import Path

from scripts.validation.check_prohibited_files import (
    validate_file,
    validate_repository,
)


ROOT = Path(__file__).resolve().parents[1]


def _file(tmp_path: Path, name: str, content: bytes = b"x") -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_current_tracked_repository_contains_no_prohibited_files() -> None:
    assert not validate_repository(ROOT)


def test_rejects_runtime_artifacts_credentials_and_secrets(tmp_path: Path) -> None:
    cases = [
        (_file(tmp_path, "model.pth"), "checkpoints/model.pth"),
        (_file(tmp_path, "study.db"), "optuna/study.db"),
        (_file(tmp_path, "image.jpg"), "datasets/VisDrone/image.jpg"),
        (_file(tmp_path, "credentials.json"), "credentials.json"),
        (
            _file(tmp_path, "notes.txt", b"github_pat_" + b"a" * 30),
            "docs/notes.txt",
        ),
    ]
    for path, relative in cases:
        assert validate_file(path, relative), relative


def test_allows_small_reviewed_fixtures_and_lightweight_results(tmp_path: Path) -> None:
    fixture = _file(tmp_path, "tiny.json", b'{"images": []}\n')
    metrics = _file(tmp_path, "metrics.json", b'{"mAP": 0.5}\n')
    assert not validate_file(fixture, "tests/fixtures/tiny.json")
    assert not validate_file(metrics, "results/bundles/example/metrics.json")


def test_fixture_and_repository_size_limits_are_enforced(tmp_path: Path) -> None:
    fixture = _file(tmp_path, "large.json", b"x" * 1025)
    ordinary = _file(tmp_path, "ordinary.txt", b"x" * 2049)
    assert validate_file(
        fixture, "tests/fixtures/large.json", max_bytes=1024
    )
    assert validate_file(ordinary, "docs/ordinary.txt", max_bytes=2048)
