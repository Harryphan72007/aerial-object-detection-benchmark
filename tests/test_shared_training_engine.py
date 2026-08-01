from __future__ import annotations

from pathlib import Path

from src.artifacts import ArtifactIdentity, write_prediction_artifact
from src.evaluation.detection_metrics import detailed_metrics
from src.models.resnet_frcnn.trainer import ResNetSharedTrainer
from src.training.state import TrainingState
from src.training.state_checkpoint import load_training_state, save_training_state
from src.utils.serialization import read_json

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "legacy_artifacts"


class _Component:
    def __init__(self) -> None:
        self.steps = 0
        self.zeroes = 0

    def step(self) -> None:
        self.steps += 1

    def zero_grad(self) -> None:
        self.zeroes += 1

    def state_dict(self) -> dict[str, int]:
        return {"steps": self.steps, "zeroes": self.zeroes}

    def load_state_dict(self, value: dict[str, int]) -> None:
        self.steps = value["steps"]
        self.zeroes = value["zeroes"]


def test_one_epoch_checkpoint_resume_next_epoch_and_legacy_evaluation(
    tmp_path: Path,
) -> None:
    optimizer = _Component()
    scheduler = _Component()
    state = TrainingState()
    backward_values: list[float] = []
    engine = ResNetSharedTrainer(accumulation_steps=2, use_amp=True)
    first = engine.run_epoch(
        [2.0, 4.0, 6.0],
        state,
        forward_loss=float,
        backward=backward_values.append,
        optimizer=optimizer,
        scheduler=scheduler,
        amp_context=lambda: __import__("contextlib").nullcontext(),
    )
    assert first.epoch == 1
    assert first.optimizer_steps == optimizer.steps == scheduler.steps == 2
    assert backward_values == [1.0, 2.0, 3.0]
    checkpoint = tmp_path / "engine_state.json"
    save_training_state(
        checkpoint, state, components={"optimizer": optimizer, "scheduler": scheduler}
    )

    resumed_optimizer = _Component()
    resumed_scheduler = _Component()
    resumed = load_training_state(
        checkpoint,
        components={"optimizer": resumed_optimizer, "scheduler": resumed_scheduler},
    )
    second = engine.run_epoch(
        [8.0],
        resumed,
        forward_loss=float,
        backward=lambda _: None,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
    )
    assert second.epoch == 2
    assert resumed.optimizer_step == 3
    assert resumed_optimizer.steps == resumed_scheduler.steps == 3

    identity = ArtifactIdentity(
        "resnet-smoke", "faster_rcnn_resnet50", "CNN", "2class", 640, 42
    )
    predictions = read_json(FIXTURE / "predictions.json")
    paths = write_prediction_artifact(
        tmp_path / "predictions.v1.json",
        identity,
        predictions,
        legacy_destination=tmp_path / "predictions.json",
    )
    metrics = detailed_metrics(FIXTURE / "ground_truth.json", paths["legacy"])
    assert metrics["per_class_detailed"]["person"]["true_positives"] == 1


def test_resume_rejects_missing_component(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    save_training_state(path, TrainingState(), components={"optimizer": _Component()})
    try:
        load_training_state(path, components={})
    except ValueError as error:
        assert "component set" in str(error)
    else:
        raise AssertionError("incompatible resume must fail")
