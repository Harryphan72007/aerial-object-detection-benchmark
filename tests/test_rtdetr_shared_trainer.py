from __future__ import annotations

from pathlib import Path

from src.artifacts import ArtifactIdentity, write_prediction_artifact
from src.evaluation.detection_metrics import detailed_metrics
from src.models.rtdetrv2.trainer import RTDetrSharedTrainer
from src.training.state import TrainingState
from src.training.state_checkpoint import load_training_state, save_training_state
from src.utils.serialization import read_json

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "legacy_artifacts"


class _Component:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def zero_grad(self) -> None:
        return None

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, value):
        self.steps = int(value["steps"])


def test_rtdetr_batch_step_resume_ema_hook_and_legacy_evaluator(tmp_path: Path) -> None:
    optimizer = _Component()
    scheduler = _Component()
    ema = _Component()
    state = TrainingState()
    trainer = RTDetrSharedTrainer(accumulation_steps=2, use_amp=True)
    result = trainer.run_epoch(
        [2.0, 4.0],
        state,
        forward_loss=float,
        backward=lambda loss: None,
        optimizer=optimizer,
        scheduler=scheduler,
        amp_context=lambda: __import__("contextlib").nullcontext(),
        after_optimizer_step=ema.step,
    )
    assert result.optimizer_steps == optimizer.steps == scheduler.steps == ema.steps == 1
    checkpoint = tmp_path / "rtdetr_state.json"
    save_training_state(
        checkpoint, state, components={"optimizer": optimizer, "scheduler": scheduler}
    )
    resumed_optimizer = _Component()
    resumed_scheduler = _Component()
    resumed = load_training_state(
        checkpoint,
        components={"optimizer": resumed_optimizer, "scheduler": resumed_scheduler},
    )
    assert resumed.epoch == 1
    assert resumed_optimizer.steps == resumed_scheduler.steps == 1

    identity = ArtifactIdentity(
        "rtdetr-smoke", "rtdetrv2_l", "End-to-end Transformer", "2class", 640, 42
    )
    paths = write_prediction_artifact(
        tmp_path / "predictions.v1.json",
        identity,
        read_json(FIXTURE / "predictions.json"),
        legacy_destination=tmp_path / "predictions.json",
    )
    metrics = detailed_metrics(FIXTURE / "ground_truth.json", paths["legacy"])
    assert metrics["per_class_detailed"]["person"]["true_positives"] == 1
