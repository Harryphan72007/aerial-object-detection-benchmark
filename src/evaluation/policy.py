"""Single detection-output policy shared by training and evaluation."""

from __future__ import annotations

from typing import Any

DETECTION_SCORE_THRESHOLD = 0.001
MAX_DETECTIONS_PER_IMAGE = 500
EVALUATION_POLICY_VERSION = "aerial_detection_policy_v1"


def detection_policy() -> dict[str, Any]:
    """Return a serializable copy of the immutable prediction policy."""

    return {
        "version": EVALUATION_POLICY_VERSION,
        "score_threshold": DETECTION_SCORE_THRESHOLD,
        "max_detections_per_image": MAX_DETECTIONS_PER_IMAGE,
    }
