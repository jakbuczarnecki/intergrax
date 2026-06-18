from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.model_inference.adapters.opencv_vision import OpenCvVisionInferenceAdapter
from intergrax.model_inference.bootstrap import build_harness_model_inference_registry
from intergrax.model_inference.contracts import VisionInferenceRequest
from intergrax.model_inference.opencv_availability import opencv_runtime_available

pytestmark = pytest.mark.unit

if not opencv_runtime_available():
    pytestmark = pytest.mark.skip(reason="opencv-python-headless runtime unavailable")

_FIXTURE_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "vision_golden"
_GOLDEN_IMAGE = _FIXTURE_DIR / "sample_target.png"


@pytest.fixture(scope="module", autouse=True)
def _ensure_golden_fixture() -> None:
    if _GOLDEN_IMAGE.is_file():
        return
    import cv2
    import numpy as np

    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 80), 255, -1)
    cv2.imwrite(str(_GOLDEN_IMAGE), image)


def test_opencv_adapter_detects_white_rectangle() -> None:
    adapter = OpenCvVisionInferenceAdapter()
    artifact = build_harness_model_inference_registry().get_artifact("vision.opencv.onnx")
    media_uri = _GOLDEN_IMAGE.resolve().as_uri()
    result = adapter.detect(
        VisionInferenceRequest(
            request_id="golden-1",
            artifact_id=artifact.artifact_id,
            media_uri=media_uri,
            top_k=3,
        ),
        artifact=artifact,
    )
    assert result.detections
    assert result.detections[0].label == "contour.region"
    assert result.detections[0].confidence > 0.0
