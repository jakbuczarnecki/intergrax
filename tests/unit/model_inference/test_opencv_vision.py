from __future__ import annotations

import pytest

from intergrax.model_inference.adapters.opencv_vision import OpenCvVisionInferenceAdapter
from intergrax.model_inference.bootstrap import build_harness_model_inference_registry
from intergrax.model_inference.contracts import VisionInferenceRequest
from intergrax.model_inference.opencv_availability import opencv_runtime_available

pytestmark = pytest.mark.unit


@pytest.mark.skipif(not opencv_runtime_available(), reason="opencv-python-headless runtime unavailable")
def test_opencv_adapter_detects_white_rectangle(vision_golden_image) -> None:
    adapter = OpenCvVisionInferenceAdapter()
    artifact = build_harness_model_inference_registry().get_artifact("vision.opencv.onnx")
    media_uri = vision_golden_image.resolve().as_uri()
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
