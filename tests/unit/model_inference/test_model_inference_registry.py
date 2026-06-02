from __future__ import annotations

from intergrax.model_inference import build_default_model_inference_registry
from intergrax.model_inference.contracts import InferenceRequest, VisionInferenceRequest


def test_default_registry_runs_stub_vision_and_ml() -> None:
    registry = build_default_model_inference_registry()
    artifact = registry.get_artifact("vision.stub.yolo")
    vision = registry.get_vision_adapter("yolo_ultralytics")
    result = vision.detect(
        VisionInferenceRequest(
            request_id="req-1",
            artifact_id=artifact.artifact_id,
            media_uri="file:///tmp/x.png",
        ),
        artifact=artifact,
    )
    assert result.detections

    ml_artifact = registry.get_artifact("ml.stub.classifier")
    ml = registry.get_ml_adapter("sklearn_classifier")
    prediction = ml.predict(
        InferenceRequest(request_id="req-2", artifact_id=ml_artifact.artifact_id, features={"a": 1.0}),
        artifact=ml_artifact,
    )
    assert prediction.predictions
