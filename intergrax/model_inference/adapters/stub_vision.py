# © Artur Czarnecki. All rights reserved.

"""Stub vision inference adapter for harness contract tests (Phase W-ML.3)."""

from __future__ import annotations

from intergrax.model_inference.contracts import (
    ModelArtifact,
    VisionBoundingBox,
    VisionDetection,
    VisionInferenceAdapter,
    VisionInferenceRequest,
    VisionInferenceResult,
)


class StubVisionInferenceAdapter(VisionInferenceAdapter):
    slug = "yolo_ultralytics"

    def detect(self, request: VisionInferenceRequest, *, artifact: ModelArtifact) -> VisionInferenceResult:
        return VisionInferenceResult(
            request_id=request.request_id,
            artifact_id=artifact.artifact_id,
            detections=[
                VisionDetection(
                    label="object.stub",
                    confidence=0.99,
                    bbox=VisionBoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0),
                )
            ],
        )
