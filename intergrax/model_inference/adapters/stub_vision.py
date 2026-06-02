# © Artur Czarnecki. All rights reserved.

"""Stub vision inference adapter for harness contract tests (Phase W-ML.3)."""

from __future__ import annotations

from intergrax.model_inference.contracts import (
    ExtendedVisionInferenceAdapter,
    ModelArtifact,
    VisionBoundingBox,
    VisionDetection,
    VisionInferenceRequest,
    VisionInferenceResult,
    VisionOcrRegion,
    VisionOcrResult,
    VisionSegment,
    VisionSegmentationResult,
)


class StubVisionInferenceAdapter(ExtendedVisionInferenceAdapter):
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

    def segment(self, request: VisionInferenceRequest, *, artifact: ModelArtifact) -> VisionSegmentationResult:
        return VisionSegmentationResult(
            request_id=request.request_id,
            artifact_id=artifact.artifact_id,
            segments=[
                VisionSegment(
                    label="segment.stub",
                    confidence=0.95,
                    bbox=VisionBoundingBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9),
                )
            ],
        )

    def ocr_regions(self, request: VisionInferenceRequest, *, artifact: ModelArtifact) -> VisionOcrResult:
        return VisionOcrResult(
            request_id=request.request_id,
            artifact_id=artifact.artifact_id,
            regions=[
                VisionOcrRegion(
                    text="[stub ocr]",
                    confidence=0.9,
                    bbox=VisionBoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=0.2),
                )
            ],
        )
