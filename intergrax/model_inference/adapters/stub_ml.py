# © Artur Czarnecki. All rights reserved.

"""Stub classical ML inference adapter for harness contract tests (Phase W-ML.5)."""

from __future__ import annotations

from intergrax.model_inference.contracts import (
    InferenceRequest,
    InferenceResult,
    ModelArtifact,
    ModelInferenceAdapter,
)


class StubModelInferenceAdapter(ModelInferenceAdapter):
    slug = "sklearn_classifier"

    def predict(self, request: InferenceRequest, *, artifact: ModelArtifact) -> InferenceResult:
        score = sum(request.features.values()) / float(max(1, len(request.features)))
        return InferenceResult(
            request_id=request.request_id,
            artifact_id=artifact.artifact_id,
            predictions={"positive": min(1.0, score)},
        )
