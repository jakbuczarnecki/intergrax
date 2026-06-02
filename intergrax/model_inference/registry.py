# © Artur Czarnecki. All rights reserved.

"""Registry for vision and classical ML inference adapters (Phase W-ML.3–W-ML.5)."""

from __future__ import annotations

from intergrax.model_inference.adapters.stub_ml import StubModelInferenceAdapter
from intergrax.model_inference.adapters.stub_vision import StubVisionInferenceAdapter
from intergrax.model_inference.contracts import (
    ModelArtifact,
    ModelArtifactFormat,
    ModelInferenceAdapter,
    VisionInferenceAdapter,
)


class ModelInferenceRegistry:
    """Typed registry for Plane C inference adapters."""

    def __init__(self) -> None:
        self._vision_adapters: dict[str, VisionInferenceAdapter] = {}
        self._ml_adapters: dict[str, ModelInferenceAdapter] = {}
        self._artifacts: dict[str, ModelArtifact] = {}

    def register_vision_adapter(self, adapter: VisionInferenceAdapter) -> None:
        self._vision_adapters[adapter.slug] = adapter

    def register_ml_adapter(self, adapter: ModelInferenceAdapter) -> None:
        self._ml_adapters[adapter.slug] = adapter

    def register_artifact(self, artifact: ModelArtifact) -> None:
        self._artifacts[artifact.artifact_id] = artifact

    def get_vision_adapter(self, slug: str) -> VisionInferenceAdapter:
        adapter = self._vision_adapters.get(slug)
        if adapter is None:
            raise KeyError(f"Vision inference adapter not registered: {slug}")
        return adapter

    def get_ml_adapter(self, slug: str) -> ModelInferenceAdapter:
        adapter = self._ml_adapters.get(slug)
        if adapter is None:
            raise KeyError(f"ML inference adapter not registered: {slug}")
        return adapter

    def get_artifact(self, artifact_id: str) -> ModelArtifact:
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            raise KeyError(f"Model artifact not registered: {artifact_id}")
        return artifact


def build_default_model_inference_registry() -> ModelInferenceRegistry:
    registry = ModelInferenceRegistry()
    registry.register_vision_adapter(StubVisionInferenceAdapter())
    registry.register_ml_adapter(StubModelInferenceAdapter())
    registry.register_artifact(
        ModelArtifact(
            artifact_id="vision.stub.yolo",
            slug="yolo_ultralytics",
            format=ModelArtifactFormat.ULTRALYTICS,
            metadata={"engine": "stub"},
        )
    )
    registry.register_artifact(
        ModelArtifact(
            artifact_id="ml.stub.classifier",
            slug="sklearn_classifier",
            format=ModelArtifactFormat.SKLEARN,
            metadata={"engine": "stub"},
        )
    )
    return registry
