# © Artur Czarnecki. All rights reserved.

"""Synchronous in-process modality inference executor."""

from __future__ import annotations

from intergrax.model_inference.contracts import (
    ExtendedVisionInferenceAdapter,
    InferenceRequest,
    InferenceResult,
    ModelArtifact,
    ModelInferenceAdapter,
    VisionInferenceAdapter,
    VisionInferenceRequest,
    VisionInferenceResult,
    VisionOcrResult,
    VisionSegmentationResult,
)
from intergrax.model_inference.execution.executor import ModalityInferenceExecutor
from intergrax.model_inference.registry import ModelInferenceRegistry


class InProcessModalityInferenceExecutor(ModalityInferenceExecutor):
    """Default executor — calls adapters on the caller thread."""

    def run_detect(
        self,
        *,
        registry: ModelInferenceRegistry,
        adapter: VisionInferenceAdapter,
        artifact: ModelArtifact,
        request: VisionInferenceRequest,
    ) -> VisionInferenceResult:
        _ = registry
        return adapter.detect(request, artifact=artifact)

    def run_segment(
        self,
        *,
        registry: ModelInferenceRegistry,
        adapter: VisionInferenceAdapter,
        artifact: ModelArtifact,
        request: VisionInferenceRequest,
    ) -> VisionSegmentationResult:
        _ = registry
        if not isinstance(adapter, ExtendedVisionInferenceAdapter):
            raise TypeError(f"Adapter {adapter.slug!r} does not support segmentation")
        return adapter.segment(request, artifact=artifact)

    def run_ocr_regions(
        self,
        *,
        registry: ModelInferenceRegistry,
        adapter: VisionInferenceAdapter,
        artifact: ModelArtifact,
        request: VisionInferenceRequest,
    ) -> VisionOcrResult:
        _ = registry
        if not isinstance(adapter, ExtendedVisionInferenceAdapter):
            raise TypeError(f"Adapter {adapter.slug!r} does not support OCR regions")
        return adapter.ocr_regions(request, artifact=artifact)

    def run_predict(
        self,
        *,
        registry: ModelInferenceRegistry,
        adapter: ModelInferenceAdapter,
        artifact: ModelArtifact,
        request: InferenceRequest,
    ) -> InferenceResult:
        _ = registry
        return adapter.predict(request, artifact=artifact)
