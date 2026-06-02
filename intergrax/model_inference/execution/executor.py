# © Artur Czarnecki. All rights reserved.

"""Typed executor surface for modality inference jobs."""

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.model_inference.contracts import (
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
from intergrax.model_inference.registry import ModelInferenceRegistry


class ModalityInferenceExecutor(ABC):
    """Runs Plane C inference with optional background offload."""

    @abstractmethod
    def run_detect(
        self,
        *,
        registry: ModelInferenceRegistry,
        adapter: VisionInferenceAdapter,
        artifact: ModelArtifact,
        request: VisionInferenceRequest,
    ) -> VisionInferenceResult:
        raise NotImplementedError

    @abstractmethod
    def run_segment(
        self,
        *,
        registry: ModelInferenceRegistry,
        adapter: VisionInferenceAdapter,
        artifact: ModelArtifact,
        request: VisionInferenceRequest,
    ) -> VisionSegmentationResult:
        raise NotImplementedError

    @abstractmethod
    def run_ocr_regions(
        self,
        *,
        registry: ModelInferenceRegistry,
        adapter: VisionInferenceAdapter,
        artifact: ModelArtifact,
        request: VisionInferenceRequest,
    ) -> VisionOcrResult:
        raise NotImplementedError

    @abstractmethod
    def run_predict(
        self,
        *,
        registry: ModelInferenceRegistry,
        adapter: ModelInferenceAdapter,
        artifact: ModelArtifact,
        request: InferenceRequest,
    ) -> InferenceResult:
        raise NotImplementedError
