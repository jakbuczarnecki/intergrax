# © Artur Czarnecki. All rights reserved.

"""Bootstrap helpers for Plane C inference registries (Phase W-ML.3–W-ML.5)."""

from __future__ import annotations

import os

from intergrax.model_inference.adapters.opencv_vision import OpenCvVisionInferenceAdapter
from intergrax.model_inference.adapters.remote_serving import (
    MlInferenceHostAdapter,
    TritonVisionServingAdapter,
)
from intergrax.model_inference.adapters.stub_ml import StubModelInferenceAdapter
from intergrax.model_inference.adapters.stub_vision import StubVisionInferenceAdapter
from intergrax.model_inference.adapters.ultralytics_vision import UltralyticsVisionInferenceAdapter
from intergrax.model_inference.contracts import ModelArtifact, ModelArtifactFormat
from intergrax.model_inference.registry import ModelInferenceRegistry


def build_harness_model_inference_registry() -> ModelInferenceRegistry:
    """
    Build the harness-default inference registry.

    Vision adapter selection via ``INTERGRAX_VISION_ADAPTER``:
    ``stub`` | ``onnxruntime`` (OpenCV contours) | ``yolo_ultralytics``.
    """
    registry = ModelInferenceRegistry()
    vision_mode = (os.getenv("INTERGRAX_VISION_ADAPTER") or "onnxruntime").strip().lower()
    if vision_mode == "stub":
        vision_adapter = StubVisionInferenceAdapter()
        artifact_id = "vision.stub.yolo"
    elif vision_mode == "yolo_ultralytics":
        vision_adapter = UltralyticsVisionInferenceAdapter()
        artifact_id = "vision.yolo.ultralytics"
    else:
        vision_adapter = OpenCvVisionInferenceAdapter()
        artifact_id = "vision.opencv.onnx"

    registry.register_vision_adapter(vision_adapter)
    registry.register_vision_adapter(TritonVisionServingAdapter())
    registry.register_vision_adapter(MlInferenceHostAdapter())
    registry.register_ml_adapter(StubModelInferenceAdapter())
    registry.register_artifact(
        ModelArtifact(
            artifact_id=artifact_id,
            slug=vision_adapter.slug,
            format=ModelArtifactFormat.ONNX if vision_mode == "onnxruntime" else ModelArtifactFormat.ULTRALYTICS,
            metadata={"engine": vision_mode},
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


# Backward-compatible alias used by tools and tests.
build_default_model_inference_registry = build_harness_model_inference_registry
