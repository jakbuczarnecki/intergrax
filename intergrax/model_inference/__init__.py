# © Artur Czarnecki. All rights reserved.

"""Dedicated model inference plane (Phase W-ML.3–W-ML.5)."""

from intergrax.model_inference.contracts import (
    InferenceRequest,
    InferenceResult,
    ModelArtifact,
    ModelInferenceAdapter,
    VisionBoundingBox,
    VisionDetection,
    VisionInferenceAdapter,
    VisionInferenceRequest,
    VisionInferenceResult,
)
from intergrax.model_inference.registry import (
    ModelInferenceRegistry,
    VisionProfile,
    VisionProvider,
    build_default_model_inference_registry,
    vision_profile_from_env,
)

__all__ = [
    "InferenceRequest",
    "InferenceResult",
    "ModelArtifact",
    "ModelInferenceAdapter",
    "ModelInferenceRegistry",
    "VisionProfile",
    "VisionProvider",
    "build_default_model_inference_registry",
    "vision_profile_from_env",
    "VisionBoundingBox",
    "VisionDetection",
    "VisionInferenceAdapter",
    "VisionInferenceRequest",
    "VisionInferenceResult",
]
