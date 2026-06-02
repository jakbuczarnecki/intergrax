# © Artur Czarnecki. All rights reserved.

from intergrax.model_inference.registry.core import ModelInferenceRegistry
from intergrax.model_inference.registry.vision_adapter_registry import VisionAdapterRegistry
from intergrax.model_inference.registry.vision_profile import VisionProfile, vision_profile_from_env
from intergrax.model_inference.registry.vision_provider import VisionProvider


def build_default_model_inference_registry() -> ModelInferenceRegistry:
    from intergrax.model_inference.bootstrap import build_harness_model_inference_registry

    return build_harness_model_inference_registry()


__all__ = [
    "ModelInferenceRegistry",
    "VisionAdapterRegistry",
    "VisionProfile",
    "VisionProvider",
    "build_default_model_inference_registry",
    "vision_profile_from_env",
]
