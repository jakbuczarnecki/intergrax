# © Artur Czarnecki. All rights reserved.

"""Bootstrap helpers for Plane C inference registries (Phase W-ML.3–W-ML.5)."""

from __future__ import annotations

from intergrax.model_inference.registry import ModelInferenceRegistry, VisionProfile, vision_profile_from_env


def build_harness_model_inference_registry(
    *,
    profile: VisionProfile | None = None,
) -> ModelInferenceRegistry:
    """Build registry from :class:`VisionProfile` (env-backed by default)."""
    resolved = profile or vision_profile_from_env()
    return resolved.build_registry()


# Backward-compatible alias used by tools and tests.
build_default_model_inference_registry = build_harness_model_inference_registry
