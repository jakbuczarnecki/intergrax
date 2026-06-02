# © Artur Czarnecki. All rights reserved.

"""Shared vision tool wiring helpers."""

from __future__ import annotations

from intergrax.model_inference.contracts import ExtendedVisionInferenceAdapter, VisionInferenceAdapter
from intergrax.model_inference.registry import ModelInferenceRegistry
from intergrax.model_inference.adapters.stub_vision import StubVisionInferenceAdapter
from intergrax.runtime.modality.modality_profile import MODALITY_PROFILE_EXTRA_KEY, ModalityProfile
from intergrax.tools.providers.speech.backends import MODEL_INFERENCE_REGISTRY_EXTRA_KEY
from intergrax.tools.registry.wiring import ToolWiringContext


def resolve_registry(ctx: ToolWiringContext) -> ModelInferenceRegistry:
    registry = ctx.extras.get(MODEL_INFERENCE_REGISTRY_EXTRA_KEY)
    if registry is None:
        from intergrax.model_inference.bootstrap import build_harness_model_inference_registry

        return build_harness_model_inference_registry()
    return registry


def resolve_modality_profile(ctx: ToolWiringContext) -> ModalityProfile | None:
    raw = ctx.extras.get(MODALITY_PROFILE_EXTRA_KEY)
    if isinstance(raw, ModalityProfile):
        return raw
    return None


def assert_artifact_allowed(profile: ModalityProfile | None, artifact_id: str) -> None:
    if profile is None:
        return
    if profile.vision_model_ids and artifact_id not in profile.vision_model_ids:
        raise ValueError(f"artifact_id {artifact_id!r} not allowed by ModalityProfile {profile.profile_id!r}")


def as_extended_adapter(adapter: VisionInferenceAdapter) -> ExtendedVisionInferenceAdapter:
    if isinstance(adapter, ExtendedVisionInferenceAdapter):
        return adapter
    return StubVisionInferenceAdapter()
