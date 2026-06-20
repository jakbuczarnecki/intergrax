# © Artur Czarnecki. All rights reserved.

"""Shared Tier-3 modality profile materialization (VisionProfile + SpeechProfile)."""

from __future__ import annotations

from intergrax.applications._shared.modality_celery_wiring import (
    MODALITY_CELERY_BUNDLE_EXTRA_KEY,
    resolve_modality_celery_bundle,
)
from intergrax.model_inference.execution import (
    MODALITY_EXECUTION_PROFILE_EXTRA_KEY,
    MODALITY_EXECUTOR_EXTRA_KEY,
    ModalityExecutionProfile,
    build_modality_inference_executor,
    modality_execution_profile_from_env,
)
from intergrax.model_inference.registry import VisionProfile, vision_profile_from_env
from intergrax.model_inference.registry.vision_profile import VISION_PROFILE_EXTRA_KEY
from intergrax.runtime.modality.modality_profile import MODALITY_PROFILE_EXTRA_KEY, ModalityProfile
from intergrax.speech_adapters.registry.profile import SPEECH_PROFILE_EXTRA_KEY, SpeechProfile, speech_profile_from_env
from intergrax.tools.providers.speech.backends import MODEL_INFERENCE_REGISTRY_EXTRA_KEY, SPEECH_BACKEND_EXTRA_KEY
from intergrax.tools.registry.wiring import ToolWiringContext


def wire_modality_extras(
    ctx: ToolWiringContext,
    *,
    modality_profile: ModalityProfile | None = None,
    vision_profile: VisionProfile | None = None,
    speech_profile: SpeechProfile | None = None,
    execution_profile: ModalityExecutionProfile | None = None,
) -> None:
    """Populate tool wiring extras with typed modality profiles and live adapters."""
    resolved_vision = vision_profile or vision_profile_from_env()
    resolved_speech = speech_profile or speech_profile_from_env()
    resolved_execution = execution_profile or modality_execution_profile_from_env()
    if modality_profile is not None:
        ctx.extras[MODALITY_PROFILE_EXTRA_KEY] = modality_profile
    ctx.extras[VISION_PROFILE_EXTRA_KEY] = resolved_vision
    ctx.extras[SPEECH_PROFILE_EXTRA_KEY] = resolved_speech
    celery_bundle = resolve_modality_celery_bundle(resolved_execution)
    if celery_bundle is not None:
        ctx.extras[MODALITY_CELERY_BUNDLE_EXTRA_KEY] = celery_bundle
    ctx.extras[MODALITY_EXECUTION_PROFILE_EXTRA_KEY] = resolved_execution
    ctx.extras[MODALITY_EXECUTOR_EXTRA_KEY] = build_modality_inference_executor(resolved_execution)
    if SPEECH_BACKEND_EXTRA_KEY not in ctx.extras and ctx.speech_provider is None:
        ctx.extras[SPEECH_BACKEND_EXTRA_KEY] = resolved_speech.create_adapter()
    ctx.extras[MODEL_INFERENCE_REGISTRY_EXTRA_KEY] = resolved_vision.build_registry()
