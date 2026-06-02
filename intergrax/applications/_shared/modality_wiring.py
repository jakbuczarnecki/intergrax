# © Artur Czarnecki. All rights reserved.

"""Shared Tier-3 modality profile materialization (VisionProfile + SpeechProfile)."""

from __future__ import annotations

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
) -> None:
    """Populate tool wiring extras with typed modality profiles and live adapters."""
    resolved_vision = vision_profile or vision_profile_from_env()
    resolved_speech = speech_profile or speech_profile_from_env()
    if modality_profile is not None:
        ctx.extras[MODALITY_PROFILE_EXTRA_KEY] = modality_profile
    ctx.extras[VISION_PROFILE_EXTRA_KEY] = resolved_vision
    ctx.extras[SPEECH_PROFILE_EXTRA_KEY] = resolved_speech
    ctx.extras[SPEECH_BACKEND_EXTRA_KEY] = resolved_speech.create_adapter()
    ctx.extras[MODEL_INFERENCE_REGISTRY_EXTRA_KEY] = resolved_vision.build_registry()
