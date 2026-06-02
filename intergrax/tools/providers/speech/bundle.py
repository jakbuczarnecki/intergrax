# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.speech.contracts import (
    SpeechSynthesizeInput,
    SpeechSynthesizeOutput,
    SpeechTranscribeInput,
    SpeechTranscribeOutput,
)
from intergrax.model_inference.registry import vision_profile_from_env
from intergrax.model_inference.registry.vision_profile import VISION_PROFILE_EXTRA_KEY
from intergrax.speech_adapters.registry.profile import SPEECH_PROFILE_EXTRA_KEY, speech_profile_from_env
from intergrax.tools.providers.speech.backends import (
    MODEL_INFERENCE_REGISTRY_EXTRA_KEY,
    SPEECH_BACKEND_EXTRA_KEY,
    build_speech_backend,
)
from intergrax.tools.providers.speech.handlers import SpeechSynthesizeHandler, SpeechTranscribeHandler
from intergrax.tools.providers.speech.service import SPEECH_SYNTHESIZE_TOOL_ID, SPEECH_TRANSCRIBE_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

SPEECH_BUNDLE_ID = "speech"


def register_speech_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    from intergrax.model_inference.registry import VisionProfile
    from intergrax.speech_adapters.registry.profile import SpeechProfile

    if SPEECH_BACKEND_EXTRA_KEY not in ctx.extras:
        raw_speech = ctx.extras.get(SPEECH_PROFILE_EXTRA_KEY)
        profile = raw_speech if isinstance(raw_speech, SpeechProfile) else speech_profile_from_env()
        ctx.extras[SPEECH_PROFILE_EXTRA_KEY] = profile
        ctx.extras[SPEECH_BACKEND_EXTRA_KEY] = profile.create_adapter()
    if MODEL_INFERENCE_REGISTRY_EXTRA_KEY not in ctx.extras:
        raw_vision = ctx.extras.get(VISION_PROFILE_EXTRA_KEY)
        vision = raw_vision if isinstance(raw_vision, VisionProfile) else vision_profile_from_env()
        ctx.extras[VISION_PROFILE_EXTRA_KEY] = vision
        ctx.extras[MODEL_INFERENCE_REGISTRY_EXTRA_KEY] = vision.build_registry()
    registry.register(
        ToolContract(
            tool_id=SPEECH_SYNTHESIZE_TOOL_ID,
            name=SPEECH_SYNTHESIZE_TOOL_ID,
            description="Synthesize speech audio from text (stub/provider-backed).",
            description_short="Text-to-speech synthesis.",
            input_schema=SpeechSynthesizeInput,
            output_schema=SpeechSynthesizeOutput,
            error_mapping={},
            side_effects=True,
            category="speech",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("speech", "tts", "modality"),
        ),
        SpeechSynthesizeHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=SPEECH_TRANSCRIBE_TOOL_ID,
            name=SPEECH_TRANSCRIBE_TOOL_ID,
            description="Transcribe speech audio to text (stub/provider-backed).",
            description_short="Speech-to-text transcription.",
            input_schema=SpeechTranscribeInput,
            output_schema=SpeechTranscribeOutput,
            error_mapping={},
            side_effects=False,
            category="speech",
            risk_level=ToolRiskLevel.LOW,
            tags=("speech", "stt", "modality"),
        ),
        SpeechTranscribeHandler(ctx),
    )
