# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.speech.contracts import (
    SpeechSynthesizeInput,
    SpeechSynthesizeOutput,
    SpeechTranscribeInput,
    SpeechTranscribeOutput,
)
from intergrax.tools.providers.speech.handlers import SpeechSynthesizeHandler, SpeechTranscribeHandler
from intergrax.tools.providers.speech.service import SPEECH_SYNTHESIZE_TOOL_ID, SPEECH_TRANSCRIBE_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

SPEECH_BUNDLE_ID = "speech"


def register_speech_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
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
