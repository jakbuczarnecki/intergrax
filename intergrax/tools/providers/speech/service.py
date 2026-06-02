# © Artur Czarnecki. All rights reserved.

"""Harness speech tool services (Phase W-ML.2)."""

from __future__ import annotations

from intergrax.runtime.observability.modality_counters import record_media_bytes, record_tts_characters
from intergrax.tools.providers.speech.backends import SPEECH_BACKEND_EXTRA_KEY, build_speech_backend
from intergrax.tools.providers.vision.inference_support import measure_media_bytes
from intergrax.tools.providers.speech.contracts import (
    SpeechSynthesizeInput,
    SpeechSynthesizeOutput,
    SpeechTranscribeInput,
    SpeechTranscribeOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

SPEECH_SYNTHESIZE_TOOL_ID = "speech.synthesize"
SPEECH_TRANSCRIBE_TOOL_ID = "speech.transcribe"


def _resolve_backend(ctx: ToolWiringContext):
    backend = ctx.extras.get(SPEECH_BACKEND_EXTRA_KEY)
    if backend is None:
        return build_speech_backend()
    return backend


def speech_synthesize(ctx: ToolWiringContext, payload: SpeechSynthesizeInput) -> SpeechSynthesizeOutput:
    result = _resolve_backend(ctx).synthesize(payload)
    record_tts_characters(ctx, result.character_count)
    record_media_bytes(ctx, measure_media_bytes(result.audio_uri))
    return result


def speech_transcribe(ctx: ToolWiringContext, payload: SpeechTranscribeInput) -> SpeechTranscribeOutput:
    record_media_bytes(ctx, measure_media_bytes(payload.audio_uri))
    return _resolve_backend(ctx).transcribe(payload)
