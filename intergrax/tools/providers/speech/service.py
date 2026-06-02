# © Artur Czarnecki. All rights reserved.

"""Harness speech tool services (Phase W-ML.2)."""

from __future__ import annotations

from intergrax.tools.providers.speech.backends import SPEECH_BACKEND_EXTRA_KEY, build_speech_backend
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
    return _resolve_backend(ctx).synthesize(payload)


def speech_transcribe(ctx: ToolWiringContext, payload: SpeechTranscribeInput) -> SpeechTranscribeOutput:
    return _resolve_backend(ctx).transcribe(payload)
