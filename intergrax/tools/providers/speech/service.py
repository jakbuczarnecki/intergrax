# © Artur Czarnecki. All rights reserved.

"""Harness speech tool services (Phase W-ML.2 stub implementation)."""

from __future__ import annotations

from intergrax.tools.providers.speech.contracts import (
    SpeechSynthesizeInput,
    SpeechSynthesizeOutput,
    SpeechTranscribeInput,
    SpeechTranscribeOutput,
)

SPEECH_SYNTHESIZE_TOOL_ID = "speech.synthesize"
SPEECH_TRANSCRIBE_TOOL_ID = "speech.transcribe"


def speech_synthesize(payload: SpeechSynthesizeInput) -> SpeechSynthesizeOutput:
    return SpeechSynthesizeOutput(
        audio_uri=f"stub://speech/{payload.voice_id}.wav",
        character_count=len(payload.text),
    )


def speech_transcribe(payload: SpeechTranscribeInput) -> SpeechTranscribeOutput:
    return SpeechTranscribeOutput(
        transcript=f"[stub transcript for {payload.audio_uri}]",
        duration_ms=1000,
    )
