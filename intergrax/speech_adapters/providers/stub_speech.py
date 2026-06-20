# © Artur Czarnecki. All rights reserved.

"""Stub speech adapter for harness runs without external vendors."""

from __future__ import annotations

from intergrax.speech_adapters.contracts.io import (
    SpeechSynthesizeInput,
    SpeechSynthesizeOutput,
    SpeechTranscribeInput,
    SpeechTranscribeOutput,
)
from intergrax.speech_adapters.contracts.speech_adapter import SpeechAdapter


class StubSpeechAdapter(SpeechAdapter):
    @property
    def provider_slug(self) -> str:
        return "stub"

    def synthesize(self, payload: SpeechSynthesizeInput) -> SpeechSynthesizeOutput:
        return SpeechSynthesizeOutput(
            audio_uri=f"stub://speech/{payload.voice_id}.wav",
            character_count=len(payload.text),
        )

    def transcribe(self, payload: SpeechTranscribeInput) -> SpeechTranscribeOutput:
        return SpeechTranscribeOutput(
            transcript=f"[stub transcript for {payload.audio_uri}]",
            duration_ms=1000,
        )
