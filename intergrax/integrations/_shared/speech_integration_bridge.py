# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bridge ``SpeechProviderBackend`` integration contracts to ``SpeechAdapter`` tools."""

from __future__ import annotations

from intergrax.integrations.contracts.speech_provider import SpeechProviderBackend
from intergrax.speech_adapters.contracts.io import (
    SpeechSynthesizeInput,
    SpeechSynthesizeOutput,
    SpeechTranscribeInput,
    SpeechTranscribeOutput,
)
from intergrax.speech_adapters.contracts.speech_adapter import SpeechAdapter
from intergrax.speech_adapters.contracts.speech_provider import SpeechProvider


class IntegrationSpeechAdapter(SpeechAdapter):
    """Wrap catalog ``SpeechProviderBackend`` for Tier-0 speech tools."""

    def __init__(self, backend: SpeechProviderBackend, *, provider: SpeechProvider) -> None:
        self._backend = backend
        self.provider = provider

    def synthesize(self, payload: SpeechSynthesizeInput) -> SpeechSynthesizeOutput:
        result = self._backend.synthesize(payload.text, voice_id=payload.voice_id)
        return SpeechSynthesizeOutput(
            audio_uri=result.audio_uri,
            character_count=result.character_count,
        )

    def transcribe(self, payload: SpeechTranscribeInput) -> SpeechTranscribeOutput:
        result = self._backend.transcribe(payload.audio_uri)
        return SpeechTranscribeOutput(
            transcript=result.transcript,
            duration_ms=result.duration_ms,
        )


def speech_provider_for_slug(slug: str) -> SpeechProvider:
    normalized = slug.strip().lower()
    if normalized == "elevenlabs":
        return SpeechProvider.ELEVENLABS
    return SpeechProvider.STUB
