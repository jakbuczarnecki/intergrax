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


class IntegrationSpeechAdapter(SpeechAdapter):
    """Wrap catalog ``SpeechProviderBackend`` for Tier-0 speech tools."""

    def __init__(self, backend: SpeechProviderBackend, *, provider_slug: str) -> None:
        self._backend = backend
        self.provider_slug = provider_slug.strip().lower()

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


def infer_speech_provider_slug(backend: SpeechProviderBackend) -> str | None:
    """Best-effort slug for pre-built ``SpeechProviderBackend`` instances."""
    from intergrax.integrations._shared.p7.clients import HttpSpeechProviderBackend, SpeechAdapterBackend

    if isinstance(backend, SpeechAdapterBackend):
        return backend._slug
    if isinstance(backend, HttpSpeechProviderBackend):
        return backend._provider
    try:
        status = backend.health()
    except Exception:
        return None
    slug = getattr(status, "slug", None)
    return str(slug) if slug else None
