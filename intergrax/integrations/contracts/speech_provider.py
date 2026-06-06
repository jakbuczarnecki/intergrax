# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Speech provider integration contract (Phase M.6 P6)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class SpeechSynthesisResult(BaseModel):
    """TTS output for harness ``speech.synthesize`` tool bridge."""

    audio_uri: str
    character_count: int = Field(ge=0, default=0)


class SpeechTranscriptionResult(BaseModel):
    """STT output for harness ``speech.transcribe`` tool bridge."""

    transcript: str
    duration_ms: int = Field(ge=0, default=0)


@runtime_checkable
class SpeechProviderBackend(Protocol):
    """Unified TTS/STT catalog facade bridging ``speech_adapters/``."""

    def synthesize(self, text: str, *, voice_id: str = "default") -> SpeechSynthesisResult:
        """Convert text to speech audio reference."""

    def transcribe(self, audio_uri: str) -> SpeechTranscriptionResult:
        """Transcribe audio at ``audio_uri`` to text."""
