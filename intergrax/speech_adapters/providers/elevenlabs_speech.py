# © Artur Czarnecki. All rights reserved.

"""ElevenLabs speech synthesis adapter."""

from __future__ import annotations

import httpx

from intergrax.speech_adapters.contracts.io import (
    SpeechSynthesizeInput,
    SpeechSynthesizeOutput,
    SpeechTranscribeInput,
    SpeechTranscribeOutput,
)
from intergrax.speech_adapters.contracts.speech_adapter import SpeechAdapter


class ElevenLabsSpeechAdapter(SpeechAdapter):
    """ElevenLabs REST TTS (requires ``api_key`` or ``ELEVENLABS_API_KEY`` env)."""

    ENV_API_KEY = "ELEVENLABS_API_KEY"
    DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

    @property
    def provider_slug(self) -> str:
        return "elevenlabs"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.elevenlabs.io/v1",
        default_voice_id: str | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("ElevenLabsSpeechAdapter requires a non-empty api_key")
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._default_voice_id = default_voice_id or self.DEFAULT_VOICE_ID

    def synthesize(self, payload: SpeechSynthesizeInput) -> SpeechSynthesizeOutput:
        voice_id = payload.voice_id if payload.voice_id != "default" else self._default_voice_id
        url = f"{self._base_url}/text-to-speech/{voice_id}"
        headers = {"xi-api-key": self._api_key, "accept": "audio/mpeg"}
        body = {"text": payload.text, "model_id": "eleven_multilingual_v2"}
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
        return SpeechSynthesizeOutput(
            audio_uri=f"elevenlabs://audio/{voice_id}",
            character_count=len(payload.text),
        )

    def transcribe(self, payload: SpeechTranscribeInput) -> SpeechTranscribeOutput:
        raise NotImplementedError("ElevenLabs transcription is not configured in harness path")
