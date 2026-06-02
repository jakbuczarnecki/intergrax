# © Artur Czarnecki. All rights reserved.

"""Speech synthesis/transcription backends (Phase W-ML.2)."""

from __future__ import annotations

import os
from typing import Protocol

import httpx

from intergrax.tools.providers.speech.contracts import (
    SpeechSynthesizeInput,
    SpeechSynthesizeOutput,
    SpeechTranscribeInput,
    SpeechTranscribeOutput,
)


class SpeechBackend(Protocol):
    """Typed speech provider surface for tool handlers."""

    def synthesize(self, payload: SpeechSynthesizeInput) -> SpeechSynthesizeOutput:
        ...

    def transcribe(self, payload: SpeechTranscribeInput) -> SpeechTranscribeOutput:
        ...


class StubSpeechBackend:
    """Harness stub backend used when no vendor API key is configured."""

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


class ElevenLabsSpeechBackend:
    """ElevenLabs REST integration (requires ``ELEVENLABS_API_KEY``)."""

    def __init__(self, *, api_key: str, base_url: str = "https://api.elevenlabs.io/v1") -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def synthesize(self, payload: SpeechSynthesizeInput) -> SpeechSynthesizeOutput:
        voice_id = payload.voice_id if payload.voice_id != "default" else "21m00Tcm4TlvDq8ikWAM"
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
        raise NotImplementedError("ElevenLabs transcription is not configured in harness stub path")


def build_speech_backend() -> SpeechBackend:
    api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
    if api_key:
        return ElevenLabsSpeechBackend(api_key=api_key)
    return StubSpeechBackend()


SPEECH_BACKEND_EXTRA_KEY = "speech_backend"
MODEL_INFERENCE_REGISTRY_EXTRA_KEY = "model_inference_registry"
