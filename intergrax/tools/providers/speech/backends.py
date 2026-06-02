# © Artur Czarnecki. All rights reserved.

"""Backward-compatible speech backend shims — prefer ``intergrax.speech_adapters``."""

from __future__ import annotations

from intergrax.speech_adapters import SpeechAdapter, SpeechProfile, speech_profile_from_env
from intergrax.speech_adapters.providers.elevenlabs_speech import ElevenLabsSpeechAdapter
from intergrax.speech_adapters.providers.stub_speech import StubSpeechAdapter
from intergrax.speech_adapters.registry.profile import SPEECH_PROFILE_EXTRA_KEY

# Legacy type aliases
SpeechBackend = SpeechAdapter
StubSpeechBackend = StubSpeechAdapter
ElevenLabsSpeechBackend = ElevenLabsSpeechAdapter


def build_speech_backend(*, profile: SpeechProfile | None = None) -> SpeechAdapter:
    resolved = profile or speech_profile_from_env()
    return resolved.create_adapter()


SPEECH_BACKEND_EXTRA_KEY = "speech_backend"
MODEL_INFERENCE_REGISTRY_EXTRA_KEY = "model_inference_registry"
