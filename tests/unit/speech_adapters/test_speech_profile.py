from __future__ import annotations

import pytest

from intergrax.speech_adapters import SpeechProfile, SpeechProvider, speech_profile_from_env
from intergrax.speech_adapters.providers.elevenlabs_speech import ElevenLabsSpeechAdapter
from intergrax.speech_adapters.providers.stub_speech import StubSpeechAdapter
from intergrax.speech_adapters.registry.speech_adapter_registry import SpeechAdapterRegistry


def test_speech_profile_create_stub_adapter() -> None:
    profile = SpeechProfile(provider=SpeechProvider.STUB)
    adapter = profile.create_adapter()
    assert isinstance(adapter, StubSpeechAdapter)


def test_speech_profile_create_elevenlabs_with_secrets() -> None:
    profile = SpeechProfile(provider=SpeechProvider.ELEVENLABS)
    adapter = profile.create_adapter(secrets={"api_key": "test-key"})
    assert isinstance(adapter, ElevenLabsSpeechAdapter)


def test_speech_profile_elevenlabs_requires_api_key() -> None:
    profile = SpeechProfile(provider=SpeechProvider.ELEVENLABS)
    with pytest.raises(ValueError, match="api_key"):
        profile.create_adapter()


def test_speech_profile_from_env_selects_elevenlabs_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.delenv("INTERGRAX_SPEECH_PROVIDER", raising=False)
    profile = speech_profile_from_env()
    assert profile.provider == SpeechProvider.ELEVENLABS


def test_speech_adapter_registry_lists_builtin_providers() -> None:
    providers = SpeechAdapterRegistry.registered_providers()
    assert SpeechProvider.STUB.value in providers
    assert SpeechProvider.ELEVENLABS.value in providers
