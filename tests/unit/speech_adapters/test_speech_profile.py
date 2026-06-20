from __future__ import annotations

import pytest

from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.speech_adapters import SpeechProfile, speech_profile_from_env
from intergrax.speech_adapters.contracts.io import SpeechSynthesizeInput
from intergrax.speech_adapters.providers.elevenlabs_speech import ElevenLabsSpeechAdapter
from intergrax.speech_adapters.providers.stub_speech import StubSpeechAdapter
from intergrax.speech_adapters.registry.speech_adapter_registry import SpeechAdapterRegistry
from intergrax.integrations._shared.speech_integration_bridge import IntegrationSpeechAdapter


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


def test_speech_profile_create_stub_adapter() -> None:
    profile = SpeechProfile(provider_slug="stub")
    adapter = profile.create_adapter()
    assert isinstance(adapter, StubSpeechAdapter)
    assert adapter.provider_slug == "stub"


def test_speech_profile_create_elevenlabs_with_secrets() -> None:
    profile = SpeechProfile(provider_slug="elevenlabs")
    adapter = profile.create_adapter(secrets={"api_key": "test-key"})
    assert isinstance(adapter, ElevenLabsSpeechAdapter)


def test_speech_profile_elevenlabs_requires_api_key() -> None:
    profile = SpeechProfile(provider_slug="elevenlabs")
    with pytest.raises(ValueError, match="api_key"):
        profile.create_adapter()


def test_speech_profile_from_env_selects_elevenlabs_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.delenv("INTERGRAX_SPEECH_PROVIDER", raising=False)
    profile = speech_profile_from_env()
    assert profile.provider_slug == "elevenlabs"


def test_speech_adapter_registry_lists_builtin_providers() -> None:
    providers = SpeechAdapterRegistry.registered_providers()
    assert "stub" in providers
    assert "elevenlabs" in providers


def test_speech_profile_resolves_deepgram_from_integration_catalog() -> None:
    register_default_integrations()

    class _FakeSpeech:
        def synthesize(self, text: str, *, voice_id: str = "default") -> dict[str, object]:
            return {"audio_uri": "speech://audio/1", "character_count": len(text)}

        def transcribe(self, audio_uri: str) -> dict[str, object]:
            return {"transcript": audio_uri, "duration_ms": 100}

        def health(self) -> bool:
            return True

    from intergrax.integrations.providers.speech_provider.deepgram.bundle import create_deepgram_speech_provider

    profile = SpeechProfile.from_backend(
        create_deepgram_speech_provider(client=_FakeSpeech()),
        provider_slug="deepgram",
    )
    adapter = profile.create_adapter()
    assert isinstance(adapter, IntegrationSpeechAdapter)
    assert adapter.provider_slug == "deepgram"
    output = adapter.synthesize(SpeechSynthesizeInput(text="hi"))
    assert output.audio_uri


def test_speech_profile_catalog_slug_deepgram() -> None:
    register_default_integrations()

    class _FakeSpeech:
        def synthesize(self, text: str, *, voice_id: str = "default") -> dict[str, object]:
            return {"audio_uri": "speech://audio/2", "character_count": len(text)}

        def transcribe(self, audio_uri: str) -> dict[str, object]:
            return {"transcript": audio_uri, "duration_ms": 50}

        def health(self) -> bool:
            return True

    from intergrax.integrations.providers.speech_provider.deepgram.bundle import create_deepgram_speech_provider
    from intergrax.integrations.core.binding import IntegrationBinding

    profile = SpeechProfile.from_binding(
        IntegrationBinding.from_instance(create_deepgram_speech_provider(client=_FakeSpeech())),
        provider_slug="deepgram",
    )
    adapter = profile.create_adapter()
    assert isinstance(adapter, IntegrationSpeechAdapter)
    assert adapter.provider_slug == "deepgram"


def test_speech_adapter_registry_external_slug() -> None:
    class _CustomAdapter(StubSpeechAdapter):
        provider_slug = "acme_speech"

    SpeechAdapterRegistry.register("acme_speech", lambda: _CustomAdapter())
    adapter = SpeechAdapterRegistry.create("acme_speech")
    assert adapter.provider_slug == "acme_speech"
