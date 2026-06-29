# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deepgram speech provider integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.speech_provider import SpeechProviderBackend
from intergrax.runtime.integrations.categories.ai import SpeechProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

DEEPGRAM_SPEECH_PROVIDER_PROVIDER_ID = "deepgram"


class DeepgramSpeechProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Deepgram speech provider integration."""

    pass


DeepgramSpeechProviderClient = SpeechProviderBackend

class DeepgramSpeechProviderIntegration(SpeechProviderIntegrationContract):
    """
    Single public Deepgram speech provider entrypoint.

    Legacy catalog factory (create_deepgram_speech_provider) owns catalog behavior; legacy factories use from_client().
    """

    config: DeepgramSpeechProviderIntegrationConfig = DeepgramSpeechProviderIntegrationConfig()
    _client: DeepgramSpeechProviderClient | None = PrivateAttr(default=None)
    

    def synthesize(self, text, voice_id: str = 'default'):
        return self._require_client().synthesize(text, voice_id=voice_id)

    def transcribe(self, audio_uri):
        return self._require_client().transcribe(audio_uri)

    def _require_client(self) -> SpeechProviderBackend:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: DeepgramSpeechProviderClient,
        *,
        enabled: bool = False,
    ) -> DeepgramSpeechProviderIntegration:
        integration = cls.for_provider(
            provider_id=DEEPGRAM_SPEECH_PROVIDER_PROVIDER_ID,
            display_name="Deepgram",
            config=DeepgramSpeechProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> DeepgramSpeechProviderClient | None:
        return self._client

SpeechProviderBackend.register(DeepgramSpeechProviderIntegration)
