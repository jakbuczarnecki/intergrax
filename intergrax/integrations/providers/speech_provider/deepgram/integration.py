# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deepgram speech provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.ai import SpeechProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

DEEPGRAM_SPEECH_PROVIDER_PROVIDER_ID = "deepgram"


class DeepgramSpeechProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Deepgram speech provider integration."""

    pass


@runtime_checkable
class DeepgramSpeechProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class DeepgramSpeechProviderIntegration(SpeechProviderIntegrationContract):
    """
    Deepgram speech provider integration.

    The legacy facade (create_deepgram_speech_provider) remains separate and backward-compatible.
    """

    config: DeepgramSpeechProviderIntegrationConfig = DeepgramSpeechProviderIntegrationConfig()
    _client: DeepgramSpeechProviderClient | None = PrivateAttr(default=None)

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
