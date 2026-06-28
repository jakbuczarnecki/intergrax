# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Elevenlabs speech provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.ai import SpeechProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

ELEVENLABS_SPEECH_PROVIDER_PROVIDER_ID = "elevenlabs"


class ElevenlabsSpeechProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Elevenlabs speech provider integration."""

    pass


@runtime_checkable
class ElevenlabsSpeechProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class ElevenlabsSpeechProviderIntegration(SpeechProviderIntegrationContract):
    """
    Elevenlabs speech provider integration.

    The legacy facade (create_elevenlabs_speech_provider) remains separate and backward-compatible.
    """

    config: ElevenlabsSpeechProviderIntegrationConfig = ElevenlabsSpeechProviderIntegrationConfig()
    _client: ElevenlabsSpeechProviderClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: ElevenlabsSpeechProviderClient,
        *,
        enabled: bool = False,
    ) -> ElevenlabsSpeechProviderIntegration:
        integration = cls.for_provider(
            provider_id=ELEVENLABS_SPEECH_PROVIDER_PROVIDER_ID,
            display_name="Elevenlabs",
            config=ElevenlabsSpeechProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ElevenlabsSpeechProviderClient | None:
        return self._client
