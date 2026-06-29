# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Elevenlabs speech provider integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.speech_provider import SpeechProviderBackend
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
    Single public Elevenlabs speech provider entrypoint.

    Legacy catalog factory (create_elevenlabs_speech_provider) delegates to this class.
    """

    config: ElevenlabsSpeechProviderIntegrationConfig = ElevenlabsSpeechProviderIntegrationConfig()
    _client: ElevenlabsSpeechProviderClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> ElevenlabsSpeechProviderIntegration:
        integration = cls.for_provider(
            provider_id=ELEVENLABS_SPEECH_PROVIDER_PROVIDER_ID,
            display_name="Elevenlabs",
            config=ElevenlabsSpeechProviderIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Elevenlabs integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

SpeechProviderBackend.register(ElevenlabsSpeechProviderIntegration)
