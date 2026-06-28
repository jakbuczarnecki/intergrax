# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ollama interaction surface integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import InteractionSurfaceIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

OLLAMA_INTERACTION_SURFACE_PROVIDER_ID = "ollama"


class OllamaInteractionSurfaceIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Ollama interaction surface integration."""

    pass


@runtime_checkable
class OllamaInteractionSurfaceClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class OllamaInteractionSurfaceIntegration(InteractionSurfaceIntegrationContract):
    """
    Ollama interaction surface integration.

    The legacy facade (create_ollama_interaction_surface) remains separate and backward-compatible.
    """

    config: OllamaInteractionSurfaceIntegrationConfig = OllamaInteractionSurfaceIntegrationConfig()
    _client: OllamaInteractionSurfaceClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: OllamaInteractionSurfaceClient,
        *,
        enabled: bool = False,
    ) -> OllamaInteractionSurfaceIntegration:
        integration = cls.for_provider(
            provider_id=OLLAMA_INTERACTION_SURFACE_PROVIDER_ID,
            display_name="Ollama",
            config=OllamaInteractionSurfaceIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> OllamaInteractionSurfaceClient | None:
        return self._client
