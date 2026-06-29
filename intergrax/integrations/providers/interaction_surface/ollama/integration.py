# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ollama interaction surface integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Optional, Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.interaction_surface import InteractionSurface
from intergrax.runtime.integrations.categories.collaboration import InteractionSurfaceIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.utils import attribute_access


OLLAMA_INTERACTION_SURFACE_PROVIDER_ID = "ollama"


class OllamaInteractionSurfaceIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Ollama interaction surface integration."""

    pass


OllamaInteractionSurfaceClient = InteractionSurface

class OllamaInteractionSurfaceIntegration(InteractionSurfaceIntegrationContract):
    """
    Single public Ollama interaction surface entrypoint.

    Legacy catalog factory (create_ollama_interaction_surface) owns catalog behavior; legacy factories use from_client().
    """

    config: OllamaInteractionSurfaceIntegrationConfig = OllamaInteractionSurfaceIntegrationConfig()
    _client: OllamaInteractionSurfaceClient | None = PrivateAttr(default=None)
    

    def can_handle(self, payload):
        return self._require_client().can_handle(payload)

    @property
    def channel(self):
        return attribute_access.optional_str(self._require_client(), 'channel')

    def to_inbound(self, payload, *, tenant_id: str, user_id: str):
        return self._require_client().to_inbound(payload, tenant_id=tenant_id, user_id=user_id)

    def to_task(self, payload, *, tenant_id: str, user_id: Optional[str] = None):
        return self._require_client().to_task(payload, tenant_id=tenant_id, user_id=user_id)

    def _require_client(self) -> InteractionSurface:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


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

InteractionSurface.register(OllamaInteractionSurfaceIntegration)
