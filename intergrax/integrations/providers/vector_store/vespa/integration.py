# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vespa vector store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

VESPA_VECTOR_STORE_PROVIDER_ID = "vespa"


class VespaVectorStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Vespa vector store integration."""

    pass


@runtime_checkable
class VespaVectorStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class VespaVectorStoreIntegration(VectorStoreIntegrationContract):
    """
    Vespa vector store integration.

    The legacy facade (create_vespa_integration) remains separate and backward-compatible.
    """

    config: VespaVectorStoreIntegrationConfig = VespaVectorStoreIntegrationConfig()
    _client: VespaVectorStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: VespaVectorStoreClient,
        *,
        enabled: bool = False,
    ) -> VespaVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=VESPA_VECTOR_STORE_PROVIDER_ID,
            display_name="Vespa",
            config=VespaVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> VespaVectorStoreClient | None:
        return self._client
