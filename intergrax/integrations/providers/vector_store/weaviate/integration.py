# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Weaviate vector store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

WEAVIATE_VECTOR_STORE_PROVIDER_ID = "weaviate"


class WeaviateVectorStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Weaviate vector store integration."""

    pass


@runtime_checkable
class WeaviateVectorStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class WeaviateVectorStoreIntegration(VectorStoreIntegrationContract):
    """
    Weaviate vector store integration.

    The legacy facade (create_weaviate_vector_store) remains separate and backward-compatible.
    """

    config: WeaviateVectorStoreIntegrationConfig = WeaviateVectorStoreIntegrationConfig()
    _client: WeaviateVectorStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: WeaviateVectorStoreClient,
        *,
        enabled: bool = False,
    ) -> WeaviateVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=WEAVIATE_VECTOR_STORE_PROVIDER_ID,
            display_name="Weaviate",
            config=WeaviateVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> WeaviateVectorStoreClient | None:
        return self._client
