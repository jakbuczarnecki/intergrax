# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pinecone vector store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PINECONE_VECTOR_STORE_PROVIDER_ID = "pinecone"


class PineconeVectorStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Pinecone vector store integration."""

    pass


@runtime_checkable
class PineconeVectorStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class PineconeVectorStoreIntegration(VectorStoreIntegrationContract):
    """
    Pinecone vector store integration.

    The legacy facade (create_pinecone_integration) remains separate and backward-compatible.
    """

    config: PineconeVectorStoreIntegrationConfig = PineconeVectorStoreIntegrationConfig()
    _client: PineconeVectorStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: PineconeVectorStoreClient,
        *,
        enabled: bool = False,
    ) -> PineconeVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=PINECONE_VECTOR_STORE_PROVIDER_ID,
            display_name="Pinecone",
            config=PineconeVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> PineconeVectorStoreClient | None:
        return self._client
