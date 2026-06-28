# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qdrant vector store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

QDRANT_VECTOR_STORE_PROVIDER_ID = "qdrant"


class QdrantVectorStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Qdrant vector store integration."""

    pass


@runtime_checkable
class QdrantVectorStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class QdrantVectorStoreIntegration(VectorStoreIntegrationContract):
    """
    Qdrant vector store integration.

    The legacy facade (create_qdrant_integration) remains separate and backward-compatible.
    """

    config: QdrantVectorStoreIntegrationConfig = QdrantVectorStoreIntegrationConfig()
    _client: QdrantVectorStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: QdrantVectorStoreClient,
        *,
        enabled: bool = False,
    ) -> QdrantVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=QDRANT_VECTOR_STORE_PROVIDER_ID,
            display_name="Qdrant",
            config=QdrantVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> QdrantVectorStoreClient | None:
        return self._client
