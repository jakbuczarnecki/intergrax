# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Milvus vector store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MILVUS_VECTOR_STORE_PROVIDER_ID = "milvus"


class MilvusVectorStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Milvus vector store integration."""

    pass


@runtime_checkable
class MilvusVectorStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class MilvusVectorStoreIntegration(VectorStoreIntegrationContract):
    """
    Milvus vector store integration.

    The legacy facade (create_milvus_vector_store) remains separate and backward-compatible.
    """

    config: MilvusVectorStoreIntegrationConfig = MilvusVectorStoreIntegrationConfig()
    _client: MilvusVectorStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: MilvusVectorStoreClient,
        *,
        enabled: bool = False,
    ) -> MilvusVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=MILVUS_VECTOR_STORE_PROVIDER_ID,
            display_name="Milvus",
            config=MilvusVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> MilvusVectorStoreClient | None:
        return self._client
