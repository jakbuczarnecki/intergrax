# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""pgvector vector store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PGVECTOR_VECTOR_STORE_PROVIDER_ID = "pgvector"


class PgvectorVectorStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for pgvector vector store integration."""

    pass


@runtime_checkable
class PgvectorVectorStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class PgvectorVectorStoreIntegration(VectorStoreIntegrationContract):
    """
    pgvector vector store integration.

    The legacy facade (create_pgvector_vector_store) remains separate and backward-compatible.
    """

    config: PgvectorVectorStoreIntegrationConfig = PgvectorVectorStoreIntegrationConfig()
    _client: PgvectorVectorStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: PgvectorVectorStoreClient,
        *,
        enabled: bool = False,
    ) -> PgvectorVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=PGVECTOR_VECTOR_STORE_PROVIDER_ID,
            display_name="pgvector",
            config=PgvectorVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> PgvectorVectorStoreClient | None:
        return self._client
