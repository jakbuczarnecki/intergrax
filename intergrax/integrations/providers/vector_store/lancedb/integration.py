# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lancedb vector store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

LANCEDB_VECTOR_STORE_PROVIDER_ID = "lancedb"


class LancedbVectorStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Lancedb vector store integration."""

    pass


@runtime_checkable
class LancedbVectorStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class LancedbVectorStoreIntegration(VectorStoreIntegrationContract):
    """
    Lancedb vector store integration.

    The legacy facade (create_lancedb_vector_store) remains separate and backward-compatible.
    """

    config: LancedbVectorStoreIntegrationConfig = LancedbVectorStoreIntegrationConfig()
    _client: LancedbVectorStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: LancedbVectorStoreClient,
        *,
        enabled: bool = False,
    ) -> LancedbVectorStoreIntegration:
        integration = cls.for_provider(
            provider_id=LANCEDB_VECTOR_STORE_PROVIDER_ID,
            display_name="Lancedb",
            config=LancedbVectorStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> LancedbVectorStoreClient | None:
        return self._client
