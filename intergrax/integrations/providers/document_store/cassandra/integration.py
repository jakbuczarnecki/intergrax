# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cassandra document store integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.data import DocumentStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

CASSANDRA_DOCUMENT_STORE_PROVIDER_ID = "cassandra"


class CassandraDocumentStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Cassandra document store integration."""

    pass


@runtime_checkable
class CassandraDocumentStoreClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class CassandraDocumentStoreIntegration(DocumentStoreIntegrationContract):
    """
    Cassandra document store integration.

    The legacy facade (create_cassandra_integration) remains separate and backward-compatible.
    """

    config: CassandraDocumentStoreIntegrationConfig = CassandraDocumentStoreIntegrationConfig()
    _client: CassandraDocumentStoreClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: CassandraDocumentStoreClient,
        *,
        enabled: bool = False,
    ) -> CassandraDocumentStoreIntegration:
        integration = cls.for_provider(
            provider_id=CASSANDRA_DOCUMENT_STORE_PROVIDER_ID,
            display_name="Cassandra",
            config=CassandraDocumentStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> CassandraDocumentStoreClient | None:
        return self._client
