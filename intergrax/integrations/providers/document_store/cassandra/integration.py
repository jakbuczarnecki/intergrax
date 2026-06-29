# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cassandra document store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.integrations.categories.data import DocumentStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

CASSANDRA_DOCUMENT_STORE_PROVIDER_ID = "cassandra"


class CassandraDocumentStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Cassandra document store integration."""

    pass


CassandraDocumentStoreClient = DocumentStore

class CassandraDocumentStoreIntegration(DocumentStoreIntegrationContract):
    """
    Single public Cassandra document store entrypoint.

    Legacy catalog factory (create_cassandra_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: CassandraDocumentStoreIntegrationConfig = CassandraDocumentStoreIntegrationConfig()
    _client: CassandraDocumentStoreClient | None = PrivateAttr(default=None)
    

    def close(self):
        return self._require_client().close()

    def delete(self, partition_key, row_key):
        return self._require_client().delete(partition_key, row_key)

    def get(self, partition_key, row_key):
        return self._require_client().get(partition_key, row_key)

    def put(self, document):
        return self._require_client().put(document)

    def query(self, partition_key, limit: int = 100, row_key_prefix: Optional[str] = None):
        return self._require_client().query(partition_key, limit=limit, row_key_prefix=row_key_prefix)

    def _require_client(self) -> DocumentStore:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


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

DocumentStore.register(CassandraDocumentStoreIntegration)
