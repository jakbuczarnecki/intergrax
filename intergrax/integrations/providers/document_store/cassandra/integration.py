# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cassandra document store integration (PROOF-RECEIPTS-1C)."""

from __future__ import annotations

from pydantic import Field, PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.integrations.document_store import (
    DocumentStoreVendorIntegrationConfig,
    DocumentStoreVendorIntegrationContract,
)

CASSANDRA_DOCUMENT_STORE_PROVIDER_ID = "cassandra"


class CassandraDocumentStoreIntegrationConfig(DocumentStoreVendorIntegrationConfig):
    """Typed config for Cassandra document store integration."""


CassandraDocumentStoreClient = DocumentStore


class CassandraDocumentStoreIntegration(DocumentStoreVendorIntegrationContract):
    """
    Single public Cassandra document store entrypoint.

    Legacy catalog factory (create_cassandra_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: CassandraDocumentStoreIntegrationConfig = Field(
        default_factory=CassandraDocumentStoreIntegrationConfig
    )
    _store: DocumentStore | None = PrivateAttr(default=None)

    def as_document_store(self) -> DocumentStore:
        return self._require_store()

    def _require_store(self) -> DocumentStore:
        if self._store is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._store

    @classmethod
    def from_client(
        cls,
        client: DocumentStore,
        *,
        enabled: bool = False,
    ) -> CassandraDocumentStoreIntegration:
        integration = cls.for_provider(
            provider_id=CASSANDRA_DOCUMENT_STORE_PROVIDER_ID,
            display_name="Cassandra",
            config=CassandraDocumentStoreIntegrationConfig(enabled=enabled),
        )
        integration._store = client
        return integration

    @property
    def client(self) -> DocumentStore | None:
        return self._store
