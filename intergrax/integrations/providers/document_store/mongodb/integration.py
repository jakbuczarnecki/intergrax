# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MongoDB document store integration (PROOF-RECEIPTS-1C)."""

from __future__ import annotations

from pydantic import Field, PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.integrations.document_store import (
    DocumentStoreVendorIntegrationConfig,
    DocumentStoreVendorIntegrationContract,
)

MONGODB_DOCUMENT_STORE_PROVIDER_ID = "mongodb"


class MongoDBDocumentStoreIntegrationConfig(DocumentStoreVendorIntegrationConfig):
    """Typed config for MongoDB document store integration."""


MongoDBDocumentStoreClient = DocumentStore


class MongoDBDocumentStoreIntegration(DocumentStoreVendorIntegrationContract):
    """
    Single public MongoDB document store entrypoint.

    Legacy catalog factory (create_mongodb_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: MongoDBDocumentStoreIntegrationConfig = Field(
        default_factory=MongoDBDocumentStoreIntegrationConfig
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
    ) -> MongoDBDocumentStoreIntegration:
        integration = cls.for_provider(
            provider_id=MONGODB_DOCUMENT_STORE_PROVIDER_ID,
            display_name="MongoDB",
            config=MongoDBDocumentStoreIntegrationConfig(enabled=enabled),
        )
        integration._store = client
        return integration

    @property
    def client(self) -> DocumentStore | None:
        return self._store


# Compatibility-only aliases (historical import paths).
MongodbDocumentStoreIntegration = MongoDBDocumentStoreIntegration
MongodbDocumentStoreIntegrationConfig = MongoDBDocumentStoreIntegrationConfig
MongodbDocumentStoreClient = MongoDBDocumentStoreClient
