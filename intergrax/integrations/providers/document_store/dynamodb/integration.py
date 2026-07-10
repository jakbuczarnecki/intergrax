# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""DynamoDB document store integration (PROOF-RECEIPTS-1C)."""

from __future__ import annotations

from pydantic import Field, PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.integrations.document_store import (
    DocumentStoreVendorIntegrationConfig,
    DocumentStoreVendorIntegrationContract,
)

DYNAMODB_DOCUMENT_STORE_PROVIDER_ID = "dynamodb"


class DynamoDBDocumentStoreIntegrationConfig(DocumentStoreVendorIntegrationConfig):
    """Typed config for DynamoDB document store integration."""


DynamoDBDocumentStoreClient = DocumentStore


class DynamoDBDocumentStoreIntegration(DocumentStoreVendorIntegrationContract):
    """
    Single public DynamoDB document store entrypoint.

    Legacy catalog factory (create_dynamodb_document_store) owns catalog behavior; legacy factories use from_client().
    """

    config: DynamoDBDocumentStoreIntegrationConfig = Field(
        default_factory=DynamoDBDocumentStoreIntegrationConfig
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
    ) -> DynamoDBDocumentStoreIntegration:
        integration = cls.for_provider(
            provider_id=DYNAMODB_DOCUMENT_STORE_PROVIDER_ID,
            display_name="DynamoDB",
            config=DynamoDBDocumentStoreIntegrationConfig(enabled=enabled),
        )
        integration._store = client
        return integration

    @property
    def client(self) -> DocumentStore | None:
        return self._store


# Compatibility-only aliases (historical import paths).
DynamodbDocumentStoreIntegration = DynamoDBDocumentStoreIntegration
DynamodbDocumentStoreIntegrationConfig = DynamoDBDocumentStoreIntegrationConfig
DynamodbDocumentStoreClient = DynamoDBDocumentStoreClient
