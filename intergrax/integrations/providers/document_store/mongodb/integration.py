# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Mongodb document store integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.integrations.categories.data import DocumentStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MONGODB_DOCUMENT_STORE_PROVIDER_ID = "mongodb"


class MongodbDocumentStoreIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Mongodb document store integration."""

    pass


MongodbDocumentStoreClient = DocumentStore

class MongodbDocumentStoreIntegration(DocumentStoreIntegrationContract):
    """
    Single public Mongodb document store entrypoint.

    Legacy catalog factory (create_mongodb_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: MongodbDocumentStoreIntegrationConfig = MongodbDocumentStoreIntegrationConfig()
    _client: MongodbDocumentStoreClient | None = PrivateAttr(default=None)
    

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
        client: MongodbDocumentStoreClient,
        *,
        enabled: bool = False,
    ) -> MongodbDocumentStoreIntegration:
        integration = cls.for_provider(
            provider_id=MONGODB_DOCUMENT_STORE_PROVIDER_ID,
            display_name="Mongodb",
            config=MongodbDocumentStoreIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> MongodbDocumentStoreClient | None:
        return self._client

DocumentStore.register(MongodbDocumentStoreIntegration)
