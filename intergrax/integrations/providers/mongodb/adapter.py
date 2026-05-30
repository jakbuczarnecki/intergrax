# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MongoDB document store adapter — ``DocumentStore`` facade (no driver I/O here)."""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentQueryResult, DocumentRecord
from intergrax.integrations.providers.mongodb.client import MongoCollectionClient


class MongoDBDocumentStore:
    """
    Catalog facade over ``MongoCollectionClient``.

    Connections are opened only in ``opens.open_mongodb_document_store()``.
    Tier-3 code MUST use ``create_mongodb_document_store()`` or ``profile.resolve()``.
    """

    def __init__(self, client: MongoCollectionClient) -> None:
        self._client = client
        self._closed = False

    @property
    def mongo_client(self) -> MongoCollectionClient:
        return self._client

    def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
        self._require_open()
        return self._client.get(partition_key, row_key)

    def put(self, document: DocumentRecord) -> None:
        self._require_open()
        self._client.put(document)

    def delete(self, partition_key: str, row_key: str) -> None:
        self._require_open()
        self._client.delete(partition_key, row_key)

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: Optional[str] = None,
    ) -> DocumentQueryResult:
        self._require_open()
        return self._client.query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
        )

    def close(self) -> None:
        if not self._closed:
            self._client.close()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError(
                "MongoDB document store is closed; create a new store via create_mongodb_document_store()"
            )
