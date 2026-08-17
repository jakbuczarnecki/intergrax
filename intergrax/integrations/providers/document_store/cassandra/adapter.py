# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cassandra document store adapter — ``DocumentStore`` facade (no driver I/O here)."""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentQueryPageV1, DocumentRecord
from intergrax.integrations.providers.document_store.cassandra.client import CassandraCqlClient


class _CassandraDocumentStore:
    """
    Catalog facade over ``CassandraCqlClient``.

    Sessions are opened only in ``opens.open_cassandra_document_store()``.
    Tier-3 code MUST use ``create_cassandra_document_store()`` or ``profile.resolve()``.
    """

    def __init__(self, client: CassandraCqlClient) -> None:
        self._client = client
        self._closed = False

    @property
    def cql_client(self) -> CassandraCqlClient:
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

    def put_if_absent(self, document: DocumentRecord) -> bool:
        self._require_open()
        return self._client.put_if_absent(document)

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        self._require_open()
        return self._client.replace_if_match(
            expected=expected,
            replacement=replacement,
        )

    def delete_if_match(self, *, expected: DocumentRecord) -> bool:
        self._require_open()
        return self._client.delete_if_match(expected=expected)

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> DocumentQueryPageV1:
        self._require_open()
        if cursor is not None:
            raise ValueError("Cassandra document store does not support cursor queries")
        result = self._client.query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
        )
        return DocumentQueryPageV1(documents=tuple(result.documents))

    def close(self) -> None:
        if not self._closed:
            self._client.shutdown()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError(
                "Cassandra document store is closed; create a new store via create_cassandra_document_store()"
            )
