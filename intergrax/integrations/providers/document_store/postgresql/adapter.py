# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PostgreSQL ``ConditionalDocumentStore`` adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import (
    DocumentDataEquality,
    DocumentDataSort,
    DocumentQueryPageV1,
    DocumentRecord,
)
from intergrax.integrations.contracts.document_store_process_durability import (
    ProcessRestartDurableDocumentStore,
)
from intergrax.integrations.contracts.partition_atomic_document_store import (
    PartitionAtomicBatch,
    PartitionAtomicBatchResult,
)
from intergrax.integrations.providers.document_store.postgresql.client import (
    PostgreSQLDocumentTableClient,
)


class _PostgreSQLDocumentStore(ProcessRestartDurableDocumentStore):
    """Catalog facade over ``PostgreSQLDocumentTableClient``."""

    def __init__(self, client: PostgreSQLDocumentTableClient) -> None:
        self._client = client
        self._closed = False

    @property
    def pg_client(self) -> PostgreSQLDocumentTableClient:
        return self._client

    @property
    def query_cursor_codec(self):
        return self._client.query_cursor_codec

    @property
    def last_query_rows_examined(self) -> int:
        return self._client.last_query_rows_examined

    @property
    def survives_process_restart(self) -> bool:
        return True

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

    def execute_partition_atomic_batch(
        self,
        batch: PartitionAtomicBatch,
    ) -> PartitionAtomicBatchResult:
        self._require_open()
        return self._client.execute_partition_atomic_batch(batch)

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: Optional[str] = None,
        cursor: Optional[str] = None,
        row_key_upper_bound: Optional[str] = None,
        data_equalities: Sequence[DocumentDataEquality] = (),
        sort: Sequence[DocumentDataSort] = (),
    ) -> DocumentQueryPageV1:
        self._require_open()
        return self._client.query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
            cursor=cursor,
            row_key_upper_bound=row_key_upper_bound,
            data_equalities=data_equalities,
            sort=sort,
        )

    def close(self) -> None:
        self._closed = True
        self._client.close()

    def truncate_storage(self) -> None:
        self._client.truncate_storage()

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError("PostgreSQL document store is closed")


__all__ = ["_PostgreSQLDocumentStore"]
