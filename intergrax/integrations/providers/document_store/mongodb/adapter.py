# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MongoDB document store adapter — ``DocumentStore`` facade (no driver I/O here)."""

from __future__ import annotations

import hashlib
from typing import Optional, Sequence

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import (
    DocumentDataEquality,
    DocumentDataSort,
    DocumentQueryCursorCodec,
    DocumentQueryPageV1,
    DocumentRecord,
    document_sort_key_values,
    normalize_document_data_equalities,
    normalize_document_data_sort,
    validate_document_query_limit,
)
from intergrax.integrations.contracts.partition_atomic_document_store import (
    PartitionAtomicBatch,
    PartitionAtomicBatchResult,
)
from intergrax.integrations.providers.document_store.mongodb.client import MongoCollectionClient
from intergrax.integrations.providers.document_store.mongodb.config import MongoDBIntegrationConfig


def _cursor_codec_for_config(config: MongoDBIntegrationConfig) -> DocumentQueryCursorCodec:
    database, collection = config.qualified_collection()
    material = f"{config.uri}:{database}:{collection}".encode("utf-8")
    return DocumentQueryCursorCodec(secret=hashlib.sha256(material).digest())


class _MongoDBDocumentStore:
    """
    Catalog facade over ``MongoCollectionClient``.

    Connections are opened only in ``opens.open_mongodb_document_store()``.
    Tier-3 code MUST use ``create_mongodb_document_store()`` or ``profile.resolve()``.
    """

    def __init__(self, client: MongoCollectionClient) -> None:
        self._client = client
        self._cursor_codec = _cursor_codec_for_config(client.config)
        self._closed = False

    @property
    def mongo_client(self) -> MongoCollectionClient:
        return self._client

    @property
    def query_cursor_codec(self) -> DocumentQueryCursorCodec:
        return self._cursor_codec

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
        cursor: str | None = None,
        row_key_upper_bound: str | None = None,
        data_equalities: Sequence[DocumentDataEquality] = (),
        sort: Sequence[DocumentDataSort] = (),
    ) -> DocumentQueryPageV1:
        self._require_open()
        bounded_limit = validate_document_query_limit(limit)
        max_page_limit = validate_document_query_limit(5000)
        normalized_equalities = normalize_document_data_equalities(data_equalities)
        normalized_sort = normalize_document_data_sort(sort)

        if normalized_sort:
            fetch_limit = bounded_limit + 1 if bounded_limit < max_page_limit else bounded_limit
            documents = self._client.query(
                partition_key,
                limit=fetch_limit,
                row_key_prefix=row_key_prefix,
                row_key_upper_bound=row_key_upper_bound,
                data_equalities=normalized_equalities,
                sort=normalized_sort,
                cursor_codec=self._cursor_codec,
                cursor=cursor,
            )
            if bounded_limit < max_page_limit:
                has_more = len(documents) > bounded_limit
                page = documents[:bounded_limit]
            else:
                page = documents[:bounded_limit]
                has_more = (
                    len(page) == bounded_limit
                    and page
                    and len(
                        self._client.query(
                            partition_key,
                            limit=1,
                            row_key_prefix=row_key_prefix,
                            row_key_upper_bound=row_key_upper_bound,
                            data_equalities=normalized_equalities,
                            sort=normalized_sort,
                            cursor_codec=self._cursor_codec,
                            cursor=self._cursor_codec.encode_v2(
                                partition_key=partition_key,
                                row_key_prefix=row_key_prefix,
                                row_key_upper_bound=row_key_upper_bound,
                                data_equalities=normalized_equalities,
                                sort=normalized_sort,
                                last_row_key=page[-1].row_key,
                                last_sort_values=document_sort_key_values(
                                    page[-1],
                                    normalized_sort,
                                ),
                            ),
                        )
                    ) > 0
                )
            next_cursor = (
                self._cursor_codec.encode_v2(
                    partition_key=partition_key,
                    row_key_prefix=row_key_prefix,
                    row_key_upper_bound=row_key_upper_bound,
                    data_equalities=normalized_equalities,
                    sort=normalized_sort,
                    last_row_key=page[-1].row_key,
                    last_sort_values=document_sort_key_values(page[-1], normalized_sort),
                )
                if has_more and page
                else None
            )
            return DocumentQueryPageV1(documents=tuple(page), next_cursor=next_cursor)

        after_row_key: str | None = None
        if cursor is not None:
            after_row_key = self._cursor_codec.decode(
                cursor,
                partition_key=partition_key,
                row_key_prefix=row_key_prefix,
            ).last_row_key
        if bounded_limit < max_page_limit:
            fetch_limit = bounded_limit + 1
            documents = self._client.query(
                partition_key,
                limit=fetch_limit,
                row_key_prefix=row_key_prefix,
                after_row_key=after_row_key,
                row_key_upper_bound=row_key_upper_bound,
                data_equalities=normalized_equalities,
            )
            has_more = len(documents) > bounded_limit
            page = documents[:bounded_limit]
        else:
            documents = self._client.query(
                partition_key,
                limit=bounded_limit,
                row_key_prefix=row_key_prefix,
                after_row_key=after_row_key,
                row_key_upper_bound=row_key_upper_bound,
                data_equalities=normalized_equalities,
            )
            page = documents[:bounded_limit]
            has_more = (
                len(page) == bounded_limit
                and page
                and len(
                    self._client.query(
                        partition_key,
                        limit=1,
                        row_key_prefix=row_key_prefix,
                        after_row_key=page[-1].row_key,
                        row_key_upper_bound=row_key_upper_bound,
                        data_equalities=normalized_equalities,
                    )
                ) > 0
            )
        next_cursor = (
            self._cursor_codec.encode(
                partition_key=partition_key,
                row_key_prefix=row_key_prefix,
                last_row_key=page[-1].row_key,
            )
            if has_more and page
            else None
        )
        return DocumentQueryPageV1(documents=tuple(page), next_cursor=next_cursor)

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

    def supports_partition_atomic_batch(self) -> bool:
        self._require_open()
        return self._client.supports_partition_atomic_batch()

    def execute_partition_atomic_batch(
        self,
        batch: PartitionAtomicBatch,
    ) -> PartitionAtomicBatchResult:
        self._require_open()
        return self._client.execute_partition_atomic_batch(batch)

    def close(self) -> None:
        if not self._closed:
            self._client.close()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError(
                "MongoDB document store is closed; create a new store via create_mongodb_document_store()"
            )
