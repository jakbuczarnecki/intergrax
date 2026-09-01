# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory ``DocumentStore`` for conformance tests and lab adapters (OBS-BUS-5)."""

from __future__ import annotations

import secrets
import threading

from collections.abc import Sequence

from intergrax.integrations._shared.document_store_query_support import query_documents_with_data_filters
from intergrax.integrations.contracts.document_store import (
    DocumentDataEquality,
    DocumentDataSort,
    DocumentQueryCursorCodec,
    DocumentQueryPageV1,
    DocumentRecord,
    normalize_document_data_equalities,
    normalize_document_data_sort,
    validate_document_query_limit,
)
from intergrax.integrations.contracts.partition_atomic_document_store import (
    PartitionAtomicBatch,
    PartitionAtomicBatchResult,
    PartitionPutIfAbsentOnCreated,
    PartitionReplaceIfMatchOnCreated,
    validate_partition_atomic_batch,
)


def _require_matching_keys(*, expected: DocumentRecord, replacement: DocumentRecord) -> None:
    if (
        expected.partition_key != replacement.partition_key
        or expected.row_key != replacement.row_key
    ):
        raise ValueError(
            "replace_if_match requires expected and replacement to share "
            "partition_key and row_key"
        )


def _data_matches(current: DocumentRecord, expected: DocumentRecord) -> bool:
    return (
        current.partition_key == expected.partition_key
        and current.row_key == expected.row_key
        and dict(current.data) == dict(expected.data)
    )


class InMemoryDocumentStore:
    """Deterministic document store backing conformance suites."""

    def __init__(
        self,
        *,
        cursor_codec: DocumentQueryCursorCodec | None = None,
        cursor_secret: bytes | None = None,
    ) -> None:
        if cursor_codec is not None and cursor_secret is not None:
            raise TypeError("document_store_cursor_codec_configuration_invalid")
        self._cursor_codec = cursor_codec or DocumentQueryCursorCodec(
            secret=(
                cursor_secret
                if cursor_secret is not None
                else secrets.token_bytes(32)
            )
        )
        self._rows: dict[tuple[str, str], DocumentRecord] = {}
        self._lock = threading.RLock()
        self._last_query_rows_examined = 0

    @property
    def query_cursor_codec(self) -> DocumentQueryCursorCodec:
        return self._cursor_codec

    @property
    def last_query_rows_examined(self) -> int:
        return self._last_query_rows_examined

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        with self._lock:
            return self._rows.get((partition_key, row_key))

    def put(self, document: DocumentRecord) -> None:
        with self._lock:
            self._rows[(document.partition_key, document.row_key)] = document

    def delete(self, partition_key: str, row_key: str) -> None:
        with self._lock:
            self._rows.pop((partition_key, row_key), None)

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: str | None = None,
        cursor: str | None = None,
        row_key_upper_bound: str | None = None,
        data_equalities: Sequence[DocumentDataEquality] = (),
        sort: Sequence[DocumentDataSort] = (),
    ) -> DocumentQueryPageV1:
        validate_document_query_limit(limit)
        normalize_document_data_equalities(data_equalities)
        normalize_document_data_sort(sort)
        with self._lock:
            rows_examined_counter = [0]
            page, next_cursor = query_documents_with_data_filters(
                rows=tuple(self._rows.values()),
                partition_key=partition_key,
                limit=limit,
                row_key_prefix=row_key_prefix,
                row_key_upper_bound=row_key_upper_bound,
                data_equalities=data_equalities,
                sort=sort,
                cursor_codec=self._cursor_codec,
                cursor=cursor,
                rows_examined_counter=rows_examined_counter,
            )
            self._last_query_rows_examined = rows_examined_counter[0]
            return DocumentQueryPageV1(documents=page, next_cursor=next_cursor)

    def close(self) -> None:
        with self._lock:
            self._rows.clear()

    def put_if_absent(self, document: DocumentRecord) -> bool:
        with self._lock:
            key = (document.partition_key, document.row_key)
            if key in self._rows:
                return False
            self._rows[key] = document
            return True

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        _require_matching_keys(expected=expected, replacement=replacement)
        with self._lock:
            current = self._rows.get((expected.partition_key, expected.row_key))
            if current is None or not _data_matches(current, expected):
                return False
            self._rows[(replacement.partition_key, replacement.row_key)] = replacement
            return True

    def delete_if_match(self, *, expected: DocumentRecord) -> bool:
        with self._lock:
            key = (expected.partition_key, expected.row_key)
            current = self._rows.get(key)
            if current is None or not _data_matches(current, expected):
                return False
            del self._rows[key]
            return True

    def execute_partition_atomic_batch(
        self,
        batch: PartitionAtomicBatch,
    ) -> PartitionAtomicBatchResult:
        validated = validate_partition_atomic_batch(batch)
        with self._lock:
            snapshot = dict(self._rows)
            primary_created = self._put_if_absent_unlocked(
                validated.primary_put_if_absent,
                rows=snapshot,
            )
            if primary_created:
                for op in validated.on_created_ops:
                    if isinstance(op, PartitionPutIfAbsentOnCreated):
                        if not self._put_if_absent_unlocked(op.document, rows=snapshot):
                            raise RuntimeError("partition_atomic_batch_on_created_conflict")
                    elif isinstance(op, PartitionReplaceIfMatchOnCreated):
                        if not self._replace_if_match_unlocked(
                            expected=op.expected,
                            replacement=op.replacement,
                            rows=snapshot,
                        ):
                            raise RuntimeError("partition_atomic_batch_on_created_stale")
                    else:
                        raise TypeError("partition_atomic_batch_on_created_op_invalid")
            self._rows = snapshot
            return PartitionAtomicBatchResult(primary_created=primary_created)

    @staticmethod
    def _put_if_absent_unlocked(
        document: DocumentRecord,
        *,
        rows: dict[tuple[str, str], DocumentRecord],
    ) -> bool:
        key = (document.partition_key, document.row_key)
        if key in rows:
            return False
        rows[key] = document
        return True

    @staticmethod
    def _replace_if_match_unlocked(
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
        rows: dict[tuple[str, str], DocumentRecord],
    ) -> bool:
        _require_matching_keys(expected=expected, replacement=replacement)
        current = rows.get((expected.partition_key, expected.row_key))
        if current is None or not _data_matches(current, expected):
            return False
        rows[(replacement.partition_key, replacement.row_key)] = replacement
        return True
