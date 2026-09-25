# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PostgreSQL document table client — parameterized SQL only."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from intergrax.integrations._shared.document_store_query_support import (
    query_documents_with_data_filters,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
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
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
)

_DOCUMENT_TABLE = "intergrax_document_records"


def _record_to_payload(document: DocumentRecord) -> str:
    return json.dumps(document.model_dump(mode="json"), sort_keys=True)


def _payload_to_record(raw: str) -> DocumentRecord:
    return DocumentRecord.model_validate(json.loads(raw))


class PostgreSQLDocumentTableClient:
    """One connection provider + schema scope; each store adapter owns a client instance."""

    def __init__(
        self,
        provider: PostgreSQLConnectionProvider,
        *,
        cursor_secret: bytes,
    ) -> None:
        self._provider = provider
        self._cursor_codec = DocumentQueryCursorCodec(secret=cursor_secret)
        self._lock = threading.RLock()
        self._last_query_rows_examined = 0
        self._ensure_table()

    @property
    def connection_provider(self) -> PostgreSQLConnectionProvider:
        return self._provider

    @property
    def query_cursor_codec(self) -> DocumentQueryCursorCodec:
        return self._cursor_codec

    @property
    def last_query_rows_examined(self) -> int:
        return self._last_query_rows_examined

    def _ensure_table(self) -> None:
        with self._provider.connection() as session:
            self._provider.ensure_schema_exists(session)
            session.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_DOCUMENT_TABLE} (
                    partition_key TEXT NOT NULL,
                    row_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (partition_key, row_key)
                )
                """,
            )
            session.commit()

    def truncate_storage(self) -> None:
        with self._lock:
            with self._provider.connection() as session:
                session.execute(f"DELETE FROM {_DOCUMENT_TABLE}")
                session.commit()

    def close(self) -> None:
        """Adapter lifecycle hook; pooled connections are per-operation."""

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        with self._lock:
            with self._provider.connection() as session:
                result = session.execute(
                    f"""
                    SELECT payload FROM {_DOCUMENT_TABLE}
                    WHERE partition_key = %s AND row_key = %s
                    """,
                    (partition_key, row_key),
                )
                row = result.fetchone()
                if row is None:
                    return None
                return _payload_to_record(str(row["payload"]))

    def put(self, document: DocumentRecord) -> None:
        payload = _record_to_payload(document)
        with self._lock:
            with self._provider.connection() as session:
                session.execute(
                    f"""
                    INSERT INTO {_DOCUMENT_TABLE} (partition_key, row_key, payload)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (partition_key, row_key)
                    DO UPDATE SET payload = EXCLUDED.payload
                    """,
                    (document.partition_key, document.row_key, payload),
                )
                session.commit()

    def delete(self, partition_key: str, row_key: str) -> None:
        with self._lock:
            with self._provider.connection() as session:
                session.execute(
                    f"DELETE FROM {_DOCUMENT_TABLE} WHERE partition_key = %s AND row_key = %s",
                    (partition_key, row_key),
                )
                session.commit()

    def _load_partition_rows(
        self, partition_key: str
    ) -> dict[tuple[str, str], DocumentRecord]:
        with self._provider.connection() as session:
            result = session.execute(
                f"""
                SELECT partition_key, row_key, payload FROM {_DOCUMENT_TABLE}
                WHERE partition_key = %s
                """,
                (partition_key,),
            )
            rows: dict[tuple[str, str], DocumentRecord] = {}
            for row in result.fetchall():
                document = _payload_to_record(str(row["payload"]))
                rows[(str(row["partition_key"]), str(row["row_key"]))] = document
            return rows

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
            rows = self._load_partition_rows(partition_key)
            rows_examined_counter = [0]
            page, next_cursor = query_documents_with_data_filters(
                rows=tuple(rows.values()),
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

    def put_if_absent(self, document: DocumentRecord) -> bool:
        payload = _record_to_payload(document)
        with self._lock:
            with self._provider.connection() as session:
                result = session.execute(
                    f"""
                    INSERT INTO {_DOCUMENT_TABLE} (partition_key, row_key, payload)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (partition_key, row_key) DO NOTHING
                    """,
                    (document.partition_key, document.row_key, payload),
                )
                session.commit()
                return result.rowcount == 1

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        expected_payload = _record_to_payload(expected)
        replacement_payload = _record_to_payload(replacement)
        with self._lock:
            with self._provider.connection() as session:
                result = session.execute(
                    f"""
                    UPDATE {_DOCUMENT_TABLE}
                    SET payload = %s
                    WHERE partition_key = %s AND row_key = %s AND payload = %s
                    """,
                    (
                        replacement_payload,
                        replacement.partition_key,
                        replacement.row_key,
                        expected_payload,
                    ),
                )
                session.commit()
                return result.rowcount == 1

    def delete_if_match(self, *, expected: DocumentRecord) -> bool:
        expected_payload = _record_to_payload(expected)
        with self._lock:
            with self._provider.connection() as session:
                result = session.execute(
                    f"""
                    DELETE FROM {_DOCUMENT_TABLE}
                    WHERE partition_key = %s AND row_key = %s AND payload = %s
                    """,
                    (
                        expected.partition_key,
                        expected.row_key,
                        expected_payload,
                    ),
                )
                session.commit()
                return result.rowcount == 1

    def execute_partition_atomic_batch(
        self,
        batch: PartitionAtomicBatch,
    ) -> PartitionAtomicBatchResult:
        validated = validate_partition_atomic_batch(batch)
        with self._lock:
            rows = self._load_all_rows()
            snapshot = dict(rows)
            primary_created = InMemoryDocumentStore._put_if_absent_unlocked(
                validated.primary_put_if_absent,
                rows=snapshot,
            )
            if primary_created:
                for op in validated.on_created_ops:
                    if isinstance(op, PartitionPutIfAbsentOnCreated):
                        if not InMemoryDocumentStore._put_if_absent_unlocked(
                            op.document,
                            rows=snapshot,
                        ):
                            raise RuntimeError(
                                "partition_atomic_batch_on_created_conflict"
                            )
                    elif isinstance(op, PartitionReplaceIfMatchOnCreated):
                        if not InMemoryDocumentStore._replace_if_match_unlocked(
                            expected=op.expected,
                            replacement=op.replacement,
                            rows=snapshot,
                        ):
                            raise RuntimeError(
                                "partition_atomic_batch_on_created_stale"
                            )
                    else:
                        raise TypeError("partition_atomic_batch_on_created_op_invalid")
            self._persist_rows_snapshot(snapshot)
            return PartitionAtomicBatchResult(primary_created=primary_created)

    def _load_all_rows(self) -> dict[tuple[str, str], DocumentRecord]:
        with self._provider.connection() as session:
            result = session.execute(
                f"SELECT partition_key, row_key, payload FROM {_DOCUMENT_TABLE}",
            )
            rows: dict[tuple[str, str], DocumentRecord] = {}
            for row in result.fetchall():
                document = _payload_to_record(str(row["payload"]))
                rows[(str(row["partition_key"]), str(row["row_key"]))] = document
            return rows

    def _persist_rows_snapshot(
        self, rows: dict[tuple[str, str], DocumentRecord]
    ) -> None:
        with self._provider.connection() as session:
            session.execute(f"DELETE FROM {_DOCUMENT_TABLE}")
            for (partition_key, row_key), document in rows.items():
                session.execute(
                    f"""
                    INSERT INTO {_DOCUMENT_TABLE} (partition_key, row_key, payload)
                    VALUES (%s, %s, %s)
                    """,
                    (partition_key, row_key, _record_to_payload(document)),
                )
            session.commit()


__all__ = ["PostgreSQLDocumentTableClient"]
