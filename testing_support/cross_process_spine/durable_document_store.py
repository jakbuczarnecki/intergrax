# © Artur Czarnecki. All rights reserved.

"""SQLite-backed document store for cross-process OBS/DIAG qualification (testing only)."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

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
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore


def _record_to_row(document: DocumentRecord) -> str:
    return json.dumps(document.model_dump(mode="json"), sort_keys=True)


def _row_to_record(raw: str) -> DocumentRecord:
    return DocumentRecord.model_validate(json.loads(raw))


class SqliteFileDocumentStore:
    """Process-safe document store for cross-process diagnostic persistence proofs."""

    def __init__(
        self,
        db_path: Path,
        *,
        cursor_secret: bytes,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._cursor_codec = DocumentQueryCursorCodec(secret=cursor_secret)
        self._lock = threading.RLock()
        self._last_query_rows_examined = 0
        self._init_schema()

    @property
    def query_cursor_codec(self) -> DocumentQueryCursorCodec:
        return self._cursor_codec

    @property
    def last_query_rows_examined(self) -> int:
        return self._last_query_rows_examined

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS diagnostic_documents (
                        partition_key TEXT NOT NULL,
                        row_key TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        PRIMARY KEY (partition_key, row_key)
                    )
                    """
                )

    def _load_rows(self) -> dict[tuple[str, str], DocumentRecord]:
        with self._connect() as conn:
            rows: dict[tuple[str, str], DocumentRecord] = {}
            for partition_key, row_key, payload in conn.execute(
                "SELECT partition_key, row_key, payload FROM diagnostic_documents",
            ):
                document = _row_to_record(str(payload))
                rows[(str(partition_key), str(row_key))] = document
            return rows

    def _persist_rows(self, rows: dict[tuple[str, str], DocumentRecord]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM diagnostic_documents")
            conn.executemany(
                """
                INSERT INTO diagnostic_documents (partition_key, row_key, payload)
                VALUES (?, ?, ?)
                """,
                [
                    (pk, rk, _record_to_row(document))
                    for (pk, rk), document in rows.items()
                ],
            )

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT payload FROM diagnostic_documents
                    WHERE partition_key = ? AND row_key = ?
                    """,
                    (partition_key, row_key),
                ).fetchone()
                if row is None:
                    return None
                return _row_to_record(str(row[0]))

    def put(self, document: DocumentRecord) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO diagnostic_documents (partition_key, row_key, payload)
                    VALUES (?, ?, ?)
                    ON CONFLICT(partition_key, row_key) DO UPDATE SET payload = excluded.payload
                    """,
                    (document.partition_key, document.row_key, _record_to_row(document)),
                )

    def delete(self, partition_key: str, row_key: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM diagnostic_documents WHERE partition_key = ? AND row_key = ?",
                    (partition_key, row_key),
                )

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
            rows = self._load_rows()
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

    def close(self) -> None:
        """Release adapter lifecycle; durable file-backed state is retained."""

    def truncate_storage(self) -> None:
        """Test-harness cleanup — removes all rows; not part of durable close semantics."""
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM diagnostic_documents")

    def put_if_absent(self, document: DocumentRecord) -> bool:
        with self._lock:
            rows = self._load_rows()
            key = (document.partition_key, document.row_key)
            if key in rows:
                return False
            rows[key] = document
            self._persist_rows(rows)
            return True

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        with self._lock:
            rows = self._load_rows()
            ok = InMemoryDocumentStore._replace_if_match_unlocked(
                expected=expected,
                replacement=replacement,
                rows=rows,
            )
            if ok:
                self._persist_rows(rows)
            return ok

    def delete_if_match(self, *, expected: DocumentRecord) -> bool:
        with self._lock:
            rows = self._load_rows()
            key = (expected.partition_key, expected.row_key)
            current = rows.get(key)
            if current is None:
                return False
            if dict(current.data) != dict(expected.data):
                return False
            del rows[key]
            self._persist_rows(rows)
            return True

    def execute_partition_atomic_batch(
        self,
        batch: PartitionAtomicBatch,
    ) -> PartitionAtomicBatchResult:
        validated = validate_partition_atomic_batch(batch)
        with self._lock:
            rows = self._load_rows()
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
                            raise RuntimeError("partition_atomic_batch_on_created_conflict")
                    elif isinstance(op, PartitionReplaceIfMatchOnCreated):
                        if not InMemoryDocumentStore._replace_if_match_unlocked(
                            expected=op.expected,
                            replacement=op.replacement,
                            rows=snapshot,
                        ):
                            raise RuntimeError("partition_atomic_batch_on_created_stale")
                    else:
                        raise TypeError("partition_atomic_batch_on_created_op_invalid")
            self._persist_rows(snapshot)
            return PartitionAtomicBatchResult(primary_created=primary_created)
