# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory ``DocumentStore`` for conformance tests and lab adapters (OBS-BUS-5)."""

from __future__ import annotations

import threading
from typing import Optional

from intergrax.integrations.contracts.document_store import DocumentQueryResult, DocumentRecord


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

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], DocumentRecord] = {}
        self._lock = threading.RLock()

    def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
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
        row_key_prefix: Optional[str] = None,
    ) -> DocumentQueryResult:
        with self._lock:
            rows: list[DocumentRecord] = []
            for (pk, rk), doc in self._rows.items():
                if pk != partition_key:
                    continue
                if row_key_prefix is not None and not rk.startswith(row_key_prefix):
                    continue
                rows.append(doc)
            rows.sort(key=lambda doc: doc.row_key)
            sliced = rows[:limit]
            return DocumentQueryResult(documents=sliced, total=len(sliced))

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
