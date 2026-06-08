# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory ``DocumentStore`` for conformance tests and lab adapters (OBS-BUS-5)."""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.contracts.document_store import DocumentQueryResult, DocumentRecord


class InMemoryDocumentStore:
    """Deterministic document store backing conformance suites."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], DocumentRecord] = {}

    def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
        return self._rows.get((partition_key, row_key))

    def put(self, document: DocumentRecord) -> None:
        self._rows[(document.partition_key, document.row_key)] = document

    def delete(self, partition_key: str, row_key: str) -> None:
        self._rows.pop((partition_key, row_key), None)

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: Optional[str] = None,
    ) -> DocumentQueryResult:
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
        self._rows.clear()
