# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Optional

import pytest

from intergrax.integrations.contracts.document_store import DocumentQueryResult, DocumentRecord
from intergrax.tools.providers.records.contracts import (
    RecordsDeleteInput,
    RecordsGetInput,
    RecordsPutInput,
    RecordsQueryInput,
)
from intergrax.tools.providers.records.service import records_delete, records_get, records_put, records_query
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class InMemoryDocumentStore:
    def __init__(self) -> None:
        self._docs: dict[tuple[str, str], DocumentRecord] = {}

    def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
        return self._docs.get((partition_key, row_key))

    def put(self, document: DocumentRecord) -> None:
        self._docs[(document.partition_key, document.row_key)] = document

    def delete(self, partition_key: str, row_key: str) -> None:
        self._docs.pop((partition_key, row_key), None)

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: Optional[str] = None,
    ) -> DocumentQueryResult:
        docs = [
            item
            for (part, row), item in self._docs.items()
            if part == partition_key and (row_key_prefix is None or row.startswith(row_key_prefix))
        ]
        docs = docs[:limit]
        return DocumentQueryResult(documents=docs, total=len(docs))

    def close(self) -> None:
        return None


def test_records_put_get_query_delete() -> None:
    ctx = ToolWiringContext(document_store=InMemoryDocumentStore())
    records_put(
        ctx,
        RecordsPutInput(partition_key="tenant-1", row_key="artifact-1", data={"status": "ready"}),
    )
    got = records_get(ctx, RecordsGetInput(partition_key="tenant-1", row_key="artifact-1"))
    assert got.found is True
    assert got.document is not None
    assert got.document.data["status"] == "ready"

    listed = records_query(ctx, RecordsQueryInput(partition_key="tenant-1"))
    assert listed.total == 1

    records_delete(ctx, RecordsDeleteInput(partition_key="tenant-1", row_key="artifact-1"))
    missing = records_get(ctx, RecordsGetInput(partition_key="tenant-1", row_key="artifact-1"))
    assert missing.found is False


def test_records_not_configured() -> None:
    with pytest.raises(RuntimeError, match="document_store_not_configured"):
        records_get(ToolWiringContext(), RecordsGetInput(partition_key="a", row_key="b"))
