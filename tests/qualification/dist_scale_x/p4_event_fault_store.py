# © Artur Czarnecki. All rights reserved.

"""Qualification-only DocumentStore wrapper — denies runtime-event partitions (F-EVENT)."""

from __future__ import annotations

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentQueryPageV1,
    DocumentRecord,
)


class EventAppendFaultDocumentStore:
    """Narrow injection: F07 event paths fail; budget/other partitions delegate."""

    __slots__ = ("_inner", "_deny")

    def __init__(self, inner: ConditionalDocumentStore, *, deny: bool = True) -> None:
        self._inner = inner
        self._deny = deny

    @staticmethod
    def _is_runtime_event_partition(partition_key: str) -> bool:
        if partition_key == "runtime_event|event_id":
            return True
        if "|run|" in partition_key:
            return True
        if partition_key.endswith("|event"):
            return True
        return False

    def _check(self, partition_key: str) -> None:
        if self._deny and self._is_runtime_event_partition(partition_key):
            raise RuntimeError("qualification_event_append_outage")

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        return self._inner.get(partition_key, row_key)

    def put(self, document: DocumentRecord) -> None:
        self._check(document.partition_key)
        self._inner.put(document)

    def put_if_absent(self, document: DocumentRecord) -> bool:
        self._check(document.partition_key)
        return self._inner.put_if_absent(document)

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        self._check(expected.partition_key)
        self._check(replacement.partition_key)
        return self._inner.replace_if_match(expected=expected, replacement=replacement)

    def delete_if_match(self, *, expected: DocumentRecord) -> bool:
        return self._inner.delete_if_match(expected=expected)

    def delete(self, partition_key: str, row_key: str) -> None:
        self._inner.delete(partition_key, row_key)

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: str | None = None,
        cursor: str | None = None,
    ) -> DocumentQueryPageV1:
        return self._inner.query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
            cursor=cursor,
        )
