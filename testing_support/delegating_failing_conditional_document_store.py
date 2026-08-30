# © Artur Czarnecki. All rights reserved.

"""Deterministic ConditionalDocumentStore write-failure injection for HARDEN proofs."""

from __future__ import annotations

from enum import StrEnum

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import (
    DocumentQueryPageV1,
    DocumentRecord,
)


class ControlledDocumentStoreWriteFailure(RuntimeError):
    """Controlled storage write failure at the ConditionalDocumentStore boundary."""


class DocumentStoreWriteFailureMode(StrEnum):
    HEALTHY = "healthy"
    FAIL_WRITES = "fail_writes"


class DelegatingFailingConditionalDocumentStore:
    """
    Wraps a contract-compliant in-memory store with an explicit write-failure mode.

    Read paths always delegate. ``FAIL_WRITES`` fails ``put_if_absent`` and
    ``replace_if_match`` deterministically — no vendor coupling, no side semantics.
    """

    def __init__(self, delegate: InMemoryDocumentStore | None = None) -> None:
        self._delegate = delegate or InMemoryDocumentStore()
        self._mode = DocumentStoreWriteFailureMode.HEALTHY

    @property
    def write_failure_mode(self) -> DocumentStoreWriteFailureMode:
        return self._mode

    def set_write_failure_mode(self, mode: DocumentStoreWriteFailureMode) -> None:
        self._mode = mode

    def _maybe_fail_write(self) -> None:
        if self._mode is DocumentStoreWriteFailureMode.FAIL_WRITES:
            raise ControlledDocumentStoreWriteFailure(
                "controlled ConditionalDocumentStore write failure",
            )

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        return self._delegate.get(partition_key, row_key)

    def put(self, document: DocumentRecord) -> None:
        self._maybe_fail_write()
        self._delegate.put(document)

    def delete(self, partition_key: str, row_key: str) -> None:
        self._delegate.delete(partition_key, row_key)

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: str | None = None,
        cursor: str | None = None,
    ) -> DocumentQueryPageV1:
        return self._delegate.query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
            cursor=cursor,
        )

    def close(self) -> None:
        self._delegate.close()

    def put_if_absent(self, document: DocumentRecord) -> bool:
        self._maybe_fail_write()
        return self._delegate.put_if_absent(document)

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        self._maybe_fail_write()
        return self._delegate.replace_if_match(
            expected=expected,
            replacement=replacement,
        )

    def delete_if_match(self, *, expected: DocumentRecord) -> bool:
        return self._delegate.delete_if_match(expected=expected)


__all__ = [
    "ControlledDocumentStoreWriteFailure",
    "DelegatingFailingConditionalDocumentStore",
    "DocumentStoreWriteFailureMode",
]
