# © Artur Czarnecki. All rights reserved.

"""Deterministic CAS interleaving gates for ConditionalDocumentStore conformance proofs."""

from __future__ import annotations

import threading
from collections.abc import Callable

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import (
    DocumentQueryPageV1,
    DocumentRecord,
)


class BarrierConditionalDocumentStore:
    """
    Delegates to an in-memory ConditionalDocumentStore and synchronizes selected CAS ops.

    Problem-agnostic: gates are keyed by operation name only. No reconciliation semantics.
    """

    def __init__(
        self,
        delegate: InMemoryDocumentStore | None = None,
        *,
        put_if_absent_barrier: threading.Barrier | None = None,
        replace_if_match_barrier: threading.Barrier | None = None,
        put_if_absent_hook: Callable[[DocumentRecord], None] | None = None,
        replace_if_match_hook: Callable[[DocumentRecord, DocumentRecord], None] | None = None,
    ) -> None:
        self._delegate = delegate or InMemoryDocumentStore()
        self._put_if_absent_barrier = put_if_absent_barrier
        self._replace_if_match_barrier = replace_if_match_barrier
        self._put_if_absent_hook = put_if_absent_hook
        self._replace_if_match_hook = replace_if_match_hook

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        return self._delegate.get(partition_key, row_key)

    def put(self, document: DocumentRecord) -> None:
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
        if self._put_if_absent_barrier is not None:
            self._put_if_absent_barrier.wait(timeout=5)
        if self._put_if_absent_hook is not None:
            self._put_if_absent_hook(document)
        return self._delegate.put_if_absent(document)

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        if self._replace_if_match_barrier is not None:
            self._replace_if_match_barrier.wait(timeout=5)
        if self._replace_if_match_hook is not None:
            self._replace_if_match_hook(expected, replacement)
        return self._delegate.replace_if_match(
            expected=expected,
            replacement=replacement,
        )

    def delete_if_match(self, *, expected: DocumentRecord) -> bool:
        return self._delegate.delete_if_match(expected=expected)


__all__ = ["BarrierConditionalDocumentStore"]
