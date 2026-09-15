# © Artur Czarnecki. All rights reserved.

"""Delegated invocation correlation query adapters (P2.1-S2C3)."""

from __future__ import annotations

from datetime import datetime

from intergrax.contracts.delegated_execution_query import (
    DelegatedInvocationCorrelationQuery,
    DelegatedInvocationCorrelationQueryStore,
    DelegatedInvocationCorrelationQueryStorePage,
    delegated_correlation_backend_scan_limit,
)
from intergrax.contracts.delegated_invocation_correlation import (
    DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
    DelegatedInvocationCorrelationCompositionError,
    DelegatedInvocationCorrelationDurabilityMode,
    DelegatedInvocationCorrelationPersistenceError,
    DelegatedInvocationCorrelationRecord,
)
from intergrax.contracts.execution_identity import ExecutionId
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentDataSort,
    DocumentRecord,
    validate_document_query_limit,
)
from intergrax.runtime.execution.delegated_execution.correlation_persistence import (
    InMemoryDelegatedInvocationCorrelationBackend,
    InMemoryDelegatedInvocationCorrelationStore,
    _DOCUMENT_PARTITION,
    _QUERY_PERSISTED_AT,
    _document_to_correlation,
)
from intergrax.runtime.execution.delegated_execution.correlation_query_cursor import (
    DelegatedCorrelationQueryCursorCodec,
    decode_delegated_correlation_query_cursor,
)

_SORT_PERSISTED_DESC = (
    DocumentDataSort(path=_QUERY_PERSISTED_AT, direction="desc"),
    DocumentDataSort(path="$row_key", direction="desc"),
)


def correlation_query_order_key(
    record: DelegatedInvocationCorrelationRecord,
) -> tuple[datetime, str]:
    return (record.persisted_at, str(record.binding.execution_id))


def record_matches_query_filters(
    record: DelegatedInvocationCorrelationRecord,
    query: DelegatedInvocationCorrelationQuery,
) -> bool:
    binding = record.binding
    if query.parent_execution_id is not None:
        if binding.parent_execution_id != query.parent_execution_id:
            return False
    if query.provider_id is not None and binding.provider_id != query.provider_id:
        return False
    if query.run_id is not None and binding.run_id != query.run_id:
        return False
    if query.persisted_from is not None and record.persisted_at < query.persisted_from:
        return False
    if query.persisted_to is not None and record.persisted_at > query.persisted_to:
        return False
    return True


def record_after_cursor(
    record: DelegatedInvocationCorrelationRecord,
    *,
    last_persisted_at: datetime,
    last_execution_id: ExecutionId,
) -> bool:
    key = correlation_query_order_key(record)
    cursor_key = (last_persisted_at, str(last_execution_id))
    return key < cursor_key


def _finalize_query_page(
    *,
    collected: list[DelegatedInvocationCorrelationRecord],
    page_size: int,
    more_matches_remain_in_batch: bool,
    backend_cursor_at_start: str | None,
    backend_next_cursor: str | None,
) -> DelegatedInvocationCorrelationQueryStorePage:
    records = tuple(collected[:page_size])
    if not records:
        has_more = backend_next_cursor is not None
        return DelegatedInvocationCorrelationQueryStorePage(
            records=(),
            has_more=has_more,
            backend_continuation_cursor=backend_next_cursor if has_more else None,
        )
    has_more = more_matches_remain_in_batch or backend_next_cursor is not None
    continuation = None
    if has_more:
        if more_matches_remain_in_batch:
            continuation = backend_cursor_at_start
        else:
            continuation = backend_next_cursor
    return DelegatedInvocationCorrelationQueryStorePage(
        records=records,
        has_more=has_more,
        backend_continuation_cursor=continuation,
    )


def _collect_from_documents(
    documents: tuple[DocumentRecord, ...],
    *,
    query: DelegatedInvocationCorrelationQuery,
    keyset_after: tuple[datetime, ExecutionId] | None,
) -> tuple[list[DelegatedInvocationCorrelationRecord], bool]:
    candidates: list[DelegatedInvocationCorrelationRecord] = []
    for document in documents:
        record = _document_to_correlation(document)
        if not record_matches_query_filters(record, query):
            continue
        if keyset_after is not None and not record_after_cursor(
            record,
            last_persisted_at=keyset_after[0],
            last_execution_id=keyset_after[1],
        ):
            continue
        candidates.append(record)
    candidates.sort(key=correlation_query_order_key, reverse=True)
    collected = candidates[: query.page_size]
    more_matches_remain_in_batch = len(candidates) > query.page_size
    return collected, more_matches_remain_in_batch


class InMemoryDelegatedInvocationCorrelationQueryStore(
    DelegatedInvocationCorrelationQueryStore,
):
    """In-memory query adapter sharing backend with the write store."""

    def __init__(self, backend: InMemoryDelegatedInvocationCorrelationBackend) -> None:
        self._backend = backend
        self._cursor_codec = backend.query_cursor_codec

    @property
    def cursor_codec(self) -> DelegatedCorrelationQueryCursorCodec:
        return self._cursor_codec

    def query_page(
        self,
        query: DelegatedInvocationCorrelationQuery,
    ) -> DelegatedInvocationCorrelationQueryStorePage:
        validate_document_query_limit(query.page_size)
        candidates = [
            record
            for record in self._backend.snapshot_records()
            if record_matches_query_filters(record, query)
        ]
        candidates.sort(key=correlation_query_order_key, reverse=True)
        keyset_after: tuple[datetime, ExecutionId] | None = None
        if query.cursor is not None:
            last_persisted_at, last_execution_id, _document_cursor = (
                decode_delegated_correlation_query_cursor(
                    codec=self._cursor_codec,
                    query=query,
                    cursor=query.cursor,
                )
            )
            if last_persisted_at is not None and last_execution_id is not None:
                keyset_after = (last_persisted_at, last_execution_id)
                candidates = [
                    record
                    for record in candidates
                    if record_after_cursor(
                        record,
                        last_persisted_at=last_persisted_at,
                        last_execution_id=last_execution_id,
                    )
                ]
        page_records = candidates[: query.page_size]
        has_more = len(candidates) > query.page_size
        return DelegatedInvocationCorrelationQueryStorePage(
            records=tuple(page_records),
            has_more=has_more,
            backend_continuation_cursor=None,
        )


class DocumentStoreDelegatedInvocationCorrelationQueryStore(
    DelegatedInvocationCorrelationQueryStore,
):
    """Bounded document-store query over persisted correlation documents."""

    def __init__(
        self,
        document_store: ConditionalDocumentStore,
        *,
        cursor_secret: bytes | None = None,
    ) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "delegated correlation query requires ConditionalDocumentStore",
            )
        self._document_store = document_store
        secret = cursor_secret
        if secret is None:
            raise TypeError(
                "delegated correlation query requires cursor_secret for durable store",
            )
        self._cursor_codec = DelegatedCorrelationQueryCursorCodec(secret=secret)

    @property
    def cursor_codec(self) -> DelegatedCorrelationQueryCursorCodec:
        return self._cursor_codec

    def query_page(
        self,
        query: DelegatedInvocationCorrelationQuery,
    ) -> DelegatedInvocationCorrelationQueryStorePage:
        validate_document_query_limit(query.page_size)
        scan_limit = delegated_correlation_backend_scan_limit(query.page_size)
        keyset_after: tuple[datetime, ExecutionId] | None = None
        backend_cursor: str | None = None
        if query.cursor is not None:
            last_persisted_at, last_execution_id, backend_cursor = (
                decode_delegated_correlation_query_cursor(
                    codec=self._cursor_codec,
                    query=query,
                    cursor=query.cursor,
                )
            )
            if last_persisted_at is not None and last_execution_id is not None:
                keyset_after = (last_persisted_at, last_execution_id)

        try:
            page = self._document_store.query(
                _DOCUMENT_PARTITION,
                limit=scan_limit,
                cursor=backend_cursor,
                sort=_SORT_PERSISTED_DESC,
            )
        except ValueError as exc:
            if "document_store_cursor" in str(exc):
                raise DelegatedInvocationCorrelationPersistenceError(
                    DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
                ) from exc
            raise DelegatedInvocationCorrelationPersistenceError(
                DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
            ) from exc
        except Exception as exc:
            raise DelegatedInvocationCorrelationPersistenceError(
                DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
            ) from exc

        collected, more_in_batch = _collect_from_documents(
            page.documents,
            query=query,
            keyset_after=keyset_after,
        )
        return _finalize_query_page(
            collected=collected,
            page_size=query.page_size,
            more_matches_remain_in_batch=more_in_batch,
            backend_cursor_at_start=backend_cursor,
            backend_next_cursor=page.next_cursor,
        )


def wire_delegated_invocation_correlation_query_store(
    *,
    durability_mode: DelegatedInvocationCorrelationDurabilityMode,
    document_store: ConditionalDocumentStore | None = None,
    in_memory_backend: InMemoryDelegatedInvocationCorrelationBackend | None = None,
    cursor_secret: bytes | None = None,
) -> DelegatedInvocationCorrelationQueryStore:
    if durability_mode is DelegatedInvocationCorrelationDurabilityMode.DISABLED:
        raise DelegatedInvocationCorrelationCompositionError(
            "delegated correlation query unavailable when durability is disabled",
        )
    if durability_mode is DelegatedInvocationCorrelationDurabilityMode.REQUIRED:
        if document_store is None:
            raise DelegatedInvocationCorrelationCompositionError(
                "durable correlation query requires document store",
            )
        if cursor_secret is None:
            raise DelegatedInvocationCorrelationCompositionError(
                "durable correlation query requires cursor_secret",
            )
        return DocumentStoreDelegatedInvocationCorrelationQueryStore(
            document_store,
            cursor_secret=cursor_secret,
        )
    if in_memory_backend is None:
        in_memory_backend = InMemoryDelegatedInvocationCorrelationBackend()
    return InMemoryDelegatedInvocationCorrelationQueryStore(in_memory_backend)


def paired_in_memory_correlation_stores() -> tuple[
    InMemoryDelegatedInvocationCorrelationStore,
    InMemoryDelegatedInvocationCorrelationQueryStore,
]:
    backend = InMemoryDelegatedInvocationCorrelationBackend()
    return (
        InMemoryDelegatedInvocationCorrelationStore(backend),
        InMemoryDelegatedInvocationCorrelationQueryStore(backend),
    )


__all__ = [
    "DocumentStoreDelegatedInvocationCorrelationQueryStore",
    "InMemoryDelegatedInvocationCorrelationQueryStore",
    "correlation_query_order_key",
    "paired_in_memory_correlation_stores",
    "record_after_cursor",
    "record_matches_query_filters",
    "wire_delegated_invocation_correlation_query_store",
]
