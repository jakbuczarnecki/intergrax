# © Artur Czarnecki. All rights reserved.

"""Delegated invocation correlation query adapters (P2.1-S2C3)."""

from __future__ import annotations

from datetime import datetime

from intergrax.contracts.delegated_execution_query import (
    DelegatedInvocationCorrelationQuery,
    DelegatedInvocationCorrelationQueryStore,
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
    DocumentDataEquality,
    DocumentDataSort,
    validate_document_query_limit,
)
from intergrax.runtime.execution.delegated_execution.correlation_persistence import (
    InMemoryDelegatedInvocationCorrelationBackend,
    InMemoryDelegatedInvocationCorrelationStore,
    _DOCUMENT_PARTITION,
    _QUERY_PARENT_EXECUTION_ID,
    _QUERY_PERSISTED_AT,
    _QUERY_PROVIDER_ID,
    _document_to_correlation,
)
from intergrax.runtime.execution.delegated_execution.correlation_query_cursor import (
    decode_delegated_correlation_query_cursor,
    encode_delegated_correlation_query_cursor,
)

_QUERY_OVERFETCH_FACTOR = 4
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


class InMemoryDelegatedInvocationCorrelationQueryStore(
    DelegatedInvocationCorrelationQueryStore,
):
    """In-memory query adapter sharing backend with the write store."""

    def __init__(self, backend: InMemoryDelegatedInvocationCorrelationBackend) -> None:
        self._backend = backend

    def query_correlations(
        self,
        query: DelegatedInvocationCorrelationQuery,
    ) -> tuple[DelegatedInvocationCorrelationRecord, ...]:
        candidates = [
            record
            for record in self._backend.snapshot_records()
            if record_matches_query_filters(record, query)
        ]
        candidates.sort(key=correlation_query_order_key, reverse=True)
        if query.cursor is not None:
            last_persisted_at, last_execution_id, _document_cursor = (
                decode_delegated_correlation_query_cursor(
                    query=query,
                    cursor=query.cursor,
                )
            )
            candidates = [
                record
                for record in candidates
                if record_after_cursor(
                    record,
                    last_persisted_at=last_persisted_at,
                    last_execution_id=last_execution_id,
                )
            ]
        return tuple(candidates[: query.page_size])

    def has_more_after_page(
        self,
        query: DelegatedInvocationCorrelationQuery,
        last_record: DelegatedInvocationCorrelationRecord,
    ) -> bool:
        continuation = query.model_copy(
            update={
                "cursor": encode_delegated_correlation_query_cursor(
                    query=query,
                    last_persisted_at=last_record.persisted_at,
                    last_execution_id=last_record.binding.execution_id,
                ),
                "page_size": 1,
            },
        )
        return bool(self.query_correlations(continuation))


class DocumentStoreDelegatedInvocationCorrelationQueryStore(
    DelegatedInvocationCorrelationQueryStore,
):
    """Bounded document-store query over denormalized correlation index fields."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "delegated correlation query requires ConditionalDocumentStore",
            )
        self._document_store = document_store

    def query_correlations(
        self,
        query: DelegatedInvocationCorrelationQuery,
    ) -> tuple[DelegatedInvocationCorrelationRecord, ...]:
        validate_document_query_limit(query.page_size)
        equalities = _document_equalities_for_query(query)
        keyset_after: tuple[datetime, ExecutionId] | None = None
        document_cursor: str | None = None
        if query.cursor is not None:
            last_persisted_at, last_execution_id, document_cursor = (
                decode_delegated_correlation_query_cursor(
                    query=query,
                    cursor=query.cursor,
                )
            )
            keyset_after = (last_persisted_at, last_execution_id)

        collected: list[DelegatedInvocationCorrelationRecord] = []
        continuation = document_cursor

        while len(collected) < query.page_size:
            fetch_limit = min(
                max(
                    query.page_size,
                    (query.page_size - len(collected)) * _QUERY_OVERFETCH_FACTOR,
                ),
                500,
            )
            try:
                page = self._document_store.query(
                    _DOCUMENT_PARTITION,
                    limit=fetch_limit,
                    cursor=continuation,
                    data_equalities=equalities,
                    sort=_SORT_PERSISTED_DESC,
                )
            except Exception as exc:
                raise DelegatedInvocationCorrelationPersistenceError(
                    DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
                ) from exc
            if not page.documents:
                break
            for document in page.documents:
                record = _document_to_correlation(document)
                if not record_matches_query_filters(record, query):
                    continue
                if keyset_after is not None and not record_after_cursor(
                    record,
                    last_persisted_at=keyset_after[0],
                    last_execution_id=keyset_after[1],
                ):
                    continue
                collected.append(record)
                if len(collected) >= query.page_size:
                    break
            if len(collected) >= query.page_size:
                break
            if page.next_cursor is None:
                break
            continuation = page.next_cursor
        return tuple(collected[: query.page_size])

    def has_more_after_page(
        self,
        query: DelegatedInvocationCorrelationQuery,
        last_record: DelegatedInvocationCorrelationRecord,
    ) -> bool:
        continuation = query.model_copy(
            update={
                "cursor": encode_delegated_correlation_query_cursor(
                    query=query,
                    last_persisted_at=last_record.persisted_at,
                    last_execution_id=last_record.binding.execution_id,
                ),
                "page_size": 1,
            },
        )
        return bool(self.query_correlations(continuation))


def _document_equalities_for_query(
    query: DelegatedInvocationCorrelationQuery,
) -> tuple[DocumentDataEquality, ...]:
    equalities: list[DocumentDataEquality] = []
    if query.parent_execution_id is not None:
        equalities.append(
            DocumentDataEquality(
                path=_QUERY_PARENT_EXECUTION_ID,
                value=str(query.parent_execution_id),
            ),
        )
    if query.provider_id is not None:
        equalities.append(
            DocumentDataEquality(
                path=_QUERY_PROVIDER_ID,
                value=query.provider_id,
            ),
        )
    return tuple(equalities)


def wire_delegated_invocation_correlation_query_store(
    *,
    durability_mode: DelegatedInvocationCorrelationDurabilityMode,
    document_store: ConditionalDocumentStore | None = None,
    in_memory_backend: InMemoryDelegatedInvocationCorrelationBackend | None = None,
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
        return DocumentStoreDelegatedInvocationCorrelationQueryStore(document_store)
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
