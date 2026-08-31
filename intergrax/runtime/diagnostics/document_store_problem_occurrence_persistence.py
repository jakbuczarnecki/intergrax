# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""DocumentStore-backed durable ProblemOccurrence persistence (DIAG-ENTERPRISE-2)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentQueryCursorCodec,
    DocumentRecord,
    DocumentStore,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemId, ProblemOccurrence
from intergrax.runtime.diagnostics.problem_occurrence_id import (
    ProblemOccurrenceId,
    problem_occurrence_id_for,
)
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrenceAggregateStats,
    ProblemOccurrenceAppendResult,
    ProblemOccurrencePage,
    ProblemOccurrencePersistence,
    ProblemOccurrencePersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.problem_occurrence_query import (
    ProblemOccurrenceQueryCursorCodec,
)
from intergrax.runtime.diagnostics.problem_occurrence_record_codec import (
    decode_occurrence_stats_record,
    decode_problem_occurrence_record,
    encode_occurrence_stats_record,
    encode_problem_occurrence_record,
)

_PARTITION_PREFIX = "intergrax.diagnostic_problem_occurrence.v1"
_OCCURRENCE_ROW_PREFIX = "occ:"
_STATS_ROW_KEY = "meta:stats"
_MAX_QUERY_PAGE_LIMIT = 5000
_MAX_OCCURRENCE_PAGE_LIMIT = 1000
_MAX_SORT_MICROS = 10**16


@runtime_checkable
class DocumentStoreQueryCursorProvider(Protocol):
    @property
    def query_cursor_codec(self) -> DocumentQueryCursorCodec:
        """Authenticated codec for document-store query continuation cursors."""


def _occurrence_partition(tenant_id: str, problem_id: ProblemId) -> str:
    return f"{_PARTITION_PREFIX}:{tenant_id}:{problem_id}"


def _occurrence_row_key(
    occurrence: ProblemOccurrence,
    *,
    occurrence_id: ProblemOccurrenceId,
) -> str:
    observed_micros = _observed_at_micros(occurrence.observed_at)
    sort_token = _MAX_SORT_MICROS - observed_micros
    return f"{_OCCURRENCE_ROW_PREFIX}{sort_token:016d}:{occurrence_id}"


def _observed_at_micros(observed_at: datetime) -> int:
    if observed_at.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return int(observed_at.timestamp() * 1_000_000)


class DocumentStoreProblemOccurrencePersistence(ProblemOccurrencePersistence):
    """ConditionalDocumentStore-backed durable occurrence history per Problem."""

    def __init__(
        self,
        document_store: ConditionalDocumentStore,
        *,
        occurrence_cursor_secret: bytes,
        document_query_cursor_codec: DocumentQueryCursorCodec | None = None,
    ) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "problem occurrence persistence requires ConditionalDocumentStore",
            )
        self._document_store = document_store
        self._occurrence_cursor_codec = ProblemOccurrenceQueryCursorCodec(
            secret=occurrence_cursor_secret,
        )
        self._document_query_cursor_codec = self._resolve_document_query_cursor_codec(
            document_store,
            document_query_cursor_codec,
        )

    @staticmethod
    def _resolve_document_query_cursor_codec(
        document_store: ConditionalDocumentStore,
        document_query_cursor_codec: DocumentQueryCursorCodec | None,
    ) -> DocumentQueryCursorCodec:
        if document_query_cursor_codec is not None:
            return document_query_cursor_codec
        if isinstance(document_store, DocumentStoreQueryCursorProvider):
            return document_store.query_cursor_codec
        raise TypeError(
            "problem occurrence persistence requires document store query cursor codec",
        )

    def append_if_absent(
        self,
        *,
        tenant_id: str,
        problem_id: ProblemId,
        occurrence: ProblemOccurrence,
    ) -> ProblemOccurrenceAppendResult:
        if occurrence.subject_ref.tenant_id != tenant_id:
            raise ProblemOccurrencePersistenceIntegrityError(
                "occurrence tenant_id does not match append tenant scope",
            )
        partition_key = _occurrence_partition(tenant_id, problem_id)
        occurrence_id = problem_occurrence_id_for(occurrence)
        row_key = _occurrence_row_key(occurrence, occurrence_id=occurrence_id)
        document = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data=encode_problem_occurrence_record(occurrence),
        )
        if self._document_store.put_if_absent(document):
            self._merge_stats_on_append(
                partition_key=partition_key,
                occurrence=occurrence,
            )
            return ProblemOccurrenceAppendResult.CREATED

        existing = self._document_store.get(partition_key, row_key)
        if existing is None:
            raise ProblemOccurrencePersistenceIntegrityError(
                "occurrence append race left no durable row",
            )
        stored = decode_problem_occurrence_record(dict(existing.data))
        if stored != occurrence:
            raise ProblemOccurrencePersistenceIntegrityError(
                "conflicting durable occurrence for stable occurrence id",
            )
        return ProblemOccurrenceAppendResult.ALREADY_EXISTS

    def query_occurrences(
        self,
        *,
        tenant_id: str,
        problem_id: ProblemId,
        limit: int,
        cursor: str | None = None,
    ) -> ProblemOccurrencePage:
        if type(limit) is not int or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive int")
        if limit > _MAX_OCCURRENCE_PAGE_LIMIT:
            raise ValueError(f"limit must be <= {_MAX_OCCURRENCE_PAGE_LIMIT}")

        partition_key = _occurrence_partition(tenant_id, problem_id)
        store_cursor: str | None = None
        if cursor is not None:
            store_cursor = self._occurrence_cursor_codec.decode(
                cursor,
                tenant_id=tenant_id,
                problem_id=problem_id,
            )

        page = self._document_store.query(
            partition_key,
            limit=min(limit, _MAX_QUERY_PAGE_LIMIT),
            row_key_prefix=_OCCURRENCE_ROW_PREFIX,
            cursor=store_cursor,
        )
        items: list[ProblemOccurrence] = []
        last_row_key: str | None = None
        for document in page.documents:
            last_row_key = document.row_key
            items.append(decode_problem_occurrence_record(dict(document.data)))

        has_more = page.next_cursor is not None
        next_cursor: str | None = None
        if has_more and last_row_key is not None:
            next_store_cursor = self._document_query_cursor_codec.encode(
                partition_key=partition_key,
                row_key_prefix=_OCCURRENCE_ROW_PREFIX,
                last_row_key=last_row_key,
            )
            next_cursor = self._occurrence_cursor_codec.encode(
                tenant_id=tenant_id,
                problem_id=problem_id,
                store_cursor=next_store_cursor,
            )
        return ProblemOccurrencePage(
            items=tuple(items),
            next_cursor=next_cursor,
            has_more=has_more,
        )

    def aggregate_stats(
        self,
        *,
        tenant_id: str,
        problem_id: ProblemId,
    ) -> ProblemOccurrenceAggregateStats | None:
        partition_key = _occurrence_partition(tenant_id, problem_id)
        stats_record = self._document_store.get(partition_key, _STATS_ROW_KEY)
        if stats_record is None:
            return None
        count, first_seen_at, last_seen_at = decode_occurrence_stats_record(
            dict(stats_record.data),
        )
        return ProblemOccurrenceAggregateStats(
            occurrence_count=count,
            first_seen_at=first_seen_at,
            last_seen_at=last_seen_at,
        )

    def _merge_stats_on_append(
        self,
        *,
        partition_key: str,
        occurrence: ProblemOccurrence,
    ) -> None:
        existing = self._document_store.get(partition_key, _STATS_ROW_KEY)
        if existing is None:
            replacement = DocumentRecord(
                partition_key=partition_key,
                row_key=_STATS_ROW_KEY,
                data=encode_occurrence_stats_record(
                    occurrence_count=1,
                    first_seen_at=occurrence.observed_at,
                    last_seen_at=occurrence.observed_at,
                ),
            )
            if self._document_store.put_if_absent(replacement):
                return
            existing = self._document_store.get(partition_key, _STATS_ROW_KEY)
            if existing is None:
                raise ProblemOccurrencePersistenceIntegrityError(
                    "occurrence stats record missing after concurrent create",
                )

        count, first_seen_at, last_seen_at = decode_occurrence_stats_record(
            dict(existing.data),
        )
        replacement = DocumentRecord(
            partition_key=partition_key,
            row_key=_STATS_ROW_KEY,
            data=encode_occurrence_stats_record(
                occurrence_count=count + 1,
                first_seen_at=min(first_seen_at, occurrence.observed_at),
                last_seen_at=max(last_seen_at, occurrence.observed_at),
            ),
        )
        if not self._document_store.replace_if_match(
            expected=existing,
            replacement=replacement,
        ):
            refreshed = self._document_store.get(partition_key, _STATS_ROW_KEY)
            if refreshed is None:
                raise ProblemOccurrencePersistenceIntegrityError(
                    "occurrence stats record disappeared during merge",
                )
            self._merge_stats_on_append(partition_key=partition_key, occurrence=occurrence)


def wire_problem_occurrence_persistence(
    *,
    document_store: DocumentStore | None = None,
    occurrence_cursor_secret: bytes,
) -> ProblemOccurrencePersistence:
    """Platform composition boundary: storage capability → occurrence persistence."""
    if document_store is None:
        raise ValueError("wire_problem_occurrence_persistence requires document_store")
    if not isinstance(document_store, ConditionalDocumentStore):
        raise TypeError(
            "problem occurrence persistence requires ConditionalDocumentStore",
        )
    if not isinstance(occurrence_cursor_secret, bytes) or not occurrence_cursor_secret:
        raise ValueError("problem_occurrence_cursor_secret_invalid")
    return DocumentStoreProblemOccurrencePersistence(
        document_store,
        occurrence_cursor_secret=occurrence_cursor_secret,
    )
