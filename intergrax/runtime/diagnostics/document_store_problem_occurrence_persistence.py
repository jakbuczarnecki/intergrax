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
    ProblemOccurrenceAppendResult,
    ProblemOccurrencePage,
    ProblemOccurrencePersistence,
    ProblemOccurrencePersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.problem_occurrence_query import (
    ProblemOccurrenceQueryCursorCodec,
    _MIN_OCCURRENCE_CURSOR_SECRET_BYTES,
)
from intergrax.runtime.diagnostics.problem_occurrence_record_codec import (
    decode_problem_occurrence_record,
    encode_problem_occurrence_record,
)

_PARTITION_PREFIX = "intergrax.diagnostic_problem_occurrence.v1"
_OCCURRENCE_ROW_PREFIX = "occ:"
_MAX_QUERY_PAGE_LIMIT = 5000
_MAX_OCCURRENCE_PAGE_LIMIT = 1000
_MAX_SORT_MICROS = 10**16
_UTC_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


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
    return _datetime_to_epoch_micros(observed_at)


def _datetime_to_epoch_micros(value: datetime) -> int:
    normalized = value.astimezone(UTC)
    delta = normalized - _UTC_EPOCH
    return (
        delta.days * 86_400 * 1_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )


def _validate_occurrence_cursor_secret(secret: bytes) -> None:
    if not isinstance(secret, bytes) or not secret:
        raise ValueError("problem_occurrence_cursor_secret_invalid")
    if len(secret) < _MIN_OCCURRENCE_CURSOR_SECRET_BYTES:
        raise ValueError("problem_occurrence_cursor_secret_too_short")


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
        _validate_occurrence_cursor_secret(occurrence_cursor_secret)
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
        created = self._document_store.put_if_absent(document)
        if created:
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
    _validate_occurrence_cursor_secret(occurrence_cursor_secret)
    return DocumentStoreProblemOccurrencePersistence(
        document_store,
        occurrence_cursor_secret=occurrence_cursor_secret,
    )
