# © Artur Czarnecki. All rights reserved.

"""Delegated invocation correlation persistence adapters (P2.1-S2C)."""

from __future__ import annotations

import json
import secrets
import threading
from collections.abc import Mapping

from pydantic import ValidationError

from intergrax.contracts.delegated_correlation_query_index_backfill import (
    DelegatedCorrelationQueryIndexBackfillPage,
    DelegatedCorrelationQueryIndexBackfillRequest,
    MAX_DELEGATED_CORRELATION_BACKEND_PAGES_PER_BACKFILL_CALL,
)
from intergrax.contracts.delegated_execution_query import (
    delegated_correlation_backend_scan_limit,
)
from intergrax.contracts.delegated_invocation_correlation import (
    DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
    DelegatedInvocationCorrelationCompositionError,
    DelegatedInvocationCorrelationConflictError,
    DelegatedInvocationCorrelationDurabilityMode,
    DelegatedInvocationCorrelationIntegrityError,
    DelegatedInvocationCorrelationPersistenceError,
    DelegatedInvocationCorrelationRecord,
    DelegatedInvocationCorrelationStore,
    correlation_records_equivalent,
)
from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentDataSort,
    DocumentRecord,
)
from intergrax.runtime.execution.delegated_execution.correlation_query_cursor import (
    DelegatedCorrelationQueryCursorCodec,
)

_DOCUMENT_PARTITION = "intergrax.delegated_invocation_correlation.v1"
_QUERY_PARENT_EXECUTION_ID = "query_parent_execution_id"
_QUERY_PROVIDER_ID = "query_provider_id"
_QUERY_PERSISTED_AT = "query_persisted_at"
_QUERY_RUN_ID = "query_run_id"


def correlation_document_query_fields(
    record: DelegatedInvocationCorrelationRecord,
) -> dict[str, str]:
    """Denormalized query index fields for bounded DocumentStore queries."""
    return {
        _QUERY_PARENT_EXECUTION_ID: str(record.binding.parent_execution_id),
        _QUERY_PROVIDER_ID: record.binding.provider_id,
        _QUERY_PERSISTED_AT: record.persisted_at.isoformat(),
        _QUERY_RUN_ID: str(record.binding.run_id),
    }


def document_has_complete_query_index(data: Mapping[str, object]) -> bool:
    required = (
        _QUERY_PARENT_EXECUTION_ID,
        _QUERY_PROVIDER_ID,
        _QUERY_PERSISTED_AT,
        _QUERY_RUN_ID,
    )
    return all(isinstance(data.get(key), str) and data.get(key) for key in required)


def encode_correlation_record(record: DelegatedInvocationCorrelationRecord) -> bytes:
    return record.model_dump_json().encode("utf-8")


def decode_correlation_record(raw: bytes) -> DelegatedInvocationCorrelationRecord:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DelegatedInvocationCorrelationIntegrityError(
            "correlation record is not valid JSON",
        ) from exc
    try:
        return DelegatedInvocationCorrelationRecord.model_validate(payload)
    except ValidationError as exc:
        raise DelegatedInvocationCorrelationIntegrityError(
            "correlation record failed schema validation",
        ) from exc


class InMemoryDelegatedInvocationCorrelationBackend:
    """Shared in-memory record map for write and query adapters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, DelegatedInvocationCorrelationRecord] = {}
        self.query_cursor_codec = DelegatedCorrelationQueryCursorCodec(
            secret=secrets.token_bytes(32),
        )

    def snapshot_records(self) -> tuple[DelegatedInvocationCorrelationRecord, ...]:
        with self._lock:
            return tuple(self._records.values())

    def get_record(self, key: str) -> DelegatedInvocationCorrelationRecord | None:
        with self._lock:
            return self._records.get(key)

    def persist_record(self, record: DelegatedInvocationCorrelationRecord) -> None:
        key = str(record.binding.execution_id)
        with self._lock:
            existing = self._records.get(key)
            if existing is None:
                self._records[key] = record
                return
            if correlation_records_equivalent(existing, record):
                return
            raise DelegatedInvocationCorrelationConflictError(
                "delegated invocation correlation conflict for execution_id",
            )


class InMemoryDelegatedInvocationCorrelationStore(DelegatedInvocationCorrelationStore):
    """Shared-backend in-memory store for tests and single-process hosts."""

    def __init__(
        self,
        backend: InMemoryDelegatedInvocationCorrelationBackend | None = None,
    ) -> None:
        self._backend = backend or InMemoryDelegatedInvocationCorrelationBackend()

    @property
    def is_durable(self) -> bool:
        return False

    def persist(self, record: DelegatedInvocationCorrelationRecord) -> None:
        self._backend.persist_record(record)

    def get_by_execution_id(
        self,
        execution_id: ExecutionId,
    ) -> DelegatedInvocationCorrelationRecord | None:
        key = str(validate_execution_id(execution_id))
        return self._backend.get_record(key)


class DocumentStoreDelegatedInvocationCorrelationStore(
    DelegatedInvocationCorrelationStore,
):
    """ConditionalDocumentStore-backed delegated invocation correlation."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "delegated invocation correlation requires ConditionalDocumentStore",
            )
        self._document_store = document_store

    @property
    def is_durable(self) -> bool:
        return True

    def persist(self, record: DelegatedInvocationCorrelationRecord) -> None:
        row_key = str(record.binding.execution_id)
        existing_doc = self._document_store.get(_DOCUMENT_PARTITION, row_key)
        if existing_doc is not None:
            existing = _document_to_correlation(existing_doc)
            if correlation_records_equivalent(existing, record):
                return
            raise DelegatedInvocationCorrelationConflictError(
                "delegated invocation correlation conflict for execution_id",
            )
        document = DocumentRecord(
            partition_key=_DOCUMENT_PARTITION,
            row_key=row_key,
            data={
                "correlation": encode_correlation_record(record).decode("utf-8"),
                **correlation_document_query_fields(record),
            },
        )
        if not self._document_store.put_if_absent(document):
            stored = self.get_by_execution_id(record.binding.execution_id)
            if stored is not None and correlation_records_equivalent(stored, record):
                return
            raise DelegatedInvocationCorrelationConflictError(
                "delegated invocation correlation conflict for execution_id",
            )

    def get_by_execution_id(
        self,
        execution_id: ExecutionId,
    ) -> DelegatedInvocationCorrelationRecord | None:
        row_key = str(validate_execution_id(execution_id))
        document = self._document_store.get(_DOCUMENT_PARTITION, row_key)
        if document is None:
            return None
        return _document_to_correlation(document)


def _document_to_correlation(
    document: DocumentRecord,
) -> DelegatedInvocationCorrelationRecord:
    raw = document.data.get("correlation")
    if not isinstance(raw, str):
        raise DelegatedInvocationCorrelationIntegrityError(
            "invalid delegated invocation correlation document",
        )
    return decode_correlation_record(raw.encode("utf-8"))


def backfill_correlation_document_query_index(
    document_store: ConditionalDocumentStore,
    *,
    request: DelegatedCorrelationQueryIndexBackfillRequest | None = None,
    scan_limit: int | None = None,
    cursor: str | None = None,
) -> DelegatedCorrelationQueryIndexBackfillPage:
    """
    Idempotent bounded backfill of query index fields on legacy correlation documents.

    One call performs at most one DocumentStore query and inspects at most ``scan_limit``
    backend rows (not successful updates). Does not run from the query read path.
    """
    if request is None:
        resolved_scan_limit = (
            delegated_correlation_backend_scan_limit(scan_limit)
            if scan_limit is not None
            else delegated_correlation_backend_scan_limit(100)
        )
        request = DelegatedCorrelationQueryIndexBackfillRequest(
            scan_limit=resolved_scan_limit,
            cursor=cursor,
        )
    elif scan_limit is not None or cursor is not None:
        raise TypeError(
            "pass either request=DelegatedCorrelationQueryIndexBackfillRequest "
            "or scan_limit/cursor, not both",
        )
    _ = MAX_DELEGATED_CORRELATION_BACKEND_PAGES_PER_BACKFILL_CALL
    scan_limit = request.scan_limit
    try:
        page = document_store.query(
            _DOCUMENT_PARTITION,
            limit=scan_limit,
            cursor=request.cursor,
            sort=(DocumentDataSort(path="$row_key", direction="asc"),),
        )
    except ValueError as exc:
        raise DelegatedInvocationCorrelationPersistenceError(
            DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
        ) from exc
    except Exception as exc:
        raise DelegatedInvocationCorrelationPersistenceError(
            DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
        ) from exc

    scanned_count = 0
    updated_count = 0
    for document in page.documents:
        scanned_count += 1
        if document_has_complete_query_index(document.data):
            continue
        record = _document_to_correlation(document)
        replacement = DocumentRecord(
            partition_key=document.partition_key,
            row_key=document.row_key,
            data={
                **dict(document.data),
                **correlation_document_query_fields(record),
            },
            ttl_seconds=document.ttl_seconds,
        )
        try:
            replaced = document_store.replace_if_match(
                expected=document,
                replacement=replacement,
            )
        except Exception as exc:
            raise DelegatedInvocationCorrelationPersistenceError(
                DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
            ) from exc
        if replaced:
            updated_count += 1

    return DelegatedCorrelationQueryIndexBackfillPage(
        scanned_count=scanned_count,
        updated_count=updated_count,
        next_cursor=page.next_cursor,
    )


def wire_delegated_invocation_correlation_store(
    *,
    durability_mode: DelegatedInvocationCorrelationDurabilityMode,
    document_store: ConditionalDocumentStore | None = None,
    in_memory_backend: InMemoryDelegatedInvocationCorrelationStore | None = None,
) -> DelegatedInvocationCorrelationStore:
    """Composition helper for correlation store wiring (explicit durability mode)."""
    if durability_mode is DelegatedInvocationCorrelationDurabilityMode.DISABLED:
        raise DelegatedInvocationCorrelationCompositionError(
            "disabled durability mode has no correlation store",
        )
    if durability_mode is DelegatedInvocationCorrelationDurabilityMode.REQUIRED:
        if document_store is None:
            raise DelegatedInvocationCorrelationCompositionError(
                "durable document store required for required correlation durability",
            )
        if in_memory_backend is not None:
            raise DelegatedInvocationCorrelationCompositionError(
                "in-memory backend cannot be combined with required durable correlation",
            )
        return DocumentStoreDelegatedInvocationCorrelationStore(document_store)
    if document_store is not None:
        raise DelegatedInvocationCorrelationCompositionError(
            "document store cannot be used with non-durable test correlation mode",
        )
    if in_memory_backend is not None:
        return in_memory_backend
    return InMemoryDelegatedInvocationCorrelationStore()


__all__ = [
    "DocumentStoreDelegatedInvocationCorrelationStore",
    "InMemoryDelegatedInvocationCorrelationBackend",
    "InMemoryDelegatedInvocationCorrelationStore",
    "_QUERY_RUN_ID",
    "correlation_document_query_fields",
    "backfill_correlation_document_query_index",
    "document_has_complete_query_index",
    "decode_correlation_record",
    "encode_correlation_record",
    "wire_delegated_invocation_correlation_store",
]
