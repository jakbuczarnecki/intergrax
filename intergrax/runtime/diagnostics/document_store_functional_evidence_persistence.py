# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""ConditionalDocumentStore-backed functional evidence persistence (DIAG-DURABILITY-D1)."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.execution_identity import AttemptId, RunId, TaskId, validate_run_id, validate_task_id
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
    PlatformFunctionalEvidence,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistence,
    FunctionalEvidencePersistenceConflictError,
    FunctionalEvidencePersistenceIntegrityError,
    FunctionalEvidenceQueryPage,
    FunctionalEvidenceQueryRequest,
    functional_evidence_query_order_key,
)
from intergrax.runtime.diagnostics.functional_evidence_query_cursor import (
    FunctionalEvidenceQueryCursorCodec,
    FunctionalEvidenceQueryCursorError,
)
from intergrax.runtime.diagnostics.functional_evidence_record_codec import (
    decode_functional_evidence_record,
    encode_functional_evidence_record,
)

_DOCUMENT_STORE_PARTITION_PREFIX = "intergrax.functional_evidence.v1"
_RECORD_ROW_PREFIX = "record:"
_EXEC_ROW_PREFIX = "exec:"
_QUERY_PAGE_LIMIT = 5000
_INDEX_SCHEMA = "intergrax.functional_evidence.index.v1"
_EVIDENCE_ID_FIELD = "evidence_id"


def _document_partition(tenant_id: str) -> str:
    return f"{_DOCUMENT_STORE_PARTITION_PREFIX}:{tenant_id}"


def _record_row_key(evidence_id: str) -> str:
    return f"{_RECORD_ROW_PREFIX}{evidence_id}"


def _execution_row_key(*, task_id: TaskId, run_id: RunId, evidence_id: str) -> str:
    return f"{_EXEC_ROW_PREFIX}{task_id}:{run_id}:{evidence_id}"


def _execution_row_prefix(*, task_id: TaskId, run_id: RunId) -> str:
    return f"{_EXEC_ROW_PREFIX}{task_id}:{run_id}:"


def _encode_index_ref(evidence_id: str) -> dict[str, str]:
    return {
        "schema_version": _INDEX_SCHEMA,
        _EVIDENCE_ID_FIELD: evidence_id,
    }


def _decode_index_ref(data: object) -> str:
    if not isinstance(data, dict):
        raise FunctionalEvidencePersistenceIntegrityError(
            "invalid functional evidence index",
        )
    schema_version = data.get("schema_version")
    if schema_version != _INDEX_SCHEMA:
        raise FunctionalEvidencePersistenceIntegrityError(
            "unsupported functional evidence index schema",
        )
    evidence_id = data.get(_EVIDENCE_ID_FIELD)
    if not isinstance(evidence_id, str) or not evidence_id:
        raise FunctionalEvidencePersistenceIntegrityError(
            "invalid functional evidence index reference",
        )
    return evidence_id


@dataclass(frozen=True, order=True, slots=True)
class _SortedEntry:
    order_key: tuple[datetime, str]
    evidence_id: str


class DocumentStoreFunctionalEvidencePersistence(FunctionalEvidencePersistence):
    """ConditionalDocumentStore-backed append-only functional evidence store."""

    def __init__(
        self,
        document_store: ConditionalDocumentStore,
        *,
        cursor_secret: bytes,
        query_page_limit: int = _QUERY_PAGE_LIMIT,
    ) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "functional evidence persistence requires ConditionalDocumentStore",
            )
        if not isinstance(cursor_secret, bytes) or not cursor_secret:
            raise ValueError("functional_evidence_cursor_secret_invalid")
        self._document_store = document_store
        self._cursor_codec = FunctionalEvidenceQueryCursorCodec(secret=cursor_secret)
        if type(query_page_limit) is not int or isinstance(query_page_limit, bool):
            raise ValueError("functional_evidence_query_page_limit_invalid")
        if query_page_limit < 1:
            raise ValueError("functional_evidence_query_page_limit_invalid")
        self._query_page_limit = query_page_limit

    @property
    def query_cursor_codec(self) -> FunctionalEvidenceQueryCursorCodec:
        return self._cursor_codec

    def append(self, evidence: PlatformFunctionalEvidence) -> PlatformFunctionalEvidence:
        partition_key = _document_partition(evidence.scope.tenant_id)
        record_row_key = _record_row_key(str(evidence.evidence_id))
        encoded = encode_functional_evidence_record(evidence)
        canonical_document = DocumentRecord(
            partition_key=partition_key,
            row_key=record_row_key,
            data=encoded,
        )

        if self._document_store.put_if_absent(canonical_document):
            self._ensure_execution_index(evidence=evidence, partition_key=partition_key)
            return evidence

        existing_record = self._document_store.get(partition_key, record_row_key)
        if existing_record is None:
            raise RuntimeError("functional evidence persistence append failed")
        return self._resolve_existing_record_and_repair_indexes(
            existing_record,
            evidence,
            partition_key=partition_key,
        )

    def query_evidence(
        self,
        request: FunctionalEvidenceQueryRequest,
    ) -> FunctionalEvidenceQueryPage:
        tenant_id = _require_tenant_id(request.tenant_id)
        task_id = validate_task_id(request.task_id)
        run_id = validate_run_id(request.run_id)
        attempt_id = _validate_attempt_filter(request.attempt_id)
        page_size = _validate_page_size(request.page_size)
        sorted_entries = self._sorted_entries_for_execution(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
        )
        records_by_id = self._records_for_execution(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            sorted_entries=sorted_entries,
        )
        start_pos = _resolve_start_position(
            sorted_entries=sorted_entries,
            cursor=request.cursor,
            cursor_codec=self._cursor_codec,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            kind=request.kind,
        )
        items, scan_pos = _collect_page(
            sorted_entries=sorted_entries,
            records_by_id=records_by_id,
            start_pos=start_pos,
            page_size=page_size,
            attempt_id=attempt_id,
            kind=request.kind,
        )
        next_cursor = _resolve_next_cursor(
            sorted_entries=sorted_entries,
            records_by_id=records_by_id,
            items=items,
            scan_pos=scan_pos,
            cursor_codec=self._cursor_codec,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            kind=request.kind,
        )
        return FunctionalEvidenceQueryPage(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            items=items,
            next_cursor=next_cursor,
        )

    def _sorted_entries_for_execution(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> list[_SortedEntry]:
        partition_key = _document_partition(tenant_id)
        prefix = _execution_row_prefix(task_id=task_id, run_id=run_id)
        documents: list[DocumentRecord] = []
        cursor: str | None = None
        while True:
            page = self._document_store.query(
                partition_key,
                limit=self._query_page_limit,
                row_key_prefix=prefix,
                cursor=cursor,
            )
            documents.extend(page.documents)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor

        entries: list[_SortedEntry] = []
        for document in documents:
            evidence_id = _decode_index_ref(dict(document.data))
            record = self._document_store.get(
                partition_key,
                _record_row_key(evidence_id),
            )
            if record is None:
                raise FunctionalEvidencePersistenceIntegrityError(
                    "canonical functional evidence record missing for index",
                )
            evidence = self._document_to_evidence(record)
            if str(evidence.evidence_id) != evidence_id:
                raise FunctionalEvidencePersistenceIntegrityError(
                    "canonical functional evidence id does not match index reference",
                )
            _validate_execution_scope(
                evidence,
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
            )
            entries.append(
                _SortedEntry(
                    order_key=functional_evidence_query_order_key(evidence),
                    evidence_id=str(evidence.evidence_id),
                )
            )
        entries.sort()
        return entries

    def _records_for_execution(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        sorted_entries: list[_SortedEntry],
    ) -> dict[str, PlatformFunctionalEvidence]:
        partition_key = _document_partition(tenant_id)
        records: dict[str, PlatformFunctionalEvidence] = {}
        for entry in sorted_entries:
            record = self._document_store.get(
                partition_key,
                _record_row_key(entry.evidence_id),
            )
            if record is None:
                raise FunctionalEvidencePersistenceIntegrityError(
                    "canonical functional evidence record missing for index",
                )
            evidence = self._document_to_evidence(record)
            _validate_execution_scope(
                evidence,
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
            )
            records[entry.evidence_id] = evidence
        return records

    def _execution_index_document(
        self,
        *,
        evidence: PlatformFunctionalEvidence,
        partition_key: str,
    ) -> DocumentRecord:
        return DocumentRecord(
            partition_key=partition_key,
            row_key=_execution_row_key(
                task_id=evidence.scope.task_id,
                run_id=evidence.scope.run_id,
                evidence_id=str(evidence.evidence_id),
            ),
            data=_encode_index_ref(str(evidence.evidence_id)),
        )

    def _ensure_execution_index(
        self,
        *,
        evidence: PlatformFunctionalEvidence,
        partition_key: str,
    ) -> None:
        exec_document = self._execution_index_document(
            evidence=evidence,
            partition_key=partition_key,
        )
        if not self._document_store.put_if_absent(exec_document):
            self._verify_index_document(exec_document, evidence)

    def _document_to_evidence(self, document: DocumentRecord) -> PlatformFunctionalEvidence:
        try:
            return decode_functional_evidence_record(dict(document.data))
        except ValueError as exc:
            raise FunctionalEvidencePersistenceIntegrityError(
                "invalid canonical functional evidence record",
            ) from exc

    def _resolve_existing_record_and_repair_indexes(
        self,
        record: DocumentRecord,
        incoming: PlatformFunctionalEvidence,
        *,
        partition_key: str,
    ) -> PlatformFunctionalEvidence:
        stored = self._document_to_evidence(record)
        if stored != incoming:
            raise FunctionalEvidencePersistenceConflictError(
                "conflicting functional evidence for evidence_id",
            )
        self._ensure_execution_index(evidence=stored, partition_key=partition_key)
        return stored

    def _verify_index_document(
        self,
        document: DocumentRecord,
        incoming: PlatformFunctionalEvidence,
    ) -> None:
        existing = self._document_store.get(document.partition_key, document.row_key)
        if existing is None:
            raise FunctionalEvidencePersistenceIntegrityError(
                "functional evidence index verification failed",
            )
        indexed_id = _decode_index_ref(dict(existing.data))
        if indexed_id != str(incoming.evidence_id):
            raise FunctionalEvidencePersistenceIntegrityError(
                "functional evidence index conflicts with expected evidence_id",
            )


def wire_functional_evidence_persistence(
    *,
    document_store: DocumentStore | None = None,
    cursor_secret: bytes,
) -> FunctionalEvidencePersistence:
    """Platform composition boundary: storage capability → functional evidence persistence."""
    if document_store is None:
        raise ValueError("wire_functional_evidence_persistence requires document_store")
    if not isinstance(document_store, ConditionalDocumentStore):
        raise TypeError(
            "functional evidence persistence requires ConditionalDocumentStore",
        )
    if not isinstance(cursor_secret, bytes) or not cursor_secret:
        raise ValueError("functional_evidence_cursor_secret_invalid")
    return DocumentStoreFunctionalEvidencePersistence(
        document_store,
        cursor_secret=cursor_secret,
    )


def _validate_execution_scope(
    evidence: PlatformFunctionalEvidence,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
) -> None:
    if (
        evidence.scope.tenant_id != tenant_id
        or evidence.scope.task_id != task_id
        or evidence.scope.run_id != run_id
    ):
        raise FunctionalEvidencePersistenceIntegrityError(
            "canonical functional evidence does not match execution index scope",
        )


def _resolve_start_position(
    *,
    sorted_entries: list[_SortedEntry],
    cursor: str | None,
    cursor_codec: FunctionalEvidenceQueryCursorCodec,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId | None,
    kind: PipelineEvidenceKind | None,
) -> int:
    if cursor is None:
        return 0
    try:
        payload = cursor_codec.decode(
            cursor,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            kind=kind,
        )
    except FunctionalEvidenceQueryCursorError as exc:
        raise FunctionalEvidencePersistenceIntegrityError(str(exc)) from exc
    cursor_key = (payload.last_recorded_at, payload.last_evidence_id)
    return bisect.bisect_right(
        sorted_entries,
        _SortedEntry(order_key=cursor_key, evidence_id=payload.last_evidence_id),
    )


def _matches_filters(
    record: PlatformFunctionalEvidence,
    *,
    attempt_id: AttemptId | None,
    kind: PipelineEvidenceKind | None,
) -> bool:
    if kind is not None and record.kind is not kind:
        return False
    if attempt_id is not None and record.scope.attempt_id != attempt_id:
        return False
    return True


def _collect_page(
    *,
    sorted_entries: list[_SortedEntry],
    records_by_id: dict[str, PlatformFunctionalEvidence],
    start_pos: int,
    page_size: int,
    attempt_id: AttemptId | None,
    kind: PipelineEvidenceKind | None,
) -> tuple[tuple[PlatformFunctionalEvidence, ...], int]:
    items: list[PlatformFunctionalEvidence] = []
    scan_pos = start_pos
    while scan_pos < len(sorted_entries) and len(items) < page_size:
        entry = sorted_entries[scan_pos]
        scan_pos += 1
        record = records_by_id[entry.evidence_id]
        if not _matches_filters(record, attempt_id=attempt_id, kind=kind):
            continue
        items.append(record)
    return tuple(items), scan_pos


def _resolve_next_cursor(
    *,
    sorted_entries: list[_SortedEntry],
    records_by_id: dict[str, PlatformFunctionalEvidence],
    items: tuple[PlatformFunctionalEvidence, ...],
    scan_pos: int,
    cursor_codec: FunctionalEvidenceQueryCursorCodec,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId | None,
    kind: PipelineEvidenceKind | None,
) -> str | None:
    if not items:
        return None
    if _has_more_matching(
        sorted_entries=sorted_entries,
        records_by_id=records_by_id,
        start_pos=scan_pos,
        attempt_id=attempt_id,
        kind=kind,
    ):
        last_item = items[-1]
        return cursor_codec.encode(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            kind=kind,
            last_recorded_at=last_item.provenance.recorded_at,
            last_evidence_id=last_item.evidence_id,
        )
    return None


def _has_more_matching(
    *,
    sorted_entries: list[_SortedEntry],
    records_by_id: dict[str, PlatformFunctionalEvidence],
    start_pos: int,
    attempt_id: AttemptId | None,
    kind: PipelineEvidenceKind | None,
) -> bool:
    scan_pos = start_pos
    while scan_pos < len(sorted_entries):
        entry = sorted_entries[scan_pos]
        scan_pos += 1
        record = records_by_id[entry.evidence_id]
        if _matches_filters(record, attempt_id=attempt_id, kind=kind):
            return True
    return False


def _require_tenant_id(tenant_id: str) -> str:
    if type(tenant_id) is not str:
        raise FunctionalEvidencePersistenceIntegrityError("tenant_id must be str")
    normalized = tenant_id.strip()
    if not normalized:
        raise FunctionalEvidencePersistenceIntegrityError("tenant_id must be non-empty")
    if tenant_id != normalized:
        raise FunctionalEvidencePersistenceIntegrityError(
            "tenant_id must not contain leading or trailing whitespace",
        )
    return normalized


def _validate_attempt_filter(attempt_id: AttemptId | None) -> AttemptId | None:
    if attempt_id is None:
        return None
    from intergrax.contracts.execution_identity import validate_attempt_id

    return validate_attempt_id(attempt_id)


def _validate_page_size(page_size: int) -> int:
    if type(page_size) is not int or isinstance(page_size, bool):
        raise FunctionalEvidencePersistenceIntegrityError("page_size must be int")
    if page_size < 1:
        raise FunctionalEvidencePersistenceIntegrityError("page_size must be >= 1")
    if page_size > 1000:
        raise FunctionalEvidencePersistenceIntegrityError("page_size must be <= 1000")
    return page_size


__all__ = [
    "DocumentStoreFunctionalEvidencePersistence",
    "wire_functional_evidence_persistence",
]
