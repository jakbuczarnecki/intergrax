# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""ConditionalDocumentStore-backed functional evidence persistence (DIAG-DURABILITY-D1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import AttemptId, RunId, TaskId, validate_run_id, validate_task_id
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentQueryCursorCodec,
    DocumentRecord,
    DocumentStore,
)
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
    PlatformFunctionalEvidence,
)
from intergrax.runtime.diagnostics.functional_evidence_execution_index import (
    DecodedExecutionIndexV2,
    decode_execution_index_v1,
    decode_execution_index_v2,
    encode_execution_index_v1,
    encode_execution_index_v2,
    execution_index_v1_row_key,
    execution_index_v2_row_key,
    execution_index_v2_row_key_from_evidence,
    execution_index_v2_row_key_prefix,
    index_v2_matches_filters,
)
from intergrax.runtime.diagnostics.functional_evidence_index_rebuilder import (
    FunctionalEvidenceIndexRebuilder,
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
_QUERY_PAGE_LIMIT = 5000
_QUERY_OVERFETCH_FACTOR = 4


@runtime_checkable
class DocumentStoreQueryCursorProvider(Protocol):
    @property
    def query_cursor_codec(self) -> DocumentQueryCursorCodec:
        """Authenticated codec for document-store query continuation cursors."""


def _document_partition(tenant_id: str) -> str:
    return f"{_DOCUMENT_STORE_PARTITION_PREFIX}:{tenant_id}"


def _record_row_key(evidence_id: str) -> str:
    return f"{_RECORD_ROW_PREFIX}{evidence_id}"


class DocumentStoreFunctionalEvidencePersistence(FunctionalEvidencePersistence):
    """ConditionalDocumentStore-backed append-only functional evidence store."""

    def __init__(
        self,
        document_store: ConditionalDocumentStore,
        *,
        cursor_secret: bytes,
        query_page_limit: int = _QUERY_PAGE_LIMIT,
        document_query_cursor_codec: DocumentQueryCursorCodec | None = None,
    ) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "functional evidence persistence requires ConditionalDocumentStore",
            )
        if not isinstance(cursor_secret, bytes) or not cursor_secret:
            raise ValueError("functional_evidence_cursor_secret_invalid")
        self._document_store = document_store
        self._cursor_codec = FunctionalEvidenceQueryCursorCodec(secret=cursor_secret)
        self._document_query_cursor_codec = self._resolve_document_query_cursor_codec(
            document_store,
            document_query_cursor_codec,
        )
        if type(query_page_limit) is not int or isinstance(query_page_limit, bool):
            raise ValueError("functional_evidence_query_page_limit_invalid")
        if query_page_limit < 1:
            raise ValueError("functional_evidence_query_page_limit_invalid")
        self._query_page_limit = query_page_limit
        self._index_rebuilder = FunctionalEvidenceIndexRebuilder(
            document_store,
            query_page_limit=query_page_limit,
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
            "functional evidence persistence requires document store query cursor codec",
        )

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
            self._ensure_execution_indexes(evidence=evidence, partition_key=partition_key)
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
        partition_key = _document_partition(tenant_id)
        self._index_rebuilder.ensure_v2_projection(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            partition_key=partition_key,
        )
        store_cursor = _resolve_store_cursor(
            cursor=request.cursor,
            cursor_codec=self._cursor_codec,
            document_query_cursor_codec=self._document_query_cursor_codec,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            kind=request.kind,
            partition_key=partition_key,
        )
        items, has_more = self._collect_bounded_query_page(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            partition_key=partition_key,
            attempt_id=attempt_id,
            kind=request.kind,
            page_size=page_size,
            store_cursor=store_cursor,
        )
        next_cursor = _resolve_next_cursor(
            items=items,
            has_more=has_more,
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

    def _collect_bounded_query_page(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        partition_key: str,
        attempt_id: AttemptId | None,
        kind: PipelineEvidenceKind | None,
        page_size: int,
        store_cursor: str | None,
    ) -> tuple[tuple[PlatformFunctionalEvidence, ...], bool]:
        row_key_prefix = execution_index_v2_row_key_prefix(task_id=task_id, run_id=run_id)
        collected: list[PlatformFunctionalEvidence] = []
        last_consumed_row_key: str | None = None
        continuation = store_cursor

        while len(collected) < page_size:
            fetch_limit = min(
                max(page_size, (page_size - len(collected)) * _QUERY_OVERFETCH_FACTOR),
                self._query_page_limit,
            )
            page = self._document_store.query(
                partition_key,
                limit=fetch_limit,
                row_key_prefix=row_key_prefix,
                cursor=continuation,
            )
            if not page.documents:
                return tuple(collected), False

            for index_document in page.documents:
                last_consumed_row_key = index_document.row_key
                indexed = self._decode_v2_index_document(index_document)
                if not index_v2_matches_filters(
                    indexed,
                    attempt_id=attempt_id,
                    kind=kind,
                ):
                    continue
                evidence = self._resolve_v2_index_document(
                    index_document,
                    indexed=indexed,
                    tenant_id=tenant_id,
                    task_id=task_id,
                    run_id=run_id,
                    partition_key=partition_key,
                )
                collected.append(evidence)
                if len(collected) >= page_size:
                    break

            if len(collected) >= page_size:
                has_more = self._index_has_more_matching_after(
                    partition_key=partition_key,
                    row_key_prefix=row_key_prefix,
                    after_row_key=last_consumed_row_key,
                    attempt_id=attempt_id,
                    kind=kind,
                )
                return tuple(collected), has_more

            if page.next_cursor is None:
                return tuple(collected), False

            continuation = page.next_cursor

        return tuple(collected), False

    def _index_has_more_matching_after(
        self,
        *,
        partition_key: str,
        row_key_prefix: str,
        after_row_key: str | None,
        attempt_id: AttemptId | None,
        kind: PipelineEvidenceKind | None,
    ) -> bool:
        if after_row_key is None:
            return False
        store_cursor = self._document_query_cursor_codec.encode(
            partition_key=partition_key,
            row_key_prefix=row_key_prefix,
            last_row_key=after_row_key,
        )
        continuation = store_cursor
        while True:
            page = self._document_store.query(
                partition_key,
                limit=self._query_page_limit,
                row_key_prefix=row_key_prefix,
                cursor=continuation,
            )
            if not page.documents:
                return False
            for index_document in page.documents:
                indexed = self._decode_v2_index_document(index_document)
                if index_v2_matches_filters(
                    indexed,
                    attempt_id=attempt_id,
                    kind=kind,
                ):
                    return True
            if page.next_cursor is None:
                return False
            continuation = page.next_cursor

    def _decode_v2_index_document(self, index_document: DocumentRecord) -> DecodedExecutionIndexV2:
        try:
            return decode_execution_index_v2(dict(index_document.data))
        except ValueError as exc:
            raise FunctionalEvidencePersistenceIntegrityError(
                "invalid functional evidence execution index",
            ) from exc

    def _resolve_v2_index_document(
        self,
        index_document: DocumentRecord,
        *,
        indexed: DecodedExecutionIndexV2,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        partition_key: str,
    ) -> PlatformFunctionalEvidence:
        record = self._document_store.get(
            partition_key,
            _record_row_key(indexed.evidence_id),
        )
        if record is None:
            raise FunctionalEvidencePersistenceIntegrityError(
                "canonical functional evidence record missing for index",
            )
        evidence = self._document_to_evidence(record)
        if str(evidence.evidence_id) != indexed.evidence_id:
            raise FunctionalEvidencePersistenceIntegrityError(
                "canonical functional evidence id does not match index reference",
            )
        _validate_execution_scope(
            evidence,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
        )
        order_key = functional_evidence_query_order_key(evidence)
        if order_key != (indexed.recorded_at, indexed.evidence_id):
            raise FunctionalEvidencePersistenceIntegrityError(
                "functional evidence index metadata inconsistent with canonical record",
            )
        if evidence.kind is not indexed.kind:
            raise FunctionalEvidencePersistenceIntegrityError(
                "functional evidence index metadata inconsistent with canonical record",
            )
        if evidence.scope.attempt_id != indexed.attempt_id:
            raise FunctionalEvidencePersistenceIntegrityError(
                "functional evidence index metadata inconsistent with canonical record",
            )
        expected_row_key = execution_index_v2_row_key(
            task_id=task_id,
            run_id=run_id,
            recorded_at=indexed.recorded_at,
            evidence_id=indexed.evidence_id,
        )
        if index_document.row_key != expected_row_key:
            raise FunctionalEvidencePersistenceIntegrityError(
                "functional evidence execution index row key inconsistent with metadata",
            )
        return evidence

    def _execution_index_v1_document(
        self,
        *,
        evidence: PlatformFunctionalEvidence,
        partition_key: str,
    ) -> DocumentRecord:
        return DocumentRecord(
            partition_key=partition_key,
            row_key=execution_index_v1_row_key(
                task_id=evidence.scope.task_id,
                run_id=evidence.scope.run_id,
                evidence_id=str(evidence.evidence_id),
            ),
            data=encode_execution_index_v1(str(evidence.evidence_id)),
        )

    def _execution_index_v2_document(
        self,
        *,
        evidence: PlatformFunctionalEvidence,
        partition_key: str,
    ) -> DocumentRecord:
        return DocumentRecord(
            partition_key=partition_key,
            row_key=execution_index_v2_row_key_from_evidence(evidence),
            data=encode_execution_index_v2(evidence),
        )

    def _ensure_execution_indexes(
        self,
        *,
        evidence: PlatformFunctionalEvidence,
        partition_key: str,
    ) -> None:
        for index_document in (
            self._execution_index_v1_document(evidence=evidence, partition_key=partition_key),
            self._execution_index_v2_document(evidence=evidence, partition_key=partition_key),
        ):
            if not self._document_store.put_if_absent(index_document):
                self._verify_index_document(index_document, evidence)

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
        self._ensure_execution_indexes(evidence=stored, partition_key=partition_key)
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
        if document.row_key.startswith("execidx:"):
            indexed = decode_execution_index_v2(dict(existing.data))
            if indexed.evidence_id != str(incoming.evidence_id):
                raise FunctionalEvidencePersistenceIntegrityError(
                    "functional evidence index conflicts with expected evidence_id",
                )
            return
        indexed_id = decode_execution_index_v1(dict(existing.data)).evidence_id
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


def _resolve_store_cursor(
    *,
    cursor: str | None,
    cursor_codec: FunctionalEvidenceQueryCursorCodec,
    document_query_cursor_codec: DocumentQueryCursorCodec,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId | None,
    kind: PipelineEvidenceKind | None,
    partition_key: str,
) -> str | None:
    if cursor is None:
        return None
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
    row_key_prefix = execution_index_v2_row_key_prefix(task_id=task_id, run_id=run_id)
    last_row_key = execution_index_v2_row_key(
        task_id=task_id,
        run_id=run_id,
        recorded_at=payload.last_recorded_at,
        evidence_id=payload.last_evidence_id,
    )
    return document_query_cursor_codec.encode(
        partition_key=partition_key,
        row_key_prefix=row_key_prefix,
        last_row_key=last_row_key,
    )


def _resolve_next_cursor(
    *,
    items: tuple[PlatformFunctionalEvidence, ...],
    has_more: bool,
    cursor_codec: FunctionalEvidenceQueryCursorCodec,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId | None,
    kind: PipelineEvidenceKind | None,
) -> str | None:
    if not items or not has_more:
        return None
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
    "DocumentStoreQueryCursorProvider",
    "wire_functional_evidence_persistence",
]
