# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""ConditionalDocumentStore-backed Problem persistence (DIAG-STORAGE)."""

from __future__ import annotations

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)
from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingSubjectRef
from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemId,
    ProblemReconciliationKey,
    reconciliation_keys_equal,
)
from intergrax.runtime.diagnostics.problem_persistence import (
    ProblemPersistence,
    ProblemPersistenceConflictError,
    ProblemPersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.problem_record_codec import (
    decode_problem_record,
    encode_problem_record,
)

_DOCUMENT_STORE_PARTITION_PREFIX = "intergrax.diagnostic_problem.v1"
_RECORD_ROW_PREFIX = "record:"
_RECONCILE_ROW_PREFIX = "reconcile:"
_SUBJECT_ROW_PREFIX = "subject:"
_QUERY_PAGE_LIMIT = 5000
_INDEX_SCHEMA = "intergrax.diagnostic_problem.index.v1"
_PROBLEM_ID_FIELD = "problem_id"


def _document_partition(tenant_id: str) -> str:
    return f"{_DOCUMENT_STORE_PARTITION_PREFIX}:{tenant_id}"


def _record_row_key(problem_id: ProblemId) -> str:
    return f"{_RECORD_ROW_PREFIX}{problem_id}"


def _reconciliation_row_key(reconciliation_key: ProblemReconciliationKey) -> str:
    return f"{_RECONCILE_ROW_PREFIX}{reconciliation_key.index_token()}"


def _subject_row_key(subject_ref: ProblemGroupingSubjectRef) -> str:
    return f"{_SUBJECT_ROW_PREFIX}{subject_ref.task_id}:{subject_ref.run_id}"


def _encode_index_ref(problem_id: ProblemId) -> dict[str, str]:
    return {
        "schema_version": _INDEX_SCHEMA,
        _PROBLEM_ID_FIELD: str(problem_id),
    }


def _decode_index_ref(data: object) -> ProblemId:
    if not isinstance(data, dict):
        raise ProblemPersistenceIntegrityError("invalid diagnostic problem index")
    schema_version = data.get("schema_version")
    if schema_version != _INDEX_SCHEMA:
        raise ProblemPersistenceIntegrityError(
            "unsupported diagnostic problem index schema",
        )
    problem_id = data.get(_PROBLEM_ID_FIELD)
    if not isinstance(problem_id, str) or not problem_id:
        raise ProblemPersistenceIntegrityError(
            "invalid diagnostic problem index reference",
        )
    return ProblemId(problem_id)


class DocumentStoreProblemPersistence(ProblemPersistence):
    """ConditionalDocumentStore-backed durable Problem store."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "problem persistence requires ConditionalDocumentStore",
            )
        self._document_store = document_store

    def get(self, *, tenant_id: str, problem_id: ProblemId) -> Problem | None:
        partition_key = _document_partition(tenant_id)
        record = self._document_store.get(partition_key, _record_row_key(problem_id))
        if record is None:
            return None
        problem = decode_problem_record(dict(record.data))
        self._verify_canonical_tenant(problem, tenant_id=tenant_id)
        if problem.problem_id != problem_id:
            raise ProblemPersistenceIntegrityError(
                "canonical Problem id does not match lookup key",
            )
        return problem

    def list_for_tenant(self, tenant_id: str) -> tuple[Problem, ...]:
        partition_key = _document_partition(tenant_id)
        documents: list[DocumentRecord] = []
        cursor: str | None = None
        while True:
            page = self._document_store.query(
                partition_key,
                limit=_QUERY_PAGE_LIMIT,
                row_key_prefix=_RECORD_ROW_PREFIX,
                cursor=cursor,
            )
            documents.extend(page.documents)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor

        problems: list[Problem] = []
        for document in documents:
            problem = decode_problem_record(dict(document.data))
            self._verify_canonical_tenant(problem, tenant_id=tenant_id)
            problems.append(problem)
        problems.sort(key=lambda item: str(item.problem_id))
        return tuple(problems)

    def find_by_reconciliation_key(
        self,
        *,
        tenant_id: str,
        reconciliation_key: ProblemReconciliationKey,
    ) -> Problem | None:
        partition_key = _document_partition(tenant_id)
        index_document = self._document_store.get(
            partition_key,
            _reconciliation_row_key(reconciliation_key),
        )
        if index_document is None:
            return None
        problem_id = _decode_index_ref(dict(index_document.data))
        record = self._document_store.get(partition_key, _record_row_key(problem_id))
        if record is None:
            raise ProblemPersistenceIntegrityError(
                "canonical Problem record missing for reconciliation index",
            )
        problem = decode_problem_record(dict(record.data))
        self._verify_canonical_tenant(problem, tenant_id=tenant_id)
        if problem.problem_id != problem_id:
            raise ProblemPersistenceIntegrityError(
                "canonical Problem id does not match reconciliation index reference",
            )
        if not reconciliation_keys_equal(
            problem.provenance.reconciliation_key,
            reconciliation_key,
        ):
            raise ProblemPersistenceIntegrityError(
                "canonical Problem reconciliation key does not match index scope",
            )
        return problem

    def find_by_subject_ref(
        self,
        *,
        tenant_id: str,
        subject_ref: ProblemGroupingSubjectRef,
    ) -> Problem | None:
        if subject_ref.tenant_id != tenant_id:
            raise ProblemPersistenceIntegrityError(
                "subject_ref tenant_id does not match lookup tenant scope",
            )
        partition_key = _document_partition(tenant_id)
        index_document = self._document_store.get(
            partition_key,
            _subject_row_key(subject_ref),
        )
        if index_document is None:
            return None
        problem_id = _decode_index_ref(dict(index_document.data))
        record = self._document_store.get(partition_key, _record_row_key(problem_id))
        if record is None:
            raise ProblemPersistenceIntegrityError(
                "canonical Problem record missing for subject index",
            )
        problem = decode_problem_record(dict(record.data))
        self._verify_canonical_tenant(problem, tenant_id=tenant_id)
        if problem.problem_id != problem_id:
            raise ProblemPersistenceIntegrityError(
                "canonical Problem id does not match subject index reference",
            )
        if subject_ref not in problem.current_subject_refs:
            raise ProblemPersistenceIntegrityError(
                "canonical Problem does not contain indexed subject_ref",
            )
        return problem

    def create(self, record: Problem) -> Problem:
        partition_key = _document_partition(record.tenant_id)
        self._ensure_reconciliation_index(record=record, partition_key=partition_key)
        for subject_ref in record.current_subject_refs:
            self._ensure_subject_index(
                record=record,
                subject_ref=subject_ref,
                partition_key=partition_key,
            )

        canonical_document = self._canonical_document(record)
        if self._document_store.put_if_absent(canonical_document):
            return record

        existing_record = self._document_store.get(
            partition_key,
            canonical_document.row_key,
        )
        if existing_record is None:
            raise RuntimeError("diagnostic problem persistence create failed")
        return self._resolve_existing_record_and_repair_indexes(
            existing_record,
            record,
            partition_key=partition_key,
        )

    def update(self, record: Problem, *, expected_version: int) -> Problem:
        partition_key = _document_partition(record.tenant_id)
        row_key = _record_row_key(record.problem_id)
        existing_record = self._document_store.get(partition_key, row_key)
        if existing_record is None:
            raise ProblemPersistenceConflictError("Problem does not exist")

        existing = decode_problem_record(dict(existing_record.data))
        self._verify_canonical_tenant(existing, tenant_id=record.tenant_id)
        if existing.problem_id != record.problem_id:
            raise ProblemPersistenceIntegrityError(
                "canonical Problem id does not match update target",
            )
        if existing.record_version != expected_version:
            raise ProblemPersistenceConflictError(
                "optimistic concurrency conflict for Problem",
            )

        replacement = self._canonical_document(record)
        if not self._document_store.replace_if_match(
            expected=existing_record,
            replacement=replacement,
        ):
            raise ProblemPersistenceConflictError(
                "optimistic concurrency conflict for Problem",
            )

        self._ensure_reconciliation_index(record=record, partition_key=partition_key)
        for subject_ref in record.current_subject_refs:
            self._ensure_subject_index(
                record=record,
                subject_ref=subject_ref,
                partition_key=partition_key,
            )
        return record

    def _canonical_document(self, record: Problem) -> DocumentRecord:
        return DocumentRecord(
            partition_key=_document_partition(record.tenant_id),
            row_key=_record_row_key(record.problem_id),
            data=encode_problem_record(record),
        )

    def _reconciliation_index_document(
        self,
        *,
        record: Problem,
        partition_key: str,
    ) -> DocumentRecord:
        return DocumentRecord(
            partition_key=partition_key,
            row_key=_reconciliation_row_key(record.provenance.reconciliation_key),
            data=_encode_index_ref(record.problem_id),
        )

    def _subject_index_document(
        self,
        *,
        record: Problem,
        subject_ref: ProblemGroupingSubjectRef,
        partition_key: str,
    ) -> DocumentRecord:
        return DocumentRecord(
            partition_key=partition_key,
            row_key=_subject_row_key(subject_ref),
            data=_encode_index_ref(record.problem_id),
        )

    def _ensure_reconciliation_index(
        self,
        *,
        record: Problem,
        partition_key: str,
    ) -> None:
        document = self._reconciliation_index_document(
            record=record,
            partition_key=partition_key,
        )
        if self._document_store.put_if_absent(document):
            return
        self._verify_index_document(document, record)

    def _ensure_subject_index(
        self,
        *,
        record: Problem,
        subject_ref: ProblemGroupingSubjectRef,
        partition_key: str,
    ) -> None:
        document = self._subject_index_document(
            record=record,
            subject_ref=subject_ref,
            partition_key=partition_key,
        )
        if self._document_store.put_if_absent(document):
            return
        self._verify_index_document(document, record)

    def _verify_index_document(self, document: DocumentRecord, record: Problem) -> None:
        existing = self._document_store.get(document.partition_key, document.row_key)
        if existing is None:
            raise ProblemPersistenceIntegrityError(
                "diagnostic problem index verification failed",
            )
        indexed_id = _decode_index_ref(dict(existing.data))
        if indexed_id != record.problem_id:
            raise ProblemPersistenceConflictError(
                "reconciliation key already bound to another Problem"
                if document.row_key.startswith(_RECONCILE_ROW_PREFIX)
                else "subject_ref already bound to another Problem",
            )

    def _resolve_existing_record_and_repair_indexes(
        self,
        existing_record: DocumentRecord,
        incoming: Problem,
        *,
        partition_key: str,
    ) -> Problem:
        stored = decode_problem_record(dict(existing_record.data))
        if stored != incoming:
            raise ProblemPersistenceConflictError(
                "conflicting Problem for problem_id",
            )
        self._ensure_reconciliation_index(record=stored, partition_key=partition_key)
        for subject_ref in stored.current_subject_refs:
            self._ensure_subject_index(
                record=stored,
                subject_ref=subject_ref,
                partition_key=partition_key,
            )
        return stored

    @staticmethod
    def _verify_canonical_tenant(problem: Problem, *, tenant_id: str) -> None:
        if problem.tenant_id != tenant_id:
            raise ProblemPersistenceIntegrityError(
                "canonical Problem tenant_id does not match lookup tenant scope",
            )


def wire_problem_persistence(
    *,
    document_store: DocumentStore | None = None,
) -> ProblemPersistence:
    """Platform composition boundary: storage capability → Problem persistence."""
    if document_store is None:
        raise ValueError("wire_problem_persistence requires document_store")
    if not isinstance(document_store, ConditionalDocumentStore):
        raise TypeError("problem persistence requires ConditionalDocumentStore")
    return DocumentStoreProblemPersistence(document_store)
