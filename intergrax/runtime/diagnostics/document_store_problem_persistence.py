# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""ConditionalDocumentStore-backed Problem persistence (DIAG-STORAGE)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentQueryCursorCodec,
    DocumentRecord,
    DocumentStore,
)
from intergrax.runtime.diagnostics.diagnostic_subject import diagnostic_subject_index_token
from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingSubjectRef
from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemId,
    ProblemReconciliationKey,
    ProblemStatus,
    reconciliation_keys_equal,
)
from intergrax.runtime.diagnostics.problem_persistence import (
    ProblemListPage,
    ProblemPersistence,
    ProblemPersistenceConflictError,
    ProblemPersistenceIntegrityError,
    ProblemPersistenceIntegrityReason,
)
from intergrax.runtime.diagnostics.problem_list_query import (
    ProblemListQueryCursorCodec,
    ProblemListScope,
    decode_list_index_data,
    encode_list_index_data,
    list_index_row_key,
    list_scopes_for_status,
    problem_list_row_key_prefix,
    problem_list_scope_for_status,
)
from intergrax.runtime.diagnostics.problem_list_index_reconciliation import (
    ProblemListIndexReconciler,
    ProblemListIndexReconciliationPage,
    ProblemListProjectionHealth,
    ProblemListProjectionTelemetrySnapshot,
    projection_health_from_state,
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
_LIST_QUERY_OVERFETCH_FACTOR = 4
_LIST_QUERY_MAX_INDEX_EXAMINED_FACTOR = 16


@runtime_checkable
class DocumentStoreQueryCursorProvider(Protocol):
    @property
    def query_cursor_codec(self) -> DocumentQueryCursorCodec:
        """Authenticated codec for document-store query continuation cursors."""


@dataclass(frozen=True, slots=True)
class _ListIndexUpdatePlan:
    deletes: tuple[DocumentRecord, ...]
    replacements: tuple[tuple[DocumentRecord, DocumentRecord], ...]


def _document_partition(tenant_id: str) -> str:
    return f"{_DOCUMENT_STORE_PARTITION_PREFIX}:{tenant_id}"


def _record_row_key(problem_id: ProblemId) -> str:
    return f"{_RECORD_ROW_PREFIX}{problem_id}"


def _reconciliation_row_key(reconciliation_key: ProblemReconciliationKey) -> str:
    return f"{_RECONCILE_ROW_PREFIX}{reconciliation_key.index_token()}"


def _subject_row_key(subject_ref: ProblemGroupingSubjectRef) -> str:
    return f"{_SUBJECT_ROW_PREFIX}{diagnostic_subject_index_token(subject_ref.subject)}"


def _legacy_execution_subject_row_key(subject_ref: ProblemGroupingSubjectRef) -> str | None:
    execution = subject_ref.execution()
    if execution is None:
        return None
    return f"{_SUBJECT_ROW_PREFIX}{execution.task_id}:{execution.run_id}"


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


class _IndexClaims:
    """Tracks index documents newly inserted by a single persistence invocation."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        self._document_store = document_store
        self._claimed: list[DocumentRecord] = []

    def try_claim(self, document: DocumentRecord) -> bool:
        if self._document_store.put_if_absent(document):
            self._claimed.append(document)
            return True
        return False

    def rollback_all(self, *, partition_key: str) -> None:
        for document in reversed(self._claimed):
            self._rollback_one(document, partition_key=partition_key)

    def rollback_except_required_by(
        self,
        canonical: Problem,
        *,
        partition_key: str,
    ) -> None:
        required_keys = {
            (document.partition_key, document.row_key)
            for document in _required_index_documents(
                canonical,
                partition_key=partition_key,
            )
        }
        for document in reversed(self._claimed):
            if (document.partition_key, document.row_key) in required_keys:
                continue
            self._rollback_one(document, partition_key=partition_key)

    def _rollback_one(self, document: DocumentRecord, *, partition_key: str) -> None:
        if self._document_store.delete_if_match(expected=document):
            return
        existing = self._document_store.get(document.partition_key, document.row_key)
        if existing is None:
            return
        indexed_id = _decode_index_ref(dict(existing.data))
        record = self._document_store.get(partition_key, _record_row_key(indexed_id))
        if record is None:
            raise ProblemPersistenceIntegrityError(
                "orphan diagnostic problem index remains after failed create rollback",
            )


def _required_index_documents(
    canonical: Problem,
    *,
    partition_key: str,
) -> tuple[DocumentRecord, ...]:
    documents = [
        DocumentRecord(
            partition_key=partition_key,
            row_key=_reconciliation_row_key(canonical.provenance.reconciliation_key),
            data=_encode_index_ref(canonical.problem_id),
        ),
    ]
    for subject_ref in canonical.current_subject_refs:
        documents.append(
            DocumentRecord(
                partition_key=partition_key,
                row_key=_subject_row_key(subject_ref),
                data=_encode_index_ref(canonical.problem_id),
            ),
        )
    return tuple(documents)


def _new_subject_refs(
    existing: Problem,
    record: Problem,
) -> tuple[ProblemGroupingSubjectRef, ...]:
    existing_refs = set(existing.current_subject_refs)
    return tuple(
        subject_ref
        for subject_ref in record.current_subject_refs
        if subject_ref not in existing_refs
    )


class DocumentStoreProblemPersistence(ProblemPersistence):
    """ConditionalDocumentStore-backed durable Problem store."""

    def __init__(
        self,
        document_store: ConditionalDocumentStore,
        *,
        list_cursor_secret: bytes,
        document_query_cursor_codec: DocumentQueryCursorCodec | None = None,
    ) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "problem persistence requires ConditionalDocumentStore",
            )
        self._document_store = document_store
        self._list_cursor_codec = ProblemListQueryCursorCodec(secret=list_cursor_secret)
        self._document_query_cursor_codec = self._resolve_document_query_cursor_codec(
            document_store,
            document_query_cursor_codec,
        )
        self._clock: Callable[[], datetime] = lambda: datetime.now(UTC)
        self._list_index_reconciler = ProblemListIndexReconciler(
            document_store=document_store,
            document_query_cursor_codec=self._document_query_cursor_codec,
            clock=self._clock,
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
            "problem persistence requires document store query cursor codec",
        )

    def set_clock_for_tests(self, clock: Callable[[], datetime]) -> None:
        """Inject deterministic UTC clock (tests only)."""
        self._clock = clock
        self._list_index_reconciler.clock = clock

    def reconcile_list_indexes(
        self,
        *,
        tenant_id: str,
        stale_before: datetime,
        scope: ProblemListScope | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> ProblemListIndexReconciliationPage:
        """
        Bounded maintenance reconciliation for derived list index projections.

        Projections newer than ``stale_before`` are never deleted — active writer
        transitions remain safe.
        """
        return self._list_index_reconciler.reconcile_list_indexes(
            tenant_id=tenant_id,
            stale_before=stale_before,
            scope=scope,
            limit=limit,
            cursor=cursor,
        )

    def projection_telemetry_snapshot(self) -> ProblemListProjectionTelemetrySnapshot:
        """Return process-local projection skip/repair counters for operator visibility."""
        return self._list_index_reconciler.telemetry.snapshot()

    def projection_health(self) -> ProblemListProjectionHealth:
        """Return process-local projection health derived from reads and maintenance."""
        return projection_health_from_state(
            telemetry=self._list_index_reconciler.telemetry,
            health_state=self._list_index_reconciler.health_state,
        )

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

    def query_problems(
        self,
        *,
        tenant_id: str,
        status: ProblemStatus | None = None,
        limit: int,
        cursor: str | None = None,
    ) -> ProblemListPage:
        if type(limit) is not int or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive int")
        scope = problem_list_scope_for_status(status)
        partition_key = _document_partition(tenant_id)
        row_key_prefix = problem_list_row_key_prefix(scope)
        store_cursor: str | None = None
        if cursor is not None:
            store_cursor = self._list_cursor_codec.decode(
                cursor,
                tenant_id=tenant_id,
                status_filter=scope,
            )

        self._list_index_reconciler.health_state.last_query_skip_count = 0
        problems, last_index_row_key, has_more = self._collect_bounded_query_page(
            tenant_id=tenant_id,
            partition_key=partition_key,
            row_key_prefix=row_key_prefix,
            expected_status=status,
            limit=limit,
            store_cursor=store_cursor,
        )

        next_cursor: str | None = None
        if has_more and last_index_row_key is not None:
            next_store_cursor = self._document_query_cursor_codec.encode(
                partition_key=partition_key,
                row_key_prefix=row_key_prefix,
                last_row_key=last_index_row_key,
            )
            next_cursor = self._list_cursor_codec.encode(
                tenant_id=tenant_id,
                status_filter=scope,
                store_cursor=next_store_cursor,
            )
        return ProblemListPage(
            problems=problems,
            next_cursor=next_cursor,
            has_more=has_more,
        )

    def _collect_bounded_query_page(
        self,
        *,
        tenant_id: str,
        partition_key: str,
        row_key_prefix: str,
        expected_status: ProblemStatus | None,
        limit: int,
        store_cursor: str | None,
    ) -> tuple[tuple[Problem, ...], str | None, bool]:
        max_examined = limit * _LIST_QUERY_MAX_INDEX_EXAMINED_FACTOR
        examined = 0
        collected: list[Problem] = []
        last_consumed_row_key: str | None = None
        continuation = store_cursor

        while len(collected) < limit and examined < max_examined:
            remaining = max_examined - examined
            fetch_limit = min(
                max(limit, (limit - len(collected)) * _LIST_QUERY_OVERFETCH_FACTOR),
                _QUERY_PAGE_LIMIT,
                remaining,
            )
            page = self._document_store.query(
                partition_key,
                limit=fetch_limit,
                row_key_prefix=row_key_prefix,
                cursor=continuation,
            )
            if not page.documents:
                return tuple(collected), last_consumed_row_key, False

            for index_document in page.documents:
                examined += 1
                last_consumed_row_key = index_document.row_key
                problem = self._resolve_list_index_document(
                    index_document,
                    tenant_id=tenant_id,
                    partition_key=partition_key,
                    expected_status=expected_status,
                )
                if problem is None:
                    if len(collected) >= limit:
                        break
                    if examined >= max_examined:
                        break
                    continue
                collected.append(problem)
                if len(collected) >= limit:
                    break
                if examined >= max_examined:
                    break

            if len(collected) >= limit:
                has_more = self._index_has_more_after(
                    partition_key=partition_key,
                    row_key_prefix=row_key_prefix,
                    after_row_key=last_consumed_row_key,
                )
                return tuple(collected), last_consumed_row_key, has_more

            if page.next_cursor is None:
                return tuple(collected), None, False

            continuation = page.next_cursor

        has_more = (
            last_consumed_row_key is not None
            and self._index_has_more_after(
                partition_key=partition_key,
                row_key_prefix=row_key_prefix,
                after_row_key=last_consumed_row_key,
            )
        )
        return tuple(collected), last_consumed_row_key, has_more

    def _index_has_more_after(
        self,
        *,
        partition_key: str,
        row_key_prefix: str,
        after_row_key: str | None,
    ) -> bool:
        if after_row_key is None:
            return False
        store_cursor = self._document_query_cursor_codec.encode(
            partition_key=partition_key,
            row_key_prefix=row_key_prefix,
            last_row_key=after_row_key,
        )
        probe = self._document_store.query(
            partition_key,
            limit=1,
            row_key_prefix=row_key_prefix,
            cursor=store_cursor,
        )
        return bool(probe.documents)

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
                reason=ProblemPersistenceIntegrityReason.RECONCILIATION_WINNER_CANONICAL_PENDING,
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
            legacy_key = _legacy_execution_subject_row_key(subject_ref)
            if legacy_key is not None:
                index_document = self._document_store.get(partition_key, legacy_key)
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
        claims = _IndexClaims(self._document_store)
        try:
            self._claim_indexes_for_create(
                record=record,
                partition_key=partition_key,
                claims=claims,
            )
        except ProblemPersistenceConflictError:
            claims.rollback_all(partition_key=partition_key)
            raise
        except Exception:
            claims.rollback_all(partition_key=partition_key)
            raise

        canonical_document = self._canonical_document(record)
        try:
            if self._document_store.put_if_absent(canonical_document):
                return record
        except Exception:
            existing_record = self._document_store.get(
                partition_key,
                canonical_document.row_key,
            )
            if existing_record is None:
                claims.rollback_all(partition_key=partition_key)
                raise
            stored = decode_problem_record(dict(existing_record.data))
            if stored == record:
                return self._resolve_existing_record_and_repair_indexes(
                    existing_record,
                    record,
                    partition_key=partition_key,
                )
            claims.rollback_except_required_by(stored, partition_key=partition_key)
            raise ProblemPersistenceConflictError("conflicting Problem for problem_id")

        return self._resolve_canonical_create_race(
            record=record,
            canonical_document=canonical_document,
            partition_key=partition_key,
            claims=claims,
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

        self._verify_reconciliation_index_for_update(
            record=record,
            partition_key=partition_key,
        )
        for subject_ref in existing.current_subject_refs:
            self._verify_subject_index_for_update(
                record=record,
                subject_ref=subject_ref,
                partition_key=partition_key,
            )

        new_subject_refs = _new_subject_refs(existing, record)
        claims = _IndexClaims(self._document_store)
        list_index_plan: _ListIndexUpdatePlan | None = None
        try:
            self._claim_new_subject_indexes_for_update(
                record=record,
                new_subject_refs=new_subject_refs,
                partition_key=partition_key,
                claims=claims,
            )
            list_index_plan = self._prepare_list_index_update(
                existing=existing,
                record=record,
                partition_key=partition_key,
                claims=claims,
            )
        except ProblemPersistenceConflictError:
            claims.rollback_all(partition_key=partition_key)
            raise
        except Exception:
            claims.rollback_all(partition_key=partition_key)
            raise

        replacement = self._canonical_document(record)
        try:
            if self._document_store.replace_if_match(
                expected=existing_record,
                replacement=replacement,
            ):
                if list_index_plan is not None:
                    self._finalize_list_index_update(list_index_plan)
                return record
        except Exception as exc:
            return self._resolve_uncertain_update_cas(
                record=record,
                existing=existing,
                claims=claims,
                partition_key=partition_key,
                original_exc=exc,
                list_index_plan=list_index_plan,
            )

        return self._resolve_update_cas_race(
            record=record,
            existing=existing,
            claims=claims,
            partition_key=partition_key,
            row_key=row_key,
            list_index_plan=list_index_plan,
        )

    def _claim_indexes_for_create(
        self,
        *,
        record: Problem,
        partition_key: str,
        claims: _IndexClaims,
    ) -> None:
        reconciliation_document = self._reconciliation_index_document(
            record=record,
            partition_key=partition_key,
        )
        if not claims.try_claim(reconciliation_document):
            self._verify_index_document(reconciliation_document, record)

        for subject_ref in record.current_subject_refs:
            subject_document = self._subject_index_document(
                record=record,
                subject_ref=subject_ref,
                partition_key=partition_key,
            )
            if not claims.try_claim(subject_document):
                self._verify_index_document(subject_document, record)

        for scope in list_scopes_for_status(record.status):
            list_document = self._list_index_document(
                record=record,
                scope=scope,
                partition_key=partition_key,
            )
            if not claims.try_claim(list_document):
                self._verify_list_index_document(list_document, record)

    def _resolve_canonical_create_race(
        self,
        *,
        record: Problem,
        canonical_document: DocumentRecord,
        partition_key: str,
        claims: _IndexClaims,
    ) -> Problem:
        existing_record = self._document_store.get(
            partition_key,
            canonical_document.row_key,
        )
        if existing_record is None:
            raise RuntimeError("diagnostic problem persistence create failed")
        stored = decode_problem_record(dict(existing_record.data))
        if stored == record:
            return self._resolve_existing_record_and_repair_indexes(
                existing_record,
                record,
                partition_key=partition_key,
            )
        claims.rollback_except_required_by(stored, partition_key=partition_key)
        raise ProblemPersistenceConflictError("conflicting Problem for problem_id")

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

    def _verify_reconciliation_index_for_update(
        self,
        *,
        record: Problem,
        partition_key: str,
    ) -> None:
        document = self._reconciliation_index_document(
            record=record,
            partition_key=partition_key,
        )
        self._verify_index_document(document, record)

    def _verify_subject_index_for_update(
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
        self._verify_index_document(document, record)

    def _claim_new_subject_indexes_for_update(
        self,
        *,
        record: Problem,
        new_subject_refs: tuple[ProblemGroupingSubjectRef, ...],
        partition_key: str,
        claims: _IndexClaims,
    ) -> None:
        for subject_ref in new_subject_refs:
            subject_document = self._subject_index_document(
                record=record,
                subject_ref=subject_ref,
                partition_key=partition_key,
            )
            if not claims.try_claim(subject_document):
                self._verify_index_document(subject_document, record)

    def _resolve_update_cas_race(
        self,
        *,
        record: Problem,
        existing: Problem,
        claims: _IndexClaims,
        partition_key: str,
        row_key: str,
        list_index_plan: _ListIndexUpdatePlan | None,
    ) -> Problem:
        existing_record = self._document_store.get(partition_key, row_key)
        if existing_record is None:
            claims.rollback_all(partition_key=partition_key)
            raise ProblemPersistenceConflictError(
                "optimistic concurrency conflict for Problem",
            )
        stored = decode_problem_record(dict(existing_record.data))
        if stored == record:
            if list_index_plan is not None:
                self._finalize_list_index_update(list_index_plan)
            return self._repair_indexes_for_record(stored, partition_key=partition_key)
        if stored == existing:
            claims.rollback_all(partition_key=partition_key)
        else:
            claims.rollback_except_required_by(stored, partition_key=partition_key)
        raise ProblemPersistenceConflictError(
            "optimistic concurrency conflict for Problem",
        )

    def _resolve_uncertain_update_cas(
        self,
        *,
        record: Problem,
        existing: Problem,
        claims: _IndexClaims,
        partition_key: str,
        original_exc: BaseException,
        list_index_plan: _ListIndexUpdatePlan | None,
    ) -> Problem:
        row_key = _record_row_key(record.problem_id)
        existing_record = self._document_store.get(partition_key, row_key)
        if existing_record is None:
            claims.rollback_all(partition_key=partition_key)
            raise original_exc

        stored = decode_problem_record(dict(existing_record.data))
        if stored == record:
            if list_index_plan is not None:
                self._finalize_list_index_update(list_index_plan)
            return self._repair_indexes_for_record(stored, partition_key=partition_key)
        if stored == existing:
            claims.rollback_all(partition_key=partition_key)
            raise original_exc
        claims.rollback_except_required_by(stored, partition_key=partition_key)
        raise ProblemPersistenceConflictError(
            "optimistic concurrency conflict for Problem",
        ) from original_exc

    def _repair_indexes_for_record(
        self,
        record: Problem,
        *,
        partition_key: str,
    ) -> Problem:
        self._ensure_reconciliation_index(record=record, partition_key=partition_key)
        for subject_ref in record.current_subject_refs:
            self._ensure_subject_index(
                record=record,
                subject_ref=subject_ref,
                partition_key=partition_key,
            )
        self._ensure_list_indexes(record=record, partition_key=partition_key)
        return record

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
        self._ensure_list_indexes(record=stored, partition_key=partition_key)
        return stored

    def _get_stored_list_index_document(
        self,
        *,
        record: Problem,
        scope: ProblemListScope,
        partition_key: str,
    ) -> DocumentRecord:
        row_key = list_index_row_key(scope=scope, problem=record)
        stored = self._document_store.get(partition_key, row_key)
        if stored is None:
            raise ProblemPersistenceIntegrityError(
                "diagnostic problem list index missing for update",
            )
        return stored

    def _list_index_document(
        self,
        *,
        record: Problem,
        scope: ProblemListScope,
        partition_key: str,
    ) -> DocumentRecord:
        return DocumentRecord(
            partition_key=partition_key,
            row_key=list_index_row_key(scope=scope, problem=record),
            data=encode_list_index_data(
                problem_id=record.problem_id,
                last_seen_at=record.last_seen_at,
                status=record.status,
                record_version=record.record_version,
                projection_written_at=self._clock(),
            ),
        )

    def _verify_list_index_document(self, document: DocumentRecord, record: Problem) -> None:
        existing = self._document_store.get(document.partition_key, document.row_key)
        if existing is None:
            raise ProblemPersistenceIntegrityError(
                "diagnostic problem list index verification failed",
            )
        indexed = decode_list_index_data(dict(existing.data))
        if indexed.problem_id != record.problem_id:
            raise ProblemPersistenceConflictError(
                "list index already bound to another Problem",
            )
        if (
            indexed.last_seen_at != record.last_seen_at
            or indexed.status is not record.status
            or indexed.record_version != record.record_version
        ):
            raise ProblemPersistenceConflictError(
                "list index already bound to incompatible Problem metadata",
            )

    def _resolve_list_index_document(
        self,
        index_document: DocumentRecord,
        *,
        tenant_id: str,
        partition_key: str,
        expected_status: ProblemStatus | None,
    ) -> Problem | None:
        try:
            indexed = decode_list_index_data(dict(index_document.data))
        except ValueError as exc:
            raise ProblemPersistenceIntegrityError(
                "invalid diagnostic problem list index",
            ) from exc
        record = self._document_store.get(partition_key, _record_row_key(indexed.problem_id))
        if record is None:
            self._list_index_reconciler.telemetry.skipped_missing_canonical += 1
            self._list_index_reconciler.health_state.last_query_skip_count += 1
            return None
        problem = decode_problem_record(dict(record.data))
        self._verify_canonical_tenant(problem, tenant_id=tenant_id)
        if problem.problem_id != indexed.problem_id:
            raise ProblemPersistenceIntegrityError(
                "canonical Problem id does not match list index reference",
            )
        if problem.record_version > indexed.record_version:
            self._list_index_reconciler.telemetry.skipped_version_behind += 1
            self._list_index_reconciler.health_state.last_query_skip_count += 1
            return None
        if problem.record_version < indexed.record_version:
            self._list_index_reconciler.telemetry.skipped_version_ahead += 1
            self._list_index_reconciler.health_state.last_query_skip_count += 1
            return None
        if problem.last_seen_at != indexed.last_seen_at or problem.status != indexed.status:
            self._list_index_reconciler.telemetry.same_version_integrity_failure += 1
            self._list_index_reconciler.health_state.last_query_skip_count += 1
            raise ProblemPersistenceIntegrityError(
                "diagnostic problem list index metadata inconsistent with canonical Problem",
                reason=ProblemPersistenceIntegrityReason.LIST_INDEX_CANONICAL_METADATA_MISMATCH,
            )
        if expected_status is not None and problem.status is not expected_status:
            return None
        return problem

    def _prepare_list_index_update(
        self,
        *,
        existing: Problem,
        record: Problem,
        partition_key: str,
        claims: _IndexClaims,
    ) -> _ListIndexUpdatePlan:
        old_scopes = set(list_scopes_for_status(existing.status))
        new_scopes = set(list_scopes_for_status(record.status))
        scopes_to_remove = old_scopes - new_scopes
        scopes_to_add = new_scopes - old_scopes
        scopes_to_keep = old_scopes & new_scopes

        deletes: list[DocumentRecord] = []
        replacements: list[tuple[DocumentRecord, DocumentRecord]] = []

        for scope in scopes_to_remove:
            deletes.append(
                self._get_stored_list_index_document(
                    record=existing,
                    scope=scope,
                    partition_key=partition_key,
                ),
            )

        for scope in scopes_to_keep:
            old_document = self._get_stored_list_index_document(
                record=existing,
                scope=scope,
                partition_key=partition_key,
            )
            new_document = self._list_index_document(
                record=record,
                scope=scope,
                partition_key=partition_key,
            )
            if old_document.row_key == new_document.row_key:
                replacements.append((old_document, new_document))
                continue
            deletes.append(old_document)
            if not claims.try_claim(new_document):
                self._verify_list_index_document(new_document, record)

        for scope in scopes_to_add:
            new_document = self._list_index_document(
                record=record,
                scope=scope,
                partition_key=partition_key,
            )
            if not claims.try_claim(new_document):
                self._verify_list_index_document(new_document, record)

        return _ListIndexUpdatePlan(
            deletes=tuple(deletes),
            replacements=tuple(replacements),
        )

    def _finalize_list_index_update(self, plan: _ListIndexUpdatePlan) -> None:
        for expected, replacement in plan.replacements:
            if self._document_store.replace_if_match(
                expected=expected,
                replacement=replacement,
            ):
                continue
            existing = self._document_store.get(
                expected.partition_key,
                expected.row_key,
            )
            if existing is None or dict(existing.data) != dict(replacement.data):
                raise ProblemPersistenceIntegrityError(
                    "diagnostic problem list index update failed",
                )
        for document in plan.deletes:
            if self._document_store.delete_if_match(expected=document):
                continue
            existing = self._document_store.get(document.partition_key, document.row_key)
            if existing is not None:
                raise ProblemPersistenceIntegrityError(
                    "stale diagnostic problem list index entry remains after update",
                )

    def _ensure_list_indexes(self, *, record: Problem, partition_key: str) -> None:
        for scope in list_scopes_for_status(record.status):
            document = self._list_index_document(
                record=record,
                scope=scope,
                partition_key=partition_key,
            )
            if self._document_store.put_if_absent(document):
                continue
            self._verify_list_index_document(document, record)

    @staticmethod
    def _verify_canonical_tenant(problem: Problem, *, tenant_id: str) -> None:
        if problem.tenant_id != tenant_id:
            raise ProblemPersistenceIntegrityError(
                "canonical Problem tenant_id does not match lookup tenant scope",
            )


def wire_problem_persistence(
    *,
    document_store: DocumentStore | None = None,
    list_cursor_secret: bytes,
) -> ProblemPersistence:
    """Platform composition boundary: storage capability → Problem persistence."""
    if document_store is None:
        raise ValueError("wire_problem_persistence requires document_store")
    if not isinstance(document_store, ConditionalDocumentStore):
        raise TypeError("problem persistence requires ConditionalDocumentStore")
    if not isinstance(list_cursor_secret, bytes) or not list_cursor_secret:
        raise ValueError("problem_list_cursor_secret_invalid")
    return DocumentStoreProblemPersistence(
        document_store,
        list_cursor_secret=list_cursor_secret,
    )
