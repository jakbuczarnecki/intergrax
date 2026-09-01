# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""In-memory Problem persistence (tests, conformance, local lab) — DIAG-5D."""

from __future__ import annotations

from threading import Lock

from intergrax.runtime.diagnostics.diagnostic_subject import diagnostic_subject_index_token
from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingSubjectRef
from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemId,
    ProblemReconciliationKey,
    ProblemStatus,
)
from intergrax.runtime.diagnostics.problem_persistence import (
    ProblemListPage,
    ProblemPersistence,
    ProblemPersistenceConflictError,
    ProblemPersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.problem_list_query import (
    ProblemListQueryCursorCodec,
    problem_list_scope_for_status,
    sort_problems_for_public_list,
)

_TEST_LIST_CURSOR_SECRET = b"deterministic-in-memory-problem-list-cursor-v1"


class InMemoryProblemPersistence(ProblemPersistence):
    def __init__(self, *, list_cursor_secret: bytes = _TEST_LIST_CURSOR_SECRET) -> None:
        self._records: dict[tuple[str, ProblemId], Problem] = {}
        self._by_reconciliation_key: dict[tuple[str, str], ProblemId] = {}
        self._by_subject_ref: dict[tuple[str, str, str, str], ProblemId] = {}
        self._lock = Lock()
        self._list_cursor_codec = ProblemListQueryCursorCodec(secret=list_cursor_secret)

    def get(self, *, tenant_id: str, problem_id: ProblemId) -> Problem | None:
        with self._lock:
            return self._records.get((tenant_id, problem_id))

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
        with self._lock:
            records = [
                record
                for (record_tenant_id, _), record in self._records.items()
                if record_tenant_id == tenant_id
            ]
        if status is not None:
            records = [record for record in records if record.status is status]
        ordered = list(sort_problems_for_public_list(records))

        start_offset = 0
        if cursor is not None:
            store_cursor = self._list_cursor_codec.decode(
                cursor,
                tenant_id=tenant_id,
                status_filter=problem_list_scope_for_status(status),
            )
            start_offset = int(store_cursor)

        selected = tuple(ordered[start_offset : start_offset + limit])
        next_offset = start_offset + len(selected)
        has_more = next_offset < len(ordered)
        next_cursor: str | None = None
        if has_more:
            next_cursor = self._list_cursor_codec.encode(
                tenant_id=tenant_id,
                status_filter=problem_list_scope_for_status(status),
                store_cursor=str(next_offset),
            )
        return ProblemListPage(
            problems=selected,
            next_cursor=next_cursor,
            has_more=has_more,
        )

    def find_by_reconciliation_key(
        self,
        *,
        tenant_id: str,
        reconciliation_key: ProblemReconciliationKey,
    ) -> Problem | None:
        index_key = _reconciliation_index_key(tenant_id, reconciliation_key)
        with self._lock:
            problem_id = self._by_reconciliation_key.get(index_key)
            if problem_id is None:
                return None
            return self._records.get((tenant_id, problem_id))

    def find_by_subject_ref(
        self,
        *,
        tenant_id: str,
        subject_ref: ProblemGroupingSubjectRef,
    ) -> Problem | None:
        index_key = _subject_index_key(tenant_id, subject_ref)
        with self._lock:
            problem_id = self._by_subject_ref.get(index_key)
            if problem_id is None:
                return None
            return self._records.get((tenant_id, problem_id))

    def create(
        self,
        record: Problem,
        *,
        indexed_subject_refs: tuple[ProblemGroupingSubjectRef, ...] = (),
    ) -> Problem:
        with self._lock:
            storage_key = (record.tenant_id, record.problem_id)
            existing = self._records.get(storage_key)
            if existing is not None:
                if existing == record:
                    return existing
                raise ProblemPersistenceConflictError(
                    "conflicting Problem for problem_id",
                )

            reconciliation_index = _reconciliation_index_key(
                record.tenant_id,
                record.provenance.reconciliation_key,
            )
            indexed_problem_id = self._by_reconciliation_key.get(reconciliation_index)
            if indexed_problem_id is not None and indexed_problem_id != record.problem_id:
                raise ProblemPersistenceConflictError(
                    "reconciliation key already bound to another Problem",
                )

            for subject_ref in indexed_subject_refs:
                subject_index = _subject_index_key(record.tenant_id, subject_ref)
                indexed_subject_problem = self._by_subject_ref.get(subject_index)
                if (
                    indexed_subject_problem is not None
                    and indexed_subject_problem != record.problem_id
                ):
                    raise ProblemPersistenceConflictError(
                        "subject_ref already bound to another Problem",
                    )

            self._records[storage_key] = record
            self._by_reconciliation_key[reconciliation_index] = record.problem_id
            for subject_ref in indexed_subject_refs:
                self._by_subject_ref[_subject_index_key(record.tenant_id, subject_ref)] = (
                    record.problem_id
                )
            return record

    def update(
        self,
        record: Problem,
        *,
        expected_version: int,
        indexed_subject_refs: tuple[ProblemGroupingSubjectRef, ...] = (),
    ) -> Problem:
        with self._lock:
            storage_key = (record.tenant_id, record.problem_id)
            existing = self._records.get(storage_key)
            if existing is None:
                raise ProblemPersistenceConflictError("Problem does not exist")
            if existing.record_version != expected_version:
                raise ProblemPersistenceConflictError(
                    "optimistic concurrency conflict for Problem",
                )

            reconciliation_index = _reconciliation_index_key(
                record.tenant_id,
                record.provenance.reconciliation_key,
            )
            indexed_problem_id = self._by_reconciliation_key.get(reconciliation_index)
            if indexed_problem_id is not None and indexed_problem_id != record.problem_id:
                raise ProblemPersistenceConflictError(
                    "reconciliation key already bound to another Problem",
                )

            for subject_ref in indexed_subject_refs:
                subject_index = _subject_index_key(record.tenant_id, subject_ref)
                indexed_subject_problem = self._by_subject_ref.get(subject_index)
                if (
                    indexed_subject_problem is not None
                    and indexed_subject_problem != record.problem_id
                ):
                    raise ProblemPersistenceConflictError(
                        "subject_ref already bound to another Problem",
                    )

            self._records[storage_key] = record
            self._by_reconciliation_key[reconciliation_index] = record.problem_id
            for subject_ref in indexed_subject_refs:
                self._by_subject_ref[_subject_index_key(record.tenant_id, subject_ref)] = (
                    record.problem_id
                )
            return record


def _reconciliation_index_key(
    tenant_id: str,
    reconciliation_key: ProblemReconciliationKey,
) -> tuple[str, str]:
    return (tenant_id, reconciliation_key.index_token())


def _subject_index_key(
    tenant_id: str,
    subject_ref: ProblemGroupingSubjectRef,
) -> tuple[str, str]:
    if subject_ref.tenant_id != tenant_id:
        raise ProblemPersistenceIntegrityError(
            "subject_ref tenant_id does not match lookup tenant scope",
        )
    return (tenant_id, diagnostic_subject_index_token(subject_ref.subject))
