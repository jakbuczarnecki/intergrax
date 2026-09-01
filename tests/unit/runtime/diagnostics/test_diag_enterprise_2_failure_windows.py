# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-2 write-protocol failure windows (R4 aggregate model)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_reconciliation_key,
    _sample_signature,
    _sample_subject_ref,
    sample_occurrences,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_lifecycle import Problem, ProblemLifecycleEngine
from intergrax.runtime.diagnostics.problem_occurrence_aggregate_reconciliation import (
    scan_occurrence_aggregate,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemOccurrenceAggregateHealth
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrenceAppendResult,
    ProblemOccurrencePersistence,
)
from intergrax.runtime.diagnostics.problem_persistence import (
    ProblemPersistence,
    ProblemPersistenceConflictError,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_occurrence_persistence_for_tests,
    document_store_problem_persistence_for_tests,
    lifecycle_engine_for_tests,
)

pytestmark = pytest.mark.unit

_TENANT = "diag-enterprise-2-failure-windows"
_OBSERVED_AT = datetime(2026, 8, 31, 9, 0, tzinfo=UTC)


class _FailAfterOccurrenceAppendPersistence:
    def __init__(
        self,
        delegate: ProblemPersistence,
        *,
        fail_once: bool = True,
    ) -> None:
        self._delegate = delegate
        self._fail_once = fail_once
        self._failed = False

    def get(self, *, tenant_id: str, problem_id):
        return self._delegate.get(tenant_id=tenant_id, problem_id=problem_id)

    def query_problems(self, *, tenant_id: str, status=None, limit: int, cursor=None):
        return self._delegate.query_problems(
            tenant_id=tenant_id,
            status=status,
            limit=limit,
            cursor=cursor,
        )

    def find_by_reconciliation_key(self, *, tenant_id: str, reconciliation_key):
        return self._delegate.find_by_reconciliation_key(
            tenant_id=tenant_id,
            reconciliation_key=reconciliation_key,
        )

    def find_by_subject_ref(self, *, tenant_id: str, subject_ref):
        return self._delegate.find_by_subject_ref(
            tenant_id=tenant_id,
            subject_ref=subject_ref,
        )

    def create(self, record: Problem, *, indexed_subject_refs=()):
        return self._delegate.create(record, indexed_subject_refs=indexed_subject_refs)

    def update(self, record: Problem, *, expected_version: int, indexed_subject_refs=()):
        if self._fail_once and not self._failed and record.occurrence_count > 1:
            self._failed = True
            raise ProblemPersistenceConflictError("simulated aggregate update failure")
        return self._delegate.update(
            record,
            expected_version=expected_version,
            indexed_subject_refs=indexed_subject_refs,
        )

    def close(self) -> None:
        self._delegate.close()


class _FailOccurrenceAppendPersistence:
    def __init__(self, delegate: ProblemOccurrencePersistence) -> None:
        self._delegate = delegate
        self._failed_once = False

    def append_if_absent(self, *, tenant_id: str, problem_id, occurrence):
        if not self._failed_once:
            self._failed_once = True
            raise RuntimeError("simulated occurrence append failure")
        return self._delegate.append_if_absent(
            tenant_id=tenant_id,
            problem_id=problem_id,
            occurrence=occurrence,
        )

    def capture_occurrence_repair_boundary(self, *, tenant_id: str, problem_id):
        return self._delegate.capture_occurrence_repair_boundary(
            tenant_id=tenant_id,
            problem_id=problem_id,
        )

    def query_occurrences(self, *, tenant_id: str, problem_id, limit: int, cursor=None, repair_boundary=None):
        return self._delegate.query_occurrences(
            tenant_id=tenant_id,
            problem_id=problem_id,
            limit=limit,
            cursor=cursor,
            repair_boundary=repair_boundary,
        )


def test_f1_occurrence_success_aggregate_fail_converges_on_retry() -> None:
    store = InMemoryDocumentStore()
    delegate = document_store_problem_persistence_for_tests(store)
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    failing = _FailAfterOccurrenceAppendPersistence(delegate)
    engine = lifecycle_engine_for_tests(failing, occurrence_persistence, document_store=store)

    subject = _sample_subject_ref(tenant_id=_TENANT)
    baseline = sample_problem(
        tenant_id=_TENANT,
        subject_refs=(subject,),
        reconciliation_key=_sample_reconciliation_key(tenant_id=_TENANT),
        observed_at=_OBSERVED_AT,
    )
    failing.create(baseline, indexed_subject_refs=(subject,))
    occurrence_persistence.append_if_absent(
        tenant_id=_TENANT,
        problem_id=baseline.problem_id,
        occurrence=sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0],
    )

    new_subject = _sample_subject_ref(tenant_id=_TENANT)
    from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
        STRATEGY_ID,
        STRATEGY_VERSION,
    )
    from intergrax.runtime.diagnostics.problem_grouping import (
        DeterministicProblemGroupingBasis,
        ProblemGroupingCandidate,
        ProblemGroupingMethod,
        ProblemGroupingProvenance,
        ProblemGroupingResult,
    )

    basis = DeterministicProblemGroupingBasis(signature=_sample_signature())
    candidate = ProblemGroupingCandidate(
        members=(subject, new_subject),
        provenance=ProblemGroupingProvenance(
            strategy_id=STRATEGY_ID,
            strategy_version=STRATEGY_VERSION,
            method=ProblemGroupingMethod.DETERMINISTIC,
            supporting_subject_refs=(subject, new_subject),
            basis=basis,
        ),
    )
    grouping = ProblemGroupingResult(
        tenant_id=_TENANT,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        method=ProblemGroupingMethod.DETERMINISTIC,
        candidates=(candidate,),
        ungrouped_subjects=(),
    )
    observed_at = _OBSERVED_AT + timedelta(minutes=1)
    engine.reconcile(grouping, observed_at=observed_at)

    final = delegate.get(tenant_id=_TENANT, problem_id=baseline.problem_id)
    scan = scan_occurrence_aggregate(
        occurrence_persistence,
        tenant_id=_TENANT,
        problem_id=baseline.problem_id,
    )
    assert scan.occurrence_count == 2
    assert final is not None
    assert final.occurrence_count == 2
    assert final.occurrence_aggregate_health is ProblemOccurrenceAggregateHealth.CONSISTENT


def test_f4_duplicate_append_does_not_double_count() -> None:
    occurrence_persistence = document_store_occurrence_persistence_for_tests(
        InMemoryDocumentStore(),
    )
    subject = _sample_subject_ref(tenant_id=_TENANT)
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), observed_at=_OBSERVED_AT)
    occurrence = sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0]
    assert (
        occurrence_persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )
        is ProblemOccurrenceAppendResult.CREATED
    )
    assert (
        occurrence_persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )
        is ProblemOccurrenceAppendResult.ALREADY_EXISTS
    )
    scan = scan_occurrence_aggregate(
        occurrence_persistence,
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert scan.occurrence_count == 1


def test_f5_concurrent_duplicate_append_counts_once() -> None:
    occurrence_persistence = document_store_occurrence_persistence_for_tests(
        InMemoryDocumentStore(),
    )
    subject = _sample_subject_ref(tenant_id=_TENANT)
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), observed_at=_OBSERVED_AT)
    occurrence = sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0]
    barrier = threading.Barrier(2)
    results: list[ProblemOccurrenceAppendResult] = []

    def _append() -> None:
        barrier.wait(timeout=5)
        results.append(
            occurrence_persistence.append_if_absent(
                tenant_id=_TENANT,
                problem_id=problem.problem_id,
                occurrence=occurrence,
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_append), executor.submit(_append)]
        for future in futures:
            future.result(timeout=10)
    scan = scan_occurrence_aggregate(
        occurrence_persistence,
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert scan.occurrence_count == 1
