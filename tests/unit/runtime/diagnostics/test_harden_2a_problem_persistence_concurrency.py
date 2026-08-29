# © Artur Czarnecki. All rights reserved.

"""HARDEN-2A — deterministic multi-instance Problem persistence concurrency proofs."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    STRATEGY_VERSION,
)
from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    DocumentStoreProblemPersistence,
)
from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_reconciliation_key,
    _sample_subject_ref,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    DeterministicProblemSignature,
    ProblemGroupingMethod,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemOccurrence,
    ProblemStatus,
    mint_problem_id,
)
from intergrax.runtime.diagnostics.problem_persistence import (
    ProblemPersistenceConflictError,
    ProblemPersistenceIntegrityError,
    ProblemPersistenceIntegrityReason,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from testing_support.barrier_conditional_document_store import BarrierConditionalDocumentStore

pytestmark = pytest.mark.unit

_OBSERVED_AT = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
_OBSERVED_AT_A = _OBSERVED_AT + timedelta(minutes=1)
_OBSERVED_AT_B = _OBSERVED_AT + timedelta(minutes=2)


def _append_occurrence(
    base: Problem,
    *,
    subject_ref,
    observed_at: datetime,
) -> Problem:
    new_occurrence = ProblemOccurrence(
        subject_ref=subject_ref,
        observed_at=observed_at,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        method=ProblemGroupingMethod.DETERMINISTIC,
    )
    merged_occurrences = base.occurrences + (new_occurrence,)
    merged_subject_refs = base.current_subject_refs
    if subject_ref not in merged_subject_refs:
        merged_subject_refs = base.current_subject_refs + (subject_ref,)
    return Problem(
        problem_id=base.problem_id,
        tenant_id=base.tenant_id,
        status=ProblemStatus.OPEN,
        first_seen_at=min(base.first_seen_at, observed_at),
        last_seen_at=max(base.last_seen_at, observed_at),
        occurrence_count=base.occurrence_count + 1,
        current_subject_refs=merged_subject_refs,
        occurrences=merged_occurrences,
        provenance=base.provenance,
        record_version=base.record_version + 1,
    )


def test_harden_2a_create_race_same_reconciliation_identity_one_logical_problem() -> None:
    """
  HARDEN-2A create race:

  two independent persistence instances, same reconciliation identity, different problem_id.
  """
    store = InMemoryDocumentStore()
    tenant_id = "harden-2a-create-tenant"
    shared_key = _sample_reconciliation_key(tenant_id=tenant_id)
    first = sample_problem(tenant_id=tenant_id, reconciliation_key=shared_key)
    second = sample_problem(
        tenant_id=tenant_id,
        reconciliation_key=shared_key,
        problem_id=mint_problem_id(),
    )
    assert first.problem_id != second.problem_id

    entry_barrier = threading.Barrier(2)
    results: list[Problem] = []
    errors: list[BaseException] = []

    def _create(candidate: Problem) -> None:
        persistence = DocumentStoreProblemPersistence(store)
        try:
            entry_barrier.wait(timeout=5)
            results.append(persistence.create(candidate))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            persistence.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_create, first), executor.submit(_create, second)]
        for future in futures:
            future.result(timeout=10)

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ProblemPersistenceConflictError)
    winner = results[0]
    assert winner in (first, second)

    verifier = DocumentStoreProblemPersistence(store)
    try:
        listed = verifier.list_for_tenant(tenant_id)
        assert listed == (winner,)
        by_key = verifier.find_by_reconciliation_key(
            tenant_id=tenant_id,
            reconciliation_key=shared_key,
        )
        assert by_key == winner
        loser = second if winner == first else first
        assert verifier.get(tenant_id=tenant_id, problem_id=loser.problem_id) is None
    finally:
        verifier.close()


def test_harden_2a_update_race_distinct_occurrences_cas_conflict_not_silent_success() -> None:
    """
  HARDEN-2A update race:

  baseline occurrence_count=1; two writers add different occurrences from the same version.
  """
    store = BarrierConditionalDocumentStore(
        replace_if_match_barrier=threading.Barrier(2),
    )
    tenant_id = "harden-2a-update-tenant"
    baseline_subject = _sample_subject_ref(tenant_id=tenant_id)
    subject_a = _sample_subject_ref(tenant_id=tenant_id)
    subject_b = _sample_subject_ref(tenant_id=tenant_id)
    baseline = sample_problem(tenant_id=tenant_id, subject_refs=(baseline_subject,))
    assert baseline.occurrence_count == 1
    assert baseline.record_version == 1

    seed = DocumentStoreProblemPersistence(store)
    seed.create(baseline)
    seed.close()

    update_a = _append_occurrence(
        baseline,
        subject_ref=subject_a,
        observed_at=_OBSERVED_AT_A,
    )
    update_b = _append_occurrence(
        baseline,
        subject_ref=subject_b,
        observed_at=_OBSERVED_AT_B,
    )

    entry_barrier = threading.Barrier(2)
    results: list[Problem] = []
    errors: list[BaseException] = []

    def _update(candidate: Problem) -> None:
        persistence = DocumentStoreProblemPersistence(store)
        try:
            entry_barrier.wait(timeout=5)
            results.append(persistence.update(candidate, expected_version=1))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            persistence.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_update, update_a), executor.submit(_update, update_b)]
        for future in futures:
            future.result(timeout=10)

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ProblemPersistenceConflictError)

    verifier = DocumentStoreProblemPersistence(store)
    try:
        final = verifier.get(tenant_id=tenant_id, problem_id=baseline.problem_id)
        assert final is not None
        assert final.record_version == 2
        assert final.occurrence_count == 2
        assert len(final.occurrences) == 2
        winner = results[0]
        assert final == winner
        loser_subject = subject_b if winner == update_a else subject_a
        assert all(
            occurrence.subject_ref != loser_subject for occurrence in final.occurrences
        )
        assert verifier.find_by_reconciliation_key(
            tenant_id=tenant_id,
            reconciliation_key=baseline.provenance.reconciliation_key,
        ) == final
    finally:
        verifier.close()


def test_harden_2a_concurrent_create_tenant_isolation() -> None:
    """Tenant A concurrency must not create or index Problems for tenant B."""
    store = InMemoryDocumentStore()
    tenant_a = "harden-2a-tenant-a"
    tenant_b = "harden-2a-tenant-b"
    record_a = sample_problem(tenant_id=tenant_a)
    record_b = sample_problem(tenant_id=tenant_b)
    barrier = threading.Barrier(2)
    results: list[Problem] = []

    def _create(candidate: Problem) -> None:
        persistence = DocumentStoreProblemPersistence(store)
        try:
            barrier.wait(timeout=5)
            results.append(persistence.create(candidate))
        finally:
            persistence.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_create, record_a), executor.submit(_create, record_b)]
        for future in futures:
            future.result(timeout=10)

    assert {item.tenant_id for item in results} == {tenant_a, tenant_b}
    verifier = DocumentStoreProblemPersistence(store)
    try:
        assert verifier.list_for_tenant(tenant_a) == (record_a,)
        assert verifier.list_for_tenant(tenant_b) == (record_b,)
        assert (
            verifier.find_by_reconciliation_key(
                tenant_id=tenant_b,
                reconciliation_key=record_a.provenance.reconciliation_key,
            )
            is None
        )
    finally:
        verifier.close()


def test_harden_2a_create_race_different_reconciliation_identities_remain_distinct() -> None:
    """Sanity: unrelated reconciliation identities may coexist under concurrent create."""
    store = InMemoryDocumentStore()
    tenant_id = "harden-2a-distinct-create"
    key_a = _sample_reconciliation_key(tenant_id=tenant_id)
    key_b = _sample_reconciliation_key(
        tenant_id=tenant_id,
        signature=DeterministicProblemSignature(findings=(), limitations=()),
    )
    first = sample_problem(tenant_id=tenant_id, reconciliation_key=key_a)
    second = sample_problem(tenant_id=tenant_id, reconciliation_key=key_b)
    barrier = threading.Barrier(2)
    results: list[Problem] = []

    def _create(candidate: Problem) -> None:
        persistence = DocumentStoreProblemPersistence(store)
        try:
            barrier.wait(timeout=5)
            results.append(persistence.create(candidate))
        finally:
            persistence.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_create, first), executor.submit(_create, second)]
        for future in futures:
            future.result(timeout=10)

    assert len(results) == 2
    verifier = DocumentStoreProblemPersistence(store)
    try:
        assert verifier.list_for_tenant(tenant_id) == tuple(
            sorted(results, key=lambda item: str(item.problem_id)),
        )
    finally:
        verifier.close()


def test_harden_2a_orphan_reconciliation_index_raises_typed_canonical_pending_reason() -> None:
    store = InMemoryDocumentStore()
    record = sample_problem(tenant_id="harden-2a-orphan-reconcile")
    partition_key = f"intergrax.diagnostic_problem.v1:{record.tenant_id}"
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=f"reconcile:{record.provenance.reconciliation_key.index_token()}",
            data={
                "schema_version": "intergrax.diagnostic_problem.index.v1",
                "problem_id": str(record.problem_id),
            },
        )
    )
    persistence = DocumentStoreProblemPersistence(store)
    try:
        with pytest.raises(ProblemPersistenceIntegrityError) as exc_info:
            persistence.find_by_reconciliation_key(
                tenant_id=record.tenant_id,
                reconciliation_key=record.provenance.reconciliation_key,
            )
        exc = exc_info.value
        assert (
            exc.reason
            is ProblemPersistenceIntegrityReason.RECONCILIATION_WINNER_CANONICAL_PENDING
        )
        assert "canonical Problem record missing" in str(exc)
    finally:
        persistence.close()
