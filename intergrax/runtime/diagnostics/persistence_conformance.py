# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Conformance harness for diagnostic Problem persistence backends (DIAG-STORAGE)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from intergrax.contracts.execution_identity import RunId, TaskId, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    STRATEGY_VERSION,
)
from intergrax.runtime.diagnostics.deterministic_problem_reconciliation import (
    DeterministicProblemReconciliationKey,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticFindingKind
from intergrax.runtime.diagnostics.lifecycle_analysis import (
    LifecycleAnomalyKind,
    LifecycleAnomalyScope,
    LifecycleViolationTransition,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    DeterministicFindingSignature,
    DeterministicProblemSignature,
    ProblemGroupingMethod,
    ProblemGroupingSubjectRef,
    problem_grouping_subject_ref_for_execution,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemId,
    ProblemLifecycleProvenance,
    ProblemOccurrence,
    ProblemStatus,
    mint_problem_id,
)
from intergrax.runtime.diagnostics.problem_persistence import (
    ProblemPersistence,
    ProblemPersistenceConflictError,
    ProblemPersistenceIntegrityError,
)
from intergrax.runtime.events.asof_projection import (
    RunExecutionLifecycleStatus,
    RunLifecycleViolationKind,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType

_OBSERVED_AT = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)
_OBSERVED_AT_LATER = _OBSERVED_AT + timedelta(hours=1)


def _sample_signature() -> DeterministicProblemSignature:
    return DeterministicProblemSignature(
        findings=(
            DeterministicFindingSignature(
                kind=DiagnosticFindingKind.DISALLOWED_AFTER_FAILED,
                scope=LifecycleAnomalyScope.EXECUTION,
                source_anomaly_kind=LifecycleAnomalyKind.DISALLOWED_AFTER_FAILED,
                lifecycle_transition=LifecycleViolationTransition(
                    violation_kind=RunLifecycleViolationKind.DISALLOWED_AFTER_FAILED,
                    prior_status=RunExecutionLifecycleStatus.COMPLETED,
                    violating_event_type=RuntimeEventType.RETRY_SCHEDULED,
                ),
            ),
        ),
        limitations=(),
    )


def _sample_reconciliation_key(
    *,
    tenant_id: str,
    signature: DeterministicProblemSignature | None = None,
) -> DeterministicProblemReconciliationKey:
    return DeterministicProblemReconciliationKey(
        tenant_id=tenant_id,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        signature=signature or _sample_signature(),
    )


def _sample_subject_ref(
    *,
    tenant_id: str,
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
) -> ProblemGroupingSubjectRef:
    return problem_grouping_subject_ref_for_execution(
        tenant_id=tenant_id,
        task_id=task_id or mint_task_id(),
        run_id=run_id or mint_run_id(),
    )


def sample_problem(
    *,
    problem_id: ProblemId | None = None,
    tenant_id: str = "tenant-conformance",
    subject_refs: tuple[ProblemGroupingSubjectRef, ...] | None = None,
    reconciliation_key: DeterministicProblemReconciliationKey | None = None,
    observed_at: datetime = _OBSERVED_AT,
    record_version: int = 1,
    status: ProblemStatus = ProblemStatus.OPEN,
    occurrence_count: int | None = None,
) -> Problem:
    resolved_subject_refs = subject_refs or (_sample_subject_ref(tenant_id=tenant_id),)
    resolved_key = reconciliation_key or _sample_reconciliation_key(tenant_id=tenant_id)
    occurrences = tuple(
        ProblemOccurrence(
            subject_ref=subject_ref,
            observed_at=observed_at,
            strategy_id=STRATEGY_ID,
            strategy_version=STRATEGY_VERSION,
            method=ProblemGroupingMethod.DETERMINISTIC,
        )
        for subject_ref in resolved_subject_refs
    )
    resolved_count = occurrence_count if occurrence_count is not None else len(occurrences)
    observed_times = [occurrence.observed_at for occurrence in occurrences]
    return Problem(
        problem_id=problem_id or mint_problem_id(),
        tenant_id=tenant_id,
        status=status,
        first_seen_at=min(observed_times) if observed_times else observed_at,
        last_seen_at=max(observed_times) if observed_times else observed_at,
        occurrence_count=resolved_count,
        current_subject_refs=resolved_subject_refs,
        occurrences=occurrences,
        provenance=ProblemLifecycleProvenance(
            strategy_id=STRATEGY_ID,
            strategy_version=STRATEGY_VERSION,
            method=ProblemGroupingMethod.DETERMINISTIC,
            reconciliation_key=resolved_key,
        ),
        record_version=record_version,
    )


def assert_problem_persistence_conformance(
    store: ProblemPersistence,
    *,
    label: str,
) -> None:
    """Shared behavioral contract for every ``ProblemPersistence`` backend."""
    tenant_a = f"{label}-tenant-a"
    tenant_b = f"{label}-tenant-b"

    subject_a1 = _sample_subject_ref(tenant_id=tenant_a)
    subject_a2 = _sample_subject_ref(tenant_id=tenant_a)
    subject_b = _sample_subject_ref(tenant_id=tenant_b)
    reconciliation_a = _sample_reconciliation_key(tenant_id=tenant_a)

    first = sample_problem(
        tenant_id=tenant_a,
        subject_refs=(subject_a1, subject_a2),
        reconciliation_key=reconciliation_a,
    )
    duplicate = sample_problem(
        problem_id=first.problem_id,
        tenant_id=tenant_a,
        subject_refs=(subject_a1, subject_a2),
        reconciliation_key=reconciliation_a,
    )
    foreign = sample_problem(
        tenant_id=tenant_b,
        subject_refs=(subject_b,),
        reconciliation_key=_sample_reconciliation_key(tenant_id=tenant_b),
    )

    created = store.create(first)
    assert created == first
    assert store.create(duplicate) == first
    store.create(foreign)

    assert store.get(tenant_id=tenant_a, problem_id=first.problem_id) == first
    assert store.get(tenant_id=tenant_b, problem_id=foreign.problem_id) == foreign
    assert store.get(tenant_id=tenant_b, problem_id=first.problem_id) is None

    listed_a = store.list_for_tenant(tenant_a)
    listed_b = store.list_for_tenant(tenant_b)
    assert [item.problem_id for item in listed_a] == sorted(
        [first.problem_id],
        key=str,
    )
    assert [item.problem_id for item in listed_b] == [foreign.problem_id]

    assert (
        store.find_by_reconciliation_key(
            tenant_id=tenant_a,
            reconciliation_key=reconciliation_a,
        )
        == first
    )
    assert (
        store.find_by_reconciliation_key(
            tenant_id=tenant_b,
            reconciliation_key=reconciliation_a,
        )
        is None
    )
    assert store.find_by_subject_ref(tenant_id=tenant_a, subject_ref=subject_a1) == first
    assert store.get(tenant_id=tenant_b, problem_id=first.problem_id) is None

    updated = Problem(
        problem_id=first.problem_id,
        tenant_id=first.tenant_id,
        status=first.status,
        first_seen_at=first.first_seen_at,
        last_seen_at=_OBSERVED_AT_LATER,
        occurrence_count=first.occurrence_count + 1,
        current_subject_refs=first.current_subject_refs,
        occurrences=first.occurrences,
        provenance=first.provenance,
        record_version=2,
    )
    assert store.update(updated, expected_version=1) == updated
    assert store.get(tenant_id=tenant_a, problem_id=first.problem_id) == updated

    with pytest_raises_conflict():
        store.update(updated, expected_version=1)

    missing = sample_problem(tenant_id=tenant_a)
    with pytest_raises_conflict():
        store.update(missing, expected_version=1)

    conflicting = sample_problem(
        problem_id=first.problem_id,
        tenant_id=tenant_a,
        subject_refs=(subject_a1,),
        reconciliation_key=reconciliation_a,
        observed_at=_OBSERVED_AT_LATER,
    )
    with pytest_raises_conflict():
        store.create(conflicting)

    other_signature = DeterministicProblemSignature(findings=(), limitations=())
    other_reconciliation = _sample_reconciliation_key(
        tenant_id=tenant_a,
        signature=other_signature,
    )
    collision_subject = _sample_subject_ref(tenant_id=tenant_a)
    first_collision = sample_problem(
        tenant_id=tenant_a,
        subject_refs=(collision_subject,),
        reconciliation_key=other_reconciliation,
    )
    store.create(first_collision)
    third_signature = DeterministicProblemSignature(
        findings=(
            DeterministicFindingSignature(
                kind=DiagnosticFindingKind.DISALLOWED_AFTER_FAILED,
                scope=LifecycleAnomalyScope.ATTEMPT,
                source_anomaly_kind=LifecycleAnomalyKind.EVENT_AFTER_TERMINAL,
            ),
        ),
        limitations=(),
    )
    second_collision = sample_problem(
        tenant_id=tenant_a,
        subject_refs=(collision_subject,),
        reconciliation_key=_sample_reconciliation_key(
            tenant_id=tenant_a,
            signature=third_signature,
        ),
    )
    with pytest_raises_conflict():
        store.create(second_collision)

    reconciliation_collision = _sample_reconciliation_key(
        tenant_id=f"{label}-collision-tenant",
    )
    winner = sample_problem(
        tenant_id=f"{label}-collision-tenant",
        reconciliation_key=reconciliation_collision,
    )
    store.create(winner)
    loser = sample_problem(
        tenant_id=f"{label}-collision-tenant",
        reconciliation_key=reconciliation_collision,
    )
    with pytest_raises_conflict():
        store.create(loser)
    assert store.list_for_tenant(f"{label}-collision-tenant") == (winner,)

    assert_failed_create_leaves_no_orphan_indexes(store, label=label)


def assert_failed_create_leaves_no_orphan_indexes(
    store: ProblemPersistence,
    *,
    label: str,
) -> None:
    """Failed logical create must not leave orphan indexes or poisoned keys."""
    tenant_id = f"{label}-orphan-free"
    shared = _sample_subject_ref(tenant_id=tenant_id)
    winner_key = _sample_reconciliation_key(tenant_id=tenant_id)
    loser_key = _sample_reconciliation_key(
        tenant_id=tenant_id,
        signature=DeterministicProblemSignature(findings=(), limitations=()),
    )
    winner = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(shared,),
        reconciliation_key=winner_key,
    )
    loser = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(shared,),
        reconciliation_key=loser_key,
    )
    store.create(winner)
    with pytest_raises_conflict():
        store.create(loser)

    assert store.list_for_tenant(tenant_id) == (winner,)
    assert store.get(tenant_id=tenant_id, problem_id=loser.problem_id) is None
    assert (
        store.find_by_reconciliation_key(
            tenant_id=tenant_id,
            reconciliation_key=loser.provenance.reconciliation_key,
        )
        is None
    )
    assert store.find_by_subject_ref(tenant_id=tenant_id, subject_ref=shared) == winner

    reuse_subject = _sample_subject_ref(tenant_id=tenant_id)
    reused = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(reuse_subject,),
        reconciliation_key=loser.provenance.reconciliation_key,
    )
    assert store.create(reused) == reused

    partial_tenant = f"{label}-partial-claim"
    owned_subject = _sample_subject_ref(tenant_id=partial_tenant)
    free_subject = _sample_subject_ref(tenant_id=partial_tenant)
    partial_key_a = _sample_reconciliation_key(tenant_id=partial_tenant)
    partial_key_b = _sample_reconciliation_key(
        tenant_id=partial_tenant,
        signature=DeterministicProblemSignature(findings=(), limitations=()),
    )
    problem_a = sample_problem(
        tenant_id=partial_tenant,
        subject_refs=(owned_subject,),
        reconciliation_key=partial_key_a,
    )
    store.create(problem_a)
    problem_b = sample_problem(
        tenant_id=partial_tenant,
        subject_refs=(free_subject, owned_subject),
        reconciliation_key=partial_key_b,
    )
    with pytest_raises_conflict():
        store.create(problem_b)

    assert (
        store.find_by_reconciliation_key(
            tenant_id=partial_tenant,
            reconciliation_key=problem_b.provenance.reconciliation_key,
        )
        is None
    )
    assert (
        store.find_by_subject_ref(
            tenant_id=partial_tenant,
            subject_ref=free_subject,
        )
        is None
    )
    assert (
        store.find_by_subject_ref(
            tenant_id=partial_tenant,
            subject_ref=owned_subject,
        )
        == problem_a
    )
    assert store.get(tenant_id=partial_tenant, problem_id=problem_b.problem_id) is None


def assert_problem_update_publishes_subject_indexes_atomically(
    store: ProblemPersistence,
    *,
    label: str,
) -> None:
    """Update must publish new subject indexes before canonical CAS becomes durable."""
    tenant_id = f"{label}-update-publish"
    subject_a = _sample_subject_ref(tenant_id=tenant_id)
    subject_b = _sample_subject_ref(tenant_id=tenant_id)
    reconciliation_key = _sample_reconciliation_key(tenant_id=tenant_id)
    created = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(subject_a,),
        reconciliation_key=reconciliation_key,
    )
    store.create(created)

    updated = Problem(
        problem_id=created.problem_id,
        tenant_id=created.tenant_id,
        status=created.status,
        first_seen_at=created.first_seen_at,
        last_seen_at=_OBSERVED_AT_LATER,
        occurrence_count=created.occurrence_count + 1,
        current_subject_refs=(subject_a, subject_b),
        occurrences=created.occurrences,
        provenance=created.provenance,
        record_version=2,
    )
    assert store.update(updated, expected_version=1) == updated
    assert store.get(tenant_id=tenant_id, problem_id=created.problem_id) == updated
    assert store.find_by_subject_ref(tenant_id=tenant_id, subject_ref=subject_b) == updated

    collision_tenant = f"{label}-update-collision"
    owned_subject = _sample_subject_ref(tenant_id=collision_tenant)
    free_subject = _sample_subject_ref(tenant_id=collision_tenant)
    problem_q = sample_problem(
        tenant_id=collision_tenant,
        subject_refs=(owned_subject,),
        reconciliation_key=_sample_reconciliation_key(tenant_id=collision_tenant),
    )
    store.create(problem_q)
    problem_p = sample_problem(
        tenant_id=collision_tenant,
        subject_refs=(_sample_subject_ref(tenant_id=collision_tenant),),
        reconciliation_key=_sample_reconciliation_key(
            tenant_id=collision_tenant,
            signature=DeterministicProblemSignature(findings=(), limitations=()),
        ),
    )
    store.create(problem_p)
    collision_update = Problem(
        problem_id=problem_p.problem_id,
        tenant_id=problem_p.tenant_id,
        status=problem_p.status,
        first_seen_at=problem_p.first_seen_at,
        last_seen_at=_OBSERVED_AT_LATER,
        occurrence_count=problem_p.occurrence_count + 1,
        current_subject_refs=problem_p.current_subject_refs + (owned_subject,),
        occurrences=problem_p.occurrences,
        provenance=problem_p.provenance,
        record_version=2,
    )
    with pytest_raises_conflict():
        store.update(collision_update, expected_version=1)
    assert store.get(tenant_id=collision_tenant, problem_id=problem_p.problem_id) == problem_p
    assert (
        store.find_by_subject_ref(
            tenant_id=collision_tenant,
            subject_ref=owned_subject,
        )
        == problem_q
    )

    partial_tenant = f"{label}-update-partial"
    owned = _sample_subject_ref(tenant_id=partial_tenant)
    free = _sample_subject_ref(tenant_id=partial_tenant)
    only_a = _sample_subject_ref(tenant_id=partial_tenant)
    partial_q = sample_problem(
        tenant_id=partial_tenant,
        subject_refs=(owned,),
        reconciliation_key=_sample_reconciliation_key(tenant_id=partial_tenant),
    )
    store.create(partial_q)
    partial_p = sample_problem(
        tenant_id=partial_tenant,
        subject_refs=(only_a,),
        reconciliation_key=_sample_reconciliation_key(
            tenant_id=partial_tenant,
            signature=DeterministicProblemSignature(findings=(), limitations=()),
        ),
    )
    store.create(partial_p)
    partial_update = Problem(
        problem_id=partial_p.problem_id,
        tenant_id=partial_p.tenant_id,
        status=partial_p.status,
        first_seen_at=partial_p.first_seen_at,
        last_seen_at=_OBSERVED_AT_LATER,
        occurrence_count=partial_p.occurrence_count + 2,
        current_subject_refs=(only_a, free, owned),
        occurrences=partial_p.occurrences,
        provenance=partial_p.provenance,
        record_version=2,
    )
    with pytest_raises_conflict():
        store.update(partial_update, expected_version=1)
    assert (
        store.find_by_subject_ref(
            tenant_id=partial_tenant,
            subject_ref=free,
        )
        is None
    )
    assert (
        store.find_by_subject_ref(
            tenant_id=partial_tenant,
            subject_ref=owned,
        )
        == partial_q
    )
    assert store.get(tenant_id=partial_tenant, problem_id=partial_p.problem_id) == partial_p

    resolve_tenant = f"{label}-update-resolve"
    resolve_subject = _sample_subject_ref(tenant_id=resolve_tenant)
    resolve_problem = sample_problem(
        tenant_id=resolve_tenant,
        subject_refs=(resolve_subject,),
    )
    store.create(resolve_problem)
    resolved = Problem(
        problem_id=resolve_problem.problem_id,
        tenant_id=resolve_problem.tenant_id,
        status=ProblemStatus.RESOLVED,
        first_seen_at=resolve_problem.first_seen_at,
        last_seen_at=resolve_problem.last_seen_at,
        occurrence_count=resolve_problem.occurrence_count,
        current_subject_refs=resolve_problem.current_subject_refs,
        occurrences=resolve_problem.occurrences,
        provenance=resolve_problem.provenance,
        record_version=2,
    )
    assert store.update(resolved, expected_version=1) == resolved
    assert (
        store.find_by_subject_ref(
            tenant_id=resolve_tenant,
            subject_ref=resolve_subject,
        )
        == resolved
    )


def assert_problem_persistence_typed_round_trip(
    store: ProblemPersistence,
    *,
    label: str,
) -> None:
    tenant_id = f"{label}-round-trip"
    subject_a = _sample_subject_ref(tenant_id=tenant_id)
    subject_b = _sample_subject_ref(tenant_id=tenant_id)
    observed_later = _OBSERVED_AT + timedelta(minutes=30)
    record = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(subject_a, subject_b),
        observed_at=_OBSERVED_AT,
    )
    record = Problem(
        problem_id=record.problem_id,
        tenant_id=record.tenant_id,
        status=record.status,
        first_seen_at=_OBSERVED_AT,
        last_seen_at=observed_later,
        occurrence_count=2,
        current_subject_refs=record.current_subject_refs,
        occurrences=(
            ProblemOccurrence(
                subject_ref=subject_a,
                observed_at=_OBSERVED_AT,
                strategy_id=STRATEGY_ID,
                strategy_version=STRATEGY_VERSION,
                method=ProblemGroupingMethod.DETERMINISTIC,
            ),
            ProblemOccurrence(
                subject_ref=subject_b,
                observed_at=observed_later,
                strategy_id=STRATEGY_ID,
                strategy_version=STRATEGY_VERSION,
                method=ProblemGroupingMethod.DETERMINISTIC,
            ),
        ),
        provenance=record.provenance,
        record_version=record.record_version,
    )
    store.create(record)
    loaded = store.get(tenant_id=tenant_id, problem_id=record.problem_id)
    assert loaded == record
    assert loaded is not None
    assert loaded.first_seen_at.tzinfo is not None
    assert loaded.last_seen_at.tzinfo is not None
    assert len(loaded.occurrences) == 2


class _ConflictContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not ProblemPersistenceConflictError:
            raise AssertionError("expected ProblemPersistenceConflictError") from exc
        return True


def pytest_raises_conflict() -> _ConflictContext:
    return _ConflictContext()


class _IntegrityContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not ProblemPersistenceIntegrityError:
            raise AssertionError("expected ProblemPersistenceIntegrityError") from exc
        return True


def pytest_raises_integrity() -> _IntegrityContext:
    return _IntegrityContext()
