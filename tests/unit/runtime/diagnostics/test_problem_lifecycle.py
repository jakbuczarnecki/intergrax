# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    STRATEGY_VERSION,
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessmentBuilder
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.in_memory_problem_persistence import (
    InMemoryProblemPersistence,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingAssessmentInput,
    ProblemGroupingCandidate,
    ProblemGroupingEngine,
    ProblemGroupingMethod,
    ProblemGroupingProvenance,
    ProblemGroupingStrategyRegistry,
    ProblemGroupingStrategyVersion,
    ProblemGroupingSubjectRef,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemLifecycleEngine,
    ProblemLifecycleIntegrityError,
    ProblemStatus,
    mint_problem_id,
    validate_problem_id,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_OBSERVED_AT = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)
_OBSERVED_AT_LATER = _OBSERVED_AT + timedelta(hours=1)
_OBSERVED_AT_EARLIER = _OBSERVED_AT - timedelta(hours=1)
_RESOLVED_AT = _OBSERVED_AT_LATER + timedelta(hours=2)


def assert_occurrence_timestamps_match(problem: Problem) -> None:
    observed_times = [occurrence.observed_at for occurrence in problem.occurrences]
    assert problem.first_seen_at == min(observed_times)
    assert problem.last_seen_at == max(observed_times)


def _engine() -> ProblemGroupingEngine:
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    return ProblemGroupingEngine(registry)


def _lifecycle_engine() -> ProblemLifecycleEngine:
    return ProblemLifecycleEngine(InMemoryProblemPersistence())


def _assess_attempt_sequence(
    event_types: list[RuntimeEventType],
    *,
    tenant_id: str = _TENANT_A,
) -> ProblemGroupingAssessmentInput:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    for event_type in event_types:
        event = sample_runtime_event(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        ).model_copy(update={"event_type": event_type})
        runtime_store.append(event, tenant_id=tenant_id)

    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(tenant_id, task_id, run_id)
    lifecycle = LifecycleAnomalyAnalyzer().analyze(reconstruction)
    assessment = DiagnosticAssessmentBuilder().assess(reconstruction, lifecycle)
    return ProblemGroupingAssessmentInput(assessment=assessment)


def _retry_after_completed_sequence(
    *,
    violating_event_type: RuntimeEventType = RuntimeEventType.RETRY_SCHEDULED,
) -> list[RuntimeEventType]:
    return [
        RuntimeEventType.TASK_CREATED,
        RuntimeEventType.TASK_COMPLETED,
        violating_event_type,
    ]


def _assess_retry_pair(
    *,
    tenant_id: str = _TENANT_A,
    violating_event_type: RuntimeEventType = RuntimeEventType.RETRY_SCHEDULED,
) -> tuple[ProblemGroupingAssessmentInput, ProblemGroupingAssessmentInput]:
    sequence = _retry_after_completed_sequence(
        violating_event_type=violating_event_type,
    )
    return (
        _assess_attempt_sequence(sequence, tenant_id=tenant_id),
        _assess_attempt_sequence(sequence, tenant_id=tenant_id),
    )


def _group_pair(
    *,
    tenant_id: str = _TENANT_A,
):
    first, second = _assess_retry_pair(tenant_id=tenant_id)
    grouping_result = _engine().group(
        (first, second),
        strategy_id=STRATEGY_ID,
    )
    return grouping_result, first, second


def test_first_candidate_creates_one_problem_id() -> None:
    grouping_result, _, _ = _group_pair()
    lifecycle = _lifecycle_engine()

    result = lifecycle.reconcile(grouping_result, observed_at=_OBSERVED_AT)

    assert len(result.created) == 1
    assert result.updated == ()
    assert result.unchanged == ()
    problem = result.created[0]
    validate_problem_id(problem.problem_id)
    assert problem.status is ProblemStatus.OPEN
    assert problem.occurrence_count == 2
    assert_occurrence_timestamps_match(problem)


def test_same_candidate_processed_twice_is_idempotent() -> None:
    grouping_result, _, _ = _group_pair()
    lifecycle = _lifecycle_engine()

    first = lifecycle.reconcile(grouping_result, observed_at=_OBSERVED_AT)
    second = lifecycle.reconcile(grouping_result, observed_at=_OBSERVED_AT_LATER)

    assert len(first.created) == 1
    assert second.created == ()
    assert second.updated == ()
    assert len(second.unchanged) == 1
    assert first.created[0].problem_id == second.unchanged[0].problem_id
    assert second.unchanged[0].occurrence_count == 2
    assert_occurrence_timestamps_match(second.unchanged[0])


def test_later_candidate_with_new_subject_increments_count() -> None:
    first_input, second_input = _assess_retry_pair()
    third_input, _ = _assess_retry_pair()
    lifecycle = _lifecycle_engine()

    pair_grouping = _engine().group(
        (first_input, second_input),
        strategy_id=STRATEGY_ID,
    )
    first_result = lifecycle.reconcile(pair_grouping, observed_at=_OBSERVED_AT)
    problem_id = first_result.created[0].problem_id

    extended_grouping = _engine().group(
        (first_input, second_input, third_input),
        strategy_id=STRATEGY_ID,
    )
    second_result = lifecycle.reconcile(
        extended_grouping,
        observed_at=_OBSERVED_AT_LATER,
    )

    assert second_result.created == ()
    assert len(second_result.updated) == 1
    updated = second_result.updated[0]
    assert updated.problem_id == problem_id
    assert updated.occurrence_count == 3
    assert_occurrence_timestamps_match(updated)


def test_same_recurrence_key_in_another_tenant_is_isolated() -> None:
    grouping_a, _, _ = _group_pair(tenant_id=_TENANT_A)
    grouping_b, _, _ = _group_pair(tenant_id=_TENANT_B)
    persistence = InMemoryProblemPersistence()
    lifecycle = ProblemLifecycleEngine(persistence)

    result_a = lifecycle.reconcile(grouping_a, observed_at=_OBSERVED_AT)
    result_b = lifecycle.reconcile(grouping_b, observed_at=_OBSERVED_AT)

    assert result_a.created[0].problem_id != result_b.created[0].problem_id
    assert result_a.created[0].tenant_id == _TENANT_A
    assert result_b.created[0].tenant_id == _TENANT_B


def test_different_deterministic_signature_creates_different_problem() -> None:
    lifecycle = _lifecycle_engine()

    retry_grouping = _engine().group(_assess_retry_pair(), strategy_id=STRATEGY_ID)
    first_result = lifecycle.reconcile(retry_grouping, observed_at=_OBSERVED_AT)

    failed_grouping = _engine().group(
        _assess_retry_pair(violating_event_type=RuntimeEventType.TASK_FAILED),
        strategy_id=STRATEGY_ID,
    )
    second_result = lifecycle.reconcile(failed_grouping, observed_at=_OBSERVED_AT_LATER)

    assert len(first_result.created) == 1
    assert len(second_result.created) == 1
    assert first_result.created[0].problem_id != second_result.created[0].problem_id


def test_different_strategy_version_does_not_match_old_problem() -> None:
    grouping_result, _, _ = _group_pair()
    lifecycle = _lifecycle_engine()
    initial = lifecycle.reconcile(grouping_result, observed_at=_OBSERVED_AT)
    original_id = initial.created[0].problem_id

    second_grouping = _engine().group(_assess_retry_pair(), strategy_id=STRATEGY_ID)
    candidate = second_grouping.candidates[0]
    bumped = ProblemGroupingCandidate(
        members=candidate.members,
        provenance=ProblemGroupingProvenance(
            strategy_id=candidate.provenance.strategy_id,
            strategy_version=ProblemGroupingStrategyVersion("2"),
            method=ProblemGroupingMethod.DETERMINISTIC,
            supporting_subject_refs=candidate.provenance.supporting_subject_refs,
            basis=candidate.provenance.basis,
        ),
    )
    bumped_result = grouping_result.__class__(
        tenant_id=grouping_result.tenant_id,
        strategy_id=grouping_result.strategy_id,
        strategy_version=ProblemGroupingStrategyVersion("2"),
        method=grouping_result.method,
        candidates=(bumped,),
        ungrouped_subjects=grouping_result.ungrouped_subjects,
    )

    second = lifecycle.reconcile(bumped_result, observed_at=_OBSERVED_AT_LATER)
    assert len(second.created) == 1
    assert second.created[0].problem_id != original_id


def test_first_seen_preserved_and_last_seen_advances() -> None:
    first_input, second_input = _assess_retry_pair()
    third_input, _ = _assess_retry_pair()
    lifecycle = _lifecycle_engine()

    pair_grouping = _engine().group(
        (first_input, second_input),
        strategy_id=STRATEGY_ID,
    )
    first = lifecycle.reconcile(pair_grouping, observed_at=_OBSERVED_AT)
    problem = first.created[0]
    assert problem.first_seen_at == _OBSERVED_AT
    assert problem.last_seen_at == _OBSERVED_AT

    extended = _engine().group(
        (first_input, second_input, third_input),
        strategy_id=STRATEGY_ID,
    )
    second = lifecycle.reconcile(extended, observed_at=_OBSERVED_AT_LATER)
    updated = second.updated[0]

    assert updated.first_seen_at == _OBSERVED_AT
    assert updated.last_seen_at == _OBSERVED_AT_LATER
    assert_occurrence_timestamps_match(updated)


def test_out_of_order_new_subject_lowers_first_seen_at() -> None:
    """Create at 10:00, later new subject at 09:00 — first_seen retreats."""
    first_input, second_input = _assess_retry_pair()
    third_input, _ = _assess_retry_pair()
    lifecycle = _lifecycle_engine()

    pair_grouping = _engine().group(
        (first_input, second_input),
        strategy_id=STRATEGY_ID,
    )
    created = lifecycle.reconcile(pair_grouping, observed_at=_OBSERVED_AT_LATER)
    assert created.created[0].first_seen_at == _OBSERVED_AT_LATER
    assert created.created[0].last_seen_at == _OBSERVED_AT_LATER

    extended = _engine().group(
        (first_input, second_input, third_input),
        strategy_id=STRATEGY_ID,
    )
    updated = lifecycle.reconcile(extended, observed_at=_OBSERVED_AT).updated[0]

    assert updated.first_seen_at == _OBSERVED_AT
    assert updated.last_seen_at == _OBSERVED_AT_LATER
    assert_occurrence_timestamps_match(updated)


def test_replay_same_subject_at_earlier_observed_at_is_idempotent() -> None:
    grouping_result, _, _ = _group_pair()
    lifecycle = _lifecycle_engine()

    first = lifecycle.reconcile(grouping_result, observed_at=_OBSERVED_AT)
    problem = first.created[0]
    second = lifecycle.reconcile(grouping_result, observed_at=_OBSERVED_AT_EARLIER)

    assert second.unchanged[0].first_seen_at == problem.first_seen_at
    assert second.unchanged[0].last_seen_at == problem.last_seen_at
    assert second.unchanged[0].occurrence_count == problem.occurrence_count
    assert_occurrence_timestamps_match(second.unchanged[0])


def test_replay_same_subject_at_later_observed_at_is_idempotent() -> None:
    grouping_result, _, _ = _group_pair()
    lifecycle = _lifecycle_engine()

    first = lifecycle.reconcile(grouping_result, observed_at=_OBSERVED_AT)
    problem = first.created[0]
    second = lifecycle.reconcile(grouping_result, observed_at=_OBSERVED_AT_LATER)

    assert second.unchanged[0].first_seen_at == problem.first_seen_at
    assert second.unchanged[0].last_seen_at == problem.last_seen_at
    assert second.unchanged[0].occurrence_count == problem.occurrence_count
    assert_occurrence_timestamps_match(second.unchanged[0])


def test_resolve_does_not_advance_last_seen_at() -> None:
    grouping_result, _, _ = _group_pair()
    lifecycle = _lifecycle_engine()

    created = lifecycle.reconcile(grouping_result, observed_at=_OBSERVED_AT).created[0]
    resolved = lifecycle.resolve(
        tenant_id=_TENANT_A,
        problem_id=created.problem_id,
        resolved_at=_RESOLVED_AT,
    )

    assert resolved.status is ProblemStatus.RESOLVED
    assert resolved.first_seen_at == _OBSERVED_AT
    assert resolved.last_seen_at == _OBSERVED_AT
    assert_occurrence_timestamps_match(resolved)


def test_duplicate_subject_does_not_increment_count() -> None:
    grouping_result, _, _ = _group_pair()
    lifecycle = _lifecycle_engine()

    first = lifecycle.reconcile(grouping_result, observed_at=_OBSERVED_AT)
    duplicate_members = ProblemGroupingCandidate(
        members=grouping_result.candidates[0].members,
        provenance=grouping_result.candidates[0].provenance,
    )
    duplicate_result = grouping_result.__class__(
        tenant_id=grouping_result.tenant_id,
        strategy_id=grouping_result.strategy_id,
        strategy_version=grouping_result.strategy_version,
        method=grouping_result.method,
        candidates=(duplicate_members,),
        ungrouped_subjects=(),
    )
    second = lifecycle.reconcile(duplicate_result, observed_at=_OBSERVED_AT_LATER)

    assert second.unchanged[0].occurrence_count == first.created[0].occurrence_count
    assert_occurrence_timestamps_match(second.unchanged[0])


def test_explicit_resolve_and_recurrence_reopens() -> None:
    first_input, second_input = _assess_retry_pair()
    third_input, _ = _assess_retry_pair()
    persistence = InMemoryProblemPersistence()
    lifecycle = ProblemLifecycleEngine(persistence)

    pair_grouping = _engine().group(
        (first_input, second_input),
        strategy_id=STRATEGY_ID,
    )
    created = lifecycle.reconcile(pair_grouping, observed_at=_OBSERVED_AT).created[0]
    resolved = lifecycle.resolve(
        tenant_id=_TENANT_A,
        problem_id=created.problem_id,
        resolved_at=_OBSERVED_AT_LATER,
    )
    assert resolved.status is ProblemStatus.RESOLVED

    extended = _engine().group(
        (first_input, second_input, third_input),
        strategy_id=STRATEGY_ID,
    )
    reopened = lifecycle.reconcile(extended, observed_at=_OBSERVED_AT_LATER).updated[0]
    assert reopened.status is ProblemStatus.OPEN
    assert reopened.first_seen_at == _OBSERVED_AT
    assert reopened.last_seen_at == _OBSERVED_AT_LATER
    assert reopened.occurrence_count == 3
    assert_occurrence_timestamps_match(reopened)


def test_one_occurrence_cannot_attach_to_two_existing_problems() -> None:
    retry_first, retry_second = _assess_retry_pair()
    failed_first, failed_second = _assess_retry_pair(
        violating_event_type=RuntimeEventType.TASK_FAILED,
    )
    grouping_ab = _engine().group((retry_first, retry_second), strategy_id=STRATEGY_ID)
    grouping_cd = _engine().group((failed_first, failed_second), strategy_id=STRATEGY_ID)

    persistence = InMemoryProblemPersistence()
    lifecycle = ProblemLifecycleEngine(persistence)
    lifecycle.reconcile(grouping_ab, observed_at=_OBSERVED_AT)
    lifecycle.reconcile(grouping_cd, observed_at=_OBSERVED_AT)

    shared_member = grouping_ab.candidates[0].members[0]
    other_member = grouping_cd.candidates[0].members[0]
    overlapping_candidate = ProblemGroupingCandidate(
        members=(shared_member, other_member),
        provenance=grouping_ab.candidates[0].provenance,
    )
    overlapping_result = grouping_ab.__class__(
        tenant_id=grouping_ab.tenant_id,
        strategy_id=grouping_ab.strategy_id,
        strategy_version=grouping_ab.strategy_version,
        method=grouping_ab.method,
        candidates=(overlapping_candidate,),
        ungrouped_subjects=(),
    )

    with pytest.raises(ProblemLifecycleIntegrityError):
        lifecycle.reconcile(overlapping_result, observed_at=_OBSERVED_AT_LATER)


def test_problem_contract_has_no_root_cause_fields() -> None:
    forbidden = {"root_cause", "root_cause_confidence", "cause"}
    field_names = {field.name for field in fields(Problem)}
    assert forbidden.isdisjoint(field_names)


def test_mint_problem_id_format() -> None:
    problem_id = mint_problem_id()
    assert str(problem_id).startswith("problem_")
    validate_problem_id(problem_id)


def test_overlapping_candidates_in_same_invocation_fail_closed() -> None:
    retry_a, retry_b = _assess_retry_pair()
    failed_pair = _assess_retry_pair(
        violating_event_type=RuntimeEventType.TASK_FAILED,
    )
    grouping_ab = _engine().group((retry_a, retry_b), strategy_id=STRATEGY_ID)
    failed_grouping = _engine().group(failed_pair, strategy_id=STRATEGY_ID)
    failed_candidate = failed_grouping.candidates[0]
    ref_a = grouping_ab.candidates[0].members[0]
    ref_c = failed_candidate.members[1]
    overlapping_failed = ProblemGroupingCandidate(
        members=(ref_a, ref_c),
        provenance=ProblemGroupingProvenance(
            strategy_id=failed_candidate.provenance.strategy_id,
            strategy_version=failed_candidate.provenance.strategy_version,
            method=failed_candidate.provenance.method,
            supporting_subject_refs=(ref_a, ref_c),
            basis=failed_candidate.provenance.basis,
        ),
    )

    overlapping_result = grouping_ab.__class__(
        tenant_id=grouping_ab.tenant_id,
        strategy_id=grouping_ab.strategy_id,
        strategy_version=grouping_ab.strategy_version,
        method=grouping_ab.method,
        candidates=(
            grouping_ab.candidates[0],
            overlapping_failed,
        ),
        ungrouped_subjects=(),
    )

    with pytest.raises(ProblemLifecycleIntegrityError):
        _lifecycle_engine().reconcile(overlapping_result, observed_at=_OBSERVED_AT)
