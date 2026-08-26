# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessmentBuilder
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticOccurrenceReadStatus,
    DiagnosticReadIntegrityError,
    DiagnosticReadUnavailableReason,
)
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.in_memory_problem_persistence import (
    InMemoryProblemPersistence,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingAssessmentInput,
    ProblemGroupingEngine,
    ProblemGroupingMethod,
    ProblemGroupingStrategyRegistry,
    problem_grouping_subject_ref_for_execution,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemLifecycleEngine,
    ProblemOccurrence,
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

_FORBIDDEN_FIELD_FRAGMENTS = frozenset(
    {
        "payload",
        "raw_log",
        "traceback",
        "prompt",
        "document_content",
        "root_cause",
        "confidence",
    }
)


def _grouping_engine() -> ProblemGroupingEngine:
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    return ProblemGroupingEngine(registry)


def _assess_attempt_sequence(
    event_types: list[RuntimeEventType],
    *,
    tenant_id: str = _TENANT_A,
    runtime_store: InMemoryRuntimeEventStore | None = None,
) -> ProblemGroupingAssessmentInput:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = runtime_store or InMemoryRuntimeEventStore()
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


def _assess_retry_pair(
    *,
    tenant_id: str = _TENANT_A,
    violating_event_type: RuntimeEventType = RuntimeEventType.RETRY_SCHEDULED,
    runtime_store: InMemoryRuntimeEventStore | None = None,
) -> tuple[ProblemGroupingAssessmentInput, ProblemGroupingAssessmentInput]:
    sequence = _retry_after_completed_sequence(violating_event_type)
    return (
        _assess_attempt_sequence(sequence, tenant_id=tenant_id, runtime_store=runtime_store),
        _assess_attempt_sequence(sequence, tenant_id=tenant_id, runtime_store=runtime_store),
    )


def _retry_after_completed_sequence(
    violating_event_type: RuntimeEventType = RuntimeEventType.RETRY_SCHEDULED,
) -> list[RuntimeEventType]:
    return [
        RuntimeEventType.TASK_CREATED,
        RuntimeEventType.TASK_COMPLETED,
        violating_event_type,
    ]


def _persist_problem(
    *,
    tenant_id: str = _TENANT_A,
    observed_at: datetime = _OBSERVED_AT,
    persistence: InMemoryProblemPersistence | None = None,
    runtime_store: InMemoryRuntimeEventStore | None = None,
) -> tuple[Problem, InMemoryProblemPersistence, InMemoryRuntimeEventStore]:
    persistence = persistence or InMemoryProblemPersistence()
    runtime_store = runtime_store or InMemoryRuntimeEventStore()
    lifecycle = ProblemLifecycleEngine(persistence)
    grouping = _grouping_engine().group(
        _assess_retry_pair(tenant_id=tenant_id, runtime_store=runtime_store),
        strategy_id=STRATEGY_ID,
    )
    result = lifecycle.reconcile(grouping, observed_at=observed_at)
    problem = result.created[0]
    return problem, persistence, runtime_store


def _read_service(
    persistence: InMemoryProblemPersistence,
    runtime_store: InMemoryRuntimeEventStore | None = None,
) -> DiagnosticReadService:
    return DiagnosticReadService(
        problem_persistence=persistence,
        execution_reconstructor=ExecutionReconstructor(
            runtime_events=runtime_store or InMemoryRuntimeEventStore(),
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )


def _collect_dataclass_field_names(obj: Any) -> set[str]:
    names: set[str] = set()
    if is_dataclass(obj):
        for field in fields(obj):
            names.add(field.name)
            names.update(_collect_dataclass_field_names(field.default))
    if isinstance(obj, tuple):
        for item in obj:
            names.update(_collect_dataclass_field_names(item))
    return names


def test_list_empty_tenant_returns_empty_result() -> None:
    persistence = InMemoryProblemPersistence()
    service = _read_service(persistence)

    result = service.list_problems(tenant_id=_TENANT_A)

    assert result.problems == ()
    assert result.total_count == 0
    assert result.returned_count == 0
    assert result.is_truncated is False


def test_list_returns_only_tenant_scoped_problems() -> None:
    persistence = InMemoryProblemPersistence()
    problem_a, _, _ = _persist_problem(tenant_id=_TENANT_A, persistence=persistence)
    problem_b, _, _ = _persist_problem(tenant_id=_TENANT_B, persistence=persistence)
    service = _read_service(persistence)

    result_a = service.list_problems(tenant_id=_TENANT_A)
    result_b = service.list_problems(tenant_id=_TENANT_B)

    assert {item.problem_id for item in result_a.problems} == {problem_a.problem_id}
    assert {item.problem_id for item in result_b.problems} == {problem_b.problem_id}


def test_list_status_filter() -> None:
    persistence = InMemoryProblemPersistence()
    problem, _, _ = _persist_problem(persistence=persistence)
    lifecycle = ProblemLifecycleEngine(persistence)
    lifecycle.resolve(
        tenant_id=_TENANT_A,
        problem_id=problem.problem_id,
        resolved_at=_OBSERVED_AT_LATER,
    )
    service = _read_service(persistence)

    open_only = service.list_problems(tenant_id=_TENANT_A, status=ProblemStatus.OPEN)
    resolved_only = service.list_problems(tenant_id=_TENANT_A, status=ProblemStatus.RESOLVED)

    assert open_only.problems == ()
    assert len(resolved_only.problems) == 1
    assert resolved_only.problems[0].status is ProblemStatus.RESOLVED


def test_list_deterministic_ordering() -> None:
    persistence = InMemoryProblemPersistence()
    lifecycle = ProblemLifecycleEngine(persistence)
    first_grouping = _grouping_engine().group(_assess_retry_pair(), strategy_id=STRATEGY_ID)
    lifecycle.reconcile(first_grouping, observed_at=_OBSERVED_AT_EARLIER)
    second_grouping = _grouping_engine().group(
        _assess_retry_pair(violating_event_type=RuntimeEventType.TASK_FAILED),
        strategy_id=STRATEGY_ID,
    )
    lifecycle.reconcile(second_grouping, observed_at=_OBSERVED_AT_LATER)
    service = _read_service(persistence)

    result = service.list_problems(tenant_id=_TENANT_A)

    assert result.problems[0].last_seen_at == _OBSERVED_AT_LATER
    assert result.problems[1].last_seen_at == _OBSERVED_AT_EARLIER


def test_list_limit_truncation_explicit() -> None:
    persistence = InMemoryProblemPersistence()
    lifecycle = ProblemLifecycleEngine(persistence)
    for observed_at, violating_event_type in (
        (_OBSERVED_AT_EARLIER, RuntimeEventType.RETRY_SCHEDULED),
        (_OBSERVED_AT_LATER, RuntimeEventType.TASK_FAILED),
    ):
        grouping = _grouping_engine().group(
            _assess_retry_pair(violating_event_type=violating_event_type),
            strategy_id=STRATEGY_ID,
        )
        lifecycle.reconcile(grouping, observed_at=observed_at)
    service = _read_service(persistence)

    result = service.list_problems(tenant_id=_TENANT_A, limit=1)

    assert result.total_count == 2
    assert result.returned_count == 1
    assert result.is_truncated is True


def test_list_does_not_call_execution_reconstructor() -> None:
    persistence = InMemoryProblemPersistence()
    _persist_problem(persistence=persistence)
    reconstructor = MagicMock(spec=ExecutionReconstructor)
    service = DiagnosticReadService(
        problem_persistence=persistence,
        execution_reconstructor=reconstructor,
    )

    service.list_problems(tenant_id=_TENANT_A)

    reconstructor.reconstruct_execution.assert_not_called()


def test_get_problem_returns_stable_fields() -> None:
    persistence = InMemoryProblemPersistence()
    problem, _, runtime_store = _persist_problem(persistence=persistence)
    service = _read_service(persistence, runtime_store)

    detail = service.get_problem(tenant_id=_TENANT_A, problem_id=problem.problem_id)

    assert detail is not None
    validate_problem_id(detail.problem_id)
    assert detail.problem_id == problem.problem_id
    assert detail.status is ProblemStatus.OPEN
    assert detail.first_seen_at == problem.first_seen_at
    assert detail.last_seen_at == problem.last_seen_at
    assert detail.occurrence_count == problem.occurrence_count
    assert detail.record_version == problem.record_version


def test_get_problem_occurrences_ordered_newest_first() -> None:
    persistence = InMemoryProblemPersistence()
    runtime_store = InMemoryRuntimeEventStore()
    lifecycle = ProblemLifecycleEngine(persistence)
    first_input, second_input = _assess_retry_pair(runtime_store=runtime_store)
    third_input, _ = _assess_retry_pair(runtime_store=runtime_store)
    pair_grouping = _grouping_engine().group((first_input, second_input), strategy_id=STRATEGY_ID)
    lifecycle.reconcile(pair_grouping, observed_at=_OBSERVED_AT)
    extended = _grouping_engine().group(
        (first_input, second_input, third_input),
        strategy_id=STRATEGY_ID,
    )
    result = lifecycle.reconcile(extended, observed_at=_OBSERVED_AT_LATER)
    problem = result.updated[0]
    service = _read_service(persistence, runtime_store)

    detail = service.get_problem(tenant_id=_TENANT_A, problem_id=problem.problem_id)

    assert detail is not None
    observed_times = [occurrence.observed_at for occurrence in detail.occurrences]
    assert observed_times[0] == _OBSERVED_AT_LATER
    assert observed_times.count(_OBSERVED_AT) == 2


def test_get_problem_reconstructs_through_diag_pipeline() -> None:
    persistence = InMemoryProblemPersistence()
    problem, _, runtime_store = _persist_problem(persistence=persistence)
    reconstructor = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    )
    service = DiagnosticReadService(
        problem_persistence=persistence,
        execution_reconstructor=reconstructor,
    )

    detail = service.get_problem(tenant_id=_TENANT_A, problem_id=problem.problem_id)

    assert detail is not None
    for occurrence_view in detail.occurrences:
        subject_ref = occurrence_view.subject_ref
        reconstruction = reconstructor.reconstruct_execution(
            _TENANT_A,
            subject_ref.task_id,
            subject_ref.run_id,
        )
        lifecycle = LifecycleAnomalyAnalyzer().analyze(reconstruction)
        expected = DiagnosticAssessmentBuilder().assess(reconstruction, lifecycle)
        assert occurrence_view.read_status is DiagnosticOccurrenceReadStatus.AVAILABLE
        assert occurrence_view.assessment == expected


def test_get_problem_other_tenant_returns_none() -> None:
    persistence = InMemoryProblemPersistence()
    problem, _, runtime_store = _persist_problem(persistence=persistence)
    service = _read_service(persistence, runtime_store)

    assert service.get_problem(tenant_id=_TENANT_B, problem_id=problem.problem_id) is None


def test_get_problem_malformed_occurrence_tenant_mismatch_fails_closed() -> None:
    persistence = InMemoryProblemPersistence()
    problem, _, runtime_store = _persist_problem(persistence=persistence)
    bad_occurrence = ProblemOccurrence(
        subject_ref=problem_grouping_subject_ref_for_execution(
            tenant_id=_TENANT_B,
            task_id=mint_task_id(),
            run_id=mint_run_id(),
        ),
        observed_at=_OBSERVED_AT,
        strategy_id=problem.provenance.strategy_id,
        strategy_version=problem.provenance.strategy_version,
        method=ProblemGroupingMethod.DETERMINISTIC,
    )
    corrupted = Problem(
        problem_id=problem.problem_id,
        tenant_id=problem.tenant_id,
        status=problem.status,
        first_seen_at=problem.first_seen_at,
        last_seen_at=problem.last_seen_at,
        occurrence_count=problem.occurrence_count + 1,
        current_subject_refs=problem.current_subject_refs + (bad_occurrence.subject_ref,),
        occurrences=problem.occurrences + (bad_occurrence,),
        provenance=problem.provenance,
        record_version=problem.record_version,
    )
    persistence._records[(problem.tenant_id, problem.problem_id)] = corrupted
    service = _read_service(persistence, runtime_store)

    with pytest.raises(DiagnosticReadIntegrityError):
        service.get_problem(tenant_id=_TENANT_A, problem_id=problem.problem_id)


def test_get_problem_occurrence_truncation_explicit() -> None:
    persistence = InMemoryProblemPersistence()
    runtime_store = InMemoryRuntimeEventStore()
    lifecycle = ProblemLifecycleEngine(persistence)
    first_input, second_input = _assess_retry_pair(runtime_store=runtime_store)
    third_input, _ = _assess_retry_pair(runtime_store=runtime_store)
    pair_grouping = _grouping_engine().group((first_input, second_input), strategy_id=STRATEGY_ID)
    lifecycle.reconcile(pair_grouping, observed_at=_OBSERVED_AT)
    extended = _grouping_engine().group(
        (first_input, second_input, third_input),
        strategy_id=STRATEGY_ID,
    )
    result = lifecycle.reconcile(extended, observed_at=_OBSERVED_AT_LATER)
    problem = result.updated[0]
    service = _read_service(persistence, runtime_store)

    detail = service.get_problem(
        tenant_id=_TENANT_A,
        problem_id=problem.problem_id,
        occurrence_limit=2,
    )

    assert detail is not None
    assert detail.total_occurrence_count == 3
    assert detail.returned_occurrence_count == 2
    assert detail.is_occurrences_truncated is True


def test_get_problem_unavailable_when_execution_evidence_missing() -> None:
    persistence = InMemoryProblemPersistence()
    problem, _, _ = _persist_problem(persistence=persistence)
    empty_runtime_store = InMemoryRuntimeEventStore()
    service = _read_service(persistence, empty_runtime_store)

    detail = service.get_problem(tenant_id=_TENANT_A, problem_id=problem.problem_id)

    assert detail is not None
    for occurrence_view in detail.occurrences:
        assert occurrence_view.read_status is DiagnosticOccurrenceReadStatus.UNAVAILABLE
        assert occurrence_view.unavailable_reason is (
            DiagnosticReadUnavailableReason.EXECUTION_EVIDENCE_UNAVAILABLE
        )
        assert occurrence_view.assessment is None


def test_read_dtos_exclude_forbidden_fields() -> None:
    persistence = InMemoryProblemPersistence()
    problem, _, runtime_store = _persist_problem(persistence=persistence)
    service = _read_service(persistence, runtime_store)

    list_result = service.list_problems(tenant_id=_TENANT_A)
    detail = service.get_problem(tenant_id=_TENANT_A, problem_id=problem.problem_id)

    forbidden = set()
    for summary in list_result.problems:
        forbidden.update(_collect_dataclass_field_names(summary))
    if detail is not None:
        forbidden.update(_collect_dataclass_field_names(detail))

    assert not forbidden.intersection(_FORBIDDEN_FIELD_FRAGMENTS)


def test_read_service_is_read_only() -> None:
    persistence = MagicMock(spec=InMemoryProblemPersistence)
    persistence.list_for_tenant.return_value = ()
    service = DiagnosticReadService(
        problem_persistence=persistence,
        execution_reconstructor=MagicMock(spec=ExecutionReconstructor),
    )

    service.list_problems(tenant_id=_TENANT_A)

    persistence.create.assert_not_called()
    persistence.update.assert_not_called()


def test_get_problem_does_not_mutate_persistence() -> None:
    persistence = InMemoryProblemPersistence()
    problem, _, runtime_store = _persist_problem(persistence=persistence)
    before = persistence.get(tenant_id=_TENANT_A, problem_id=problem.problem_id)
    service = _read_service(persistence, runtime_store)

    service.get_problem(tenant_id=_TENANT_A, problem_id=problem.problem_id)
    after = persistence.get(tenant_id=_TENANT_A, problem_id=problem.problem_id)

    assert before == after


def _read_service_with_list_records(
    records: tuple[Problem, ...],
) -> DiagnosticReadService:
    persistence = MagicMock(spec=InMemoryProblemPersistence)
    persistence.list_for_tenant.return_value = records
    return DiagnosticReadService(
        problem_persistence=persistence,
        execution_reconstructor=MagicMock(spec=ExecutionReconstructor),
    )


def test_list_matching_tenant_records_unchanged() -> None:
    persistence = InMemoryProblemPersistence()
    problem, _, _ = _persist_problem(persistence=persistence)
    service = _read_service(persistence)

    result = service.list_problems(tenant_id=_TENANT_A)

    assert len(result.problems) == 1
    assert result.problems[0].problem_id == problem.problem_id
    assert result.problems[0].tenant_id == _TENANT_A


def test_list_cross_tenant_record_raises_integrity_error() -> None:
    problem, _, _ = _persist_problem(tenant_id=_TENANT_A)
    cross_tenant = replace(problem, tenant_id=_TENANT_B)
    service = _read_service_with_list_records((cross_tenant,))

    with pytest.raises(
        DiagnosticReadIntegrityError,
        match="tenant_id does not match lookup tenant scope",
    ):
        service.list_problems(tenant_id=_TENANT_A)


def test_list_mixed_tenant_records_raises_without_partial_result() -> None:
    problem_a, _, _ = _persist_problem(tenant_id=_TENANT_A)
    problem_b, _, _ = _persist_problem(tenant_id=_TENANT_B)
    service = _read_service_with_list_records((problem_a, problem_b))

    with pytest.raises(DiagnosticReadIntegrityError):
        service.list_problems(tenant_id=_TENANT_A)


def test_list_cross_tenant_fails_before_summary_conversion() -> None:
    problem, _, _ = _persist_problem(tenant_id=_TENANT_A)
    cross_tenant = replace(problem, tenant_id=_TENANT_B)
    service = _read_service_with_list_records((cross_tenant,))

    with patch(
        "intergrax.runtime.diagnostics.diagnostic_read_service._summary_from_problem",
    ) as summary_mock:
        with pytest.raises(DiagnosticReadIntegrityError):
            service.list_problems(tenant_id=_TENANT_A)

    summary_mock.assert_not_called()


def test_list_status_open_filter_works() -> None:
    persistence = InMemoryProblemPersistence()
    problem, _, _ = _persist_problem(persistence=persistence)
    service = _read_service(persistence)

    result = service.list_problems(tenant_id=_TENANT_A, status=ProblemStatus.OPEN)

    assert len(result.problems) == 1
    assert result.problems[0].problem_id == problem.problem_id
    assert result.problems[0].status is ProblemStatus.OPEN


def test_list_rejects_raw_string_status() -> None:
    persistence = InMemoryProblemPersistence()
    _persist_problem(persistence=persistence)
    service = _read_service(persistence)

    with pytest.raises(TypeError, match="status must be ProblemStatus"):
        service.list_problems(tenant_id=_TENANT_A, status="open")  # type: ignore[arg-type]


def test_diagnostic_read_service_has_no_direct_persistence_imports() -> None:
    import intergrax.runtime.diagnostics.diagnostic_read_service as module

    source = open(module.__file__, encoding="utf-8").read()
    assert "RuntimeEventPersistence" not in source
    assert "CausalEvidencePersistence" not in source
