# © Artur Czarnecki. All rights reserved.

"""DG-003 — deterministic last-good → first-failed operator story."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest

from intergrax.contracts.execution_failure_evidence import ExecutionFailureKind
from intergrax.contracts.execution_event_position import ExecutionEventPosition
from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_event_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_reconstruction_models import (
    ExecutionReconstruction,
    RuntimeHistoryCompleteness,
)
from intergrax.contracts.positioned_runtime_event import PositionedRuntimeEvent
from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    STRATEGY_VERSION,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessmentBuilder,
    DiagnosticFindingKind,
)
from intergrax.runtime.diagnostics.diagnostic_operator_investigation_projection import (
    project_investigation_view,
)
from intergrax.runtime.diagnostics.diagnostic_operator_investigation_read_models import (
    DiagnosticOperatorStoryFirstFailedScope,
    DiagnosticOperatorStoryPointStatus,
)
from intergrax.runtime.diagnostics.diagnostic_operator_story_projection import (
    _evidence_refs_for_finding,
    project_operator_story,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticFinding
from intergrax.runtime.diagnostics.diagnostic_precision import DiagnosticCertainty
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyScope
from intergrax.contracts.external_operations.failure import ExternalOperationFailureKind
from intergrax.runtime.events.payloads.canonical import (
    ExternalOperationFailurePayloadV1,
)
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticOccurrenceReadStatus,
    DiagnosticProblemOccurrenceView,
    grouping_provenance_from_problem_provenance,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem
from intergrax.runtime.diagnostics.diagnostic_subject import (
    ApplicationDiagnosticSubjectRef,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingMethod,
    ProblemGroupingSubjectRef,
    problem_grouping_subject_ref_for_execution,
)
from intergrax.runtime.diagnostics.diagnostic_read_models import DiagnosticProblemDetail
from intergrax.runtime.events.payloads.canonical import ExecutionFailurePayloadV1
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from unittest.mock import MagicMock

from tests.unit.runtime.diagnostics.test_diagnostic_read_reconstruction_scale import (
    _CountingExecutionReconstructionReader,
    _occurrence_persistence_for,
    _persist_problem,
    read_service_for_tests,
)
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrencePage,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-dg003"
_PROVIDER = "celery"
_BASE_TS = datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc)
_ANALYZER = LifecycleAnomalyAnalyzer()
_BUILDER = DiagnosticAssessmentBuilder()


def _runtime_event(
    *,
    event_id: EventId | None = None,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
    event_type: RuntimeEventType,
    timestamp: datetime,
    payload: dict[str, object] | None = None,
) -> RuntimeEvent:
    base = sample_runtime_event(
        event_id=event_id,
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    return base.model_copy(
        update={
            "event_type": event_type,
            "timestamp": timestamp,
            "payload": payload or {},
        }
    )


def _execution_failed_payload() -> dict[str, object]:
    return ExecutionFailurePayloadV1(
        failure_kind=ExecutionFailureKind.DELEGATE_EXCEPTION,
        safe_summary="execution failed",
    ).to_envelope()


def _positioned(
    event: RuntimeEvent,
    position: int,
) -> PositionedRuntimeEvent:
    return PositionedRuntimeEvent(
        event=event,
        position=ExecutionEventPosition(position),
    )


def _reconstruction(
    *,
    task_id: TaskId,
    run_id: RunId,
    positioned_events: tuple[PositionedRuntimeEvent, ...],
    completeness: RuntimeHistoryCompleteness = RuntimeHistoryCompleteness.COMPLETE,
) -> ExecutionReconstruction:
    return ExecutionReconstruction(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        causal_evidence=(),
        positioned_events=positioned_events,
        attempts=(),
        runtime_history_completeness=completeness,
    )


def _assess(reconstruction: ExecutionReconstruction):
    lifecycle = _ANALYZER.analyze(reconstruction)
    return _BUILDER.assess(reconstruction, lifecycle)


def _reconstruction_position_by_event_id(
    reconstruction: ExecutionReconstruction,
) -> dict[EventId, ExecutionEventPosition]:
    return {
        row.event.event_id: row.position for row in reconstruction.positioned_events
    }


def _assert_story_evidence_position_invariant(
    story,
    reconstruction: ExecutionReconstruction,
) -> None:
    by_event = _reconstruction_position_by_event_id(reconstruction)
    for ref in story.supporting_evidence:
        if ref.event_id is None or ref.position is None:
            continue
        assert by_event[ref.event_id] == ref.position


def _event_evidence_positions(story) -> dict[EventId, int]:
    result: dict[EventId, int] = {}
    for ref in story.supporting_evidence:
        if ref.event_id is not None and ref.position is not None:
            result[ref.event_id] = ref.position.value
    return result


def _occurrence_view(
    *,
    task_id: TaskId,
    run_id: RunId,
    assessment,
) -> DiagnosticProblemOccurrenceView:
    return DiagnosticProblemOccurrenceView(
        subject_ref=problem_grouping_subject_ref_for_execution(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
        ),
        observed_at=_BASE_TS,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        method=ProblemGroupingMethod.DETERMINISTIC,
        read_status=DiagnosticOccurrenceReadStatus.AVAILABLE,
        assessment=assessment,
        execution_lineage=None,
    )


def test_standard_failure_last_good_and_first_failed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    events = (
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.STEP_STARTED,
                timestamp=_BASE_TS,
            ),
            1,
        ),
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.STEP_COMPLETED,
                timestamp=_BASE_TS + timedelta(seconds=1),
            ),
            2,
        ),
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.EXECUTION_FAILED,
                timestamp=_BASE_TS + timedelta(seconds=2),
                payload=_execution_failed_payload(),
            ),
            3,
        ),
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.STEP_STARTED,
                timestamp=_BASE_TS + timedelta(seconds=3),
            ),
            4,
        ),
    )
    reconstruction = _reconstruction(
        task_id=task_id, run_id=run_id, positioned_events=events
    )
    assessment = _assess(reconstruction)
    story = project_operator_story(
        occurrence=_occurrence_view(
            task_id=task_id, run_id=run_id, assessment=assessment
        ),
        reconstruction=reconstruction,
        assessment=assessment,
    )

    assert story.first_failed.status is DiagnosticOperatorStoryPointStatus.AVAILABLE
    assert story.first_failed.position is not None
    assert story.first_failed.position.value == 3
    assert story.first_failed.finding_kind is DiagnosticFindingKind.EXECUTION_FAILED
    assert story.last_good.status is DiagnosticOperatorStoryPointStatus.AVAILABLE
    assert story.last_good.position is not None
    assert story.last_good.position.value == 2
    assert story.transition is not None
    assert story.transition.intervening_event_count == 0
    assert story.transition.is_causal_claim is False
    assert (
        story.first_failed_scope
        is DiagnosticOperatorStoryFirstFailedScope.PROVEN_IN_AVAILABLE_EVIDENCE
    )
    assert story.supporting_evidence
    _assert_story_evidence_position_invariant(story, reconstruction)
    failure_event_id = story.first_failed.event_id
    assert failure_event_id is not None
    assert _event_evidence_positions(story)[failure_event_id] == 3


def test_timestamp_inversion_position_still_authoritative() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    late_ts = _BASE_TS + timedelta(hours=1)
    early_ts = _BASE_TS
    events = (
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.STEP_COMPLETED,
                timestamp=late_ts,
            ),
            10,
        ),
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.EXECUTION_FAILED,
                timestamp=early_ts,
                payload=_execution_failed_payload(),
            ),
            11,
        ),
    )
    reconstruction = _reconstruction(
        task_id=task_id, run_id=run_id, positioned_events=events
    )
    assessment = _assess(reconstruction)
    story = project_operator_story(
        occurrence=_occurrence_view(
            task_id=task_id, run_id=run_id, assessment=assessment
        ),
        reconstruction=reconstruction,
        assessment=assessment,
    )

    assert story.last_good.position is not None
    assert story.first_failed.position is not None
    assert story.last_good.position.value == 10
    assert story.first_failed.position.value == 11
    assert story.last_good.observed_at == late_ts
    assert story.first_failed.observed_at == early_ts


def test_duplicate_timestamps_deterministic() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    same_ts = _BASE_TS
    events = (
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.STEP_STARTED,
                timestamp=same_ts,
            ),
            1,
        ),
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.STEP_COMPLETED,
                timestamp=same_ts,
            ),
            2,
        ),
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.EXECUTION_FAILED,
                timestamp=same_ts,
                payload=_execution_failed_payload(),
            ),
            3,
        ),
    )
    reconstruction = _reconstruction(
        task_id=task_id, run_id=run_id, positioned_events=events
    )
    assessment = _assess(reconstruction)
    story_a = project_operator_story(
        occurrence=_occurrence_view(
            task_id=task_id, run_id=run_id, assessment=assessment
        ),
        reconstruction=reconstruction,
        assessment=assessment,
    )
    story_b = project_operator_story(
        occurrence=_occurrence_view(
            task_id=task_id, run_id=run_id, assessment=assessment
        ),
        reconstruction=reconstruction,
        assessment=assessment,
    )
    assert story_a == story_b
    assert story_a.last_good.position is not None
    assert story_a.last_good.position.value == 2


def test_first_event_failure_no_last_good() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    events = (
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.EXECUTION_FAILED,
                timestamp=_BASE_TS,
                payload=_execution_failed_payload(),
            ),
            1,
        ),
    )
    reconstruction = _reconstruction(
        task_id=task_id, run_id=run_id, positioned_events=events
    )
    assessment = _assess(reconstruction)
    story = project_operator_story(
        occurrence=_occurrence_view(
            task_id=task_id, run_id=run_id, assessment=assessment
        ),
        reconstruction=reconstruction,
        assessment=assessment,
    )
    assert story.first_failed.status is DiagnosticOperatorStoryPointStatus.AVAILABLE
    assert story.last_good.status is DiagnosticOperatorStoryPointStatus.UNAVAILABLE
    assert story.last_good.unavailable_reason == "no_prior_factual_event_available"


def test_no_failure_when_history_complete() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    events = (
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.STEP_STARTED,
                timestamp=_BASE_TS,
            ),
            1,
        ),
    )
    reconstruction = _reconstruction(
        task_id=task_id, run_id=run_id, positioned_events=events
    )
    assessment = _assess(reconstruction)
    story = project_operator_story(
        occurrence=_occurrence_view(
            task_id=task_id, run_id=run_id, assessment=assessment
        ),
        reconstruction=reconstruction,
        assessment=assessment,
    )
    assert story.first_failed.status is DiagnosticOperatorStoryPointStatus.UNAVAILABLE
    assert story.last_good.status is DiagnosticOperatorStoryPointStatus.UNAVAILABLE


def test_incomplete_history_scope() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    events = (
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.EXECUTION_FAILED,
                timestamp=_BASE_TS,
                payload=_execution_failed_payload(),
            ),
            5,
        ),
    )
    reconstruction = _reconstruction(
        task_id=task_id,
        run_id=run_id,
        positioned_events=events,
        completeness=RuntimeHistoryCompleteness.TRUNCATED,
    )
    assessment = _assess(reconstruction)
    story = project_operator_story(
        occurrence=_occurrence_view(
            task_id=task_id, run_id=run_id, assessment=assessment
        ),
        reconstruction=reconstruction,
        assessment=assessment,
    )
    assert story.first_failed_scope is (
        DiagnosticOperatorStoryFirstFailedScope.FIRST_OBSERVED_IN_AVAILABLE_EVIDENCE
    )
    assert any("truncated" in item.lower() for item in story.limitations)


def test_multiple_failures_first_by_position() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    events = (
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.STEP_COMPLETED,
                timestamp=_BASE_TS,
            ),
            1,
        ),
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.EXECUTION_FAILED,
                timestamp=_BASE_TS,
                payload=_execution_failed_payload(),
            ),
            2,
        ),
        _positioned(
            _runtime_event(
                event_id=mint_event_id(),
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.EXECUTION_FAILED,
                timestamp=_BASE_TS,
                payload=_execution_failed_payload(),
            ),
            8,
        ),
    )
    reconstruction = _reconstruction(
        task_id=task_id, run_id=run_id, positioned_events=events
    )
    assessment = _assess(reconstruction)
    story = project_operator_story(
        occurrence=_occurrence_view(
            task_id=task_id, run_id=run_id, assessment=assessment
        ),
        reconstruction=reconstruction,
        assessment=assessment,
    )
    assert story.first_failed.position is not None
    assert story.first_failed.position.value == 2


def test_no_execution_subject_unavailable() -> None:
    app_subject = ProblemGroupingSubjectRef(
        subject=ApplicationDiagnosticSubjectRef(
            tenant_id=_TENANT,
            application_id="app-dg003",
            instance_id="instance-1",
        ),
    )
    occurrence = DiagnosticProblemOccurrenceView(
        subject_ref=app_subject,
        observed_at=_BASE_TS,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        method=ProblemGroupingMethod.DETERMINISTIC,
        read_status=DiagnosticOccurrenceReadStatus.AVAILABLE,
        assessment=None,
        execution_lineage=None,
    )
    story = project_operator_story(
        occurrence=occurrence,
        reconstruction=None,
        assessment=None,
    )
    assert story.first_failed.unavailable_reason == "no_execution_subject"


def test_investigation_view_includes_operator_story() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    events = (
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.STEP_COMPLETED,
                timestamp=_BASE_TS,
            ),
            1,
        ),
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.EXECUTION_FAILED,
                timestamp=_BASE_TS,
                payload=_execution_failed_payload(),
            ),
            2,
        ),
    )
    reconstruction = _reconstruction(
        task_id=task_id, run_id=run_id, positioned_events=events
    )
    assessment = _assess(reconstruction)
    problem = sample_problem(tenant_id=_TENANT)
    detail = DiagnosticProblemDetail(
        problem_id=problem.problem_id,
        tenant_id=problem.tenant_id,
        status=problem.status,
        first_seen_at=problem.first_seen_at,
        last_seen_at=problem.last_seen_at,
        occurrence_count=1,
        record_version=problem.record_version,
        grouping_provenance=grouping_provenance_from_problem_provenance(
            problem.provenance
        ),
        occurrence_aggregate_health=problem.occurrence_aggregate_health,
        occurrences=(),
        returned_occurrence_count=0,
        total_occurrence_count=1,
        is_occurrences_truncated=False,
    )
    view = project_investigation_view(
        problem_detail=detail,
        occurrence=_occurrence_view(
            task_id=task_id, run_id=run_id, assessment=assessment
        ),
        reconstruction=reconstruction,
    )
    assert view.operator_story.first_failed.position is not None
    assert view.operator_story.first_failed.position.value == 2


def test_get_investigation_no_extra_reconstruction_for_story() -> None:
    problem, persistence, _, runtime_store = _persist_problem()
    stored = _occurrence_persistence_for(persistence)
    page = stored.query_occurrences(
        tenant_id=problem.tenant_id,
        problem_id=problem.problem_id,
        limit=100,
    )
    occurrence_persistence = MagicMock()
    occurrence_persistence.query_occurrences.return_value = ProblemOccurrencePage(
        items=(page.items[0],),
        next_cursor=None,
        has_more=False,
    )
    counter = _CountingExecutionReconstructionReader(
        ExecutionReconstructor(
            runtime_events=runtime_store,
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )
    service = read_service_for_tests(
        persistence,
        counter,
        occurrence_persistence=occurrence_persistence,
    )
    result = service.get_investigation(
        tenant_id=problem.tenant_id,
        problem_id=problem.problem_id,
        occurrence_index=0,
    )
    assert result.investigation is not None
    assert result.investigation.operator_story is not None
    assert counter.reconstruction_calls == 1


def test_custom_reader_story_via_reconstruction_pipeline() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    store = InMemoryRuntimeEventStore()
    failed = sample_runtime_event(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ).model_copy(
        update={
            "event_type": RuntimeEventType.EXECUTION_FAILED,
            "payload": _execution_failed_payload(),
        }
    )
    store.append(failed, tenant_id=_TENANT)
    reader = ExecutionReconstructor(
        runtime_events=store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    )
    reconstruction = reader.reconstruct_execution(_TENANT, task_id, run_id)
    assessment = _assess(reconstruction)
    story = project_operator_story(
        occurrence=_occurrence_view(
            task_id=task_id, run_id=run_id, assessment=assessment
        ),
        reconstruction=reconstruction,
        assessment=assessment,
    )
    assert story.first_failed.status is DiagnosticOperatorStoryPointStatus.AVAILABLE
    _assert_story_evidence_position_invariant(story, reconstruction)


def test_lifecycle_two_event_evidence_positions_and_failure_anchor() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    event_a_id = mint_event_id()
    event_b_id = mint_event_id()
    events = (
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.TASK_CREATED,
                timestamp=_BASE_TS,
            ),
            1,
        ),
        _positioned(
            _runtime_event(
                event_id=event_a_id,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.TASK_COMPLETED,
                timestamp=_BASE_TS + timedelta(seconds=1),
            ),
            10,
        ),
        _positioned(
            _runtime_event(
                event_id=event_b_id,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.RETRY_SCHEDULED,
                timestamp=_BASE_TS + timedelta(seconds=2),
            ),
            14,
        ),
    )
    reconstruction = _reconstruction(
        task_id=task_id, run_id=run_id, positioned_events=events
    )
    assessment = _assess(reconstruction)
    story = project_operator_story(
        occurrence=_occurrence_view(
            task_id=task_id, run_id=run_id, assessment=assessment
        ),
        reconstruction=reconstruction,
        assessment=assessment,
    )

    positions = _event_evidence_positions(story)
    assert positions[event_a_id] == 10
    assert positions[event_b_id] == 14
    assert story.first_failed.position is not None
    assert story.first_failed.position.value == 14
    assert story.first_failed.event_id == event_b_id
    _assert_story_evidence_position_invariant(story, reconstruction)


def _external_operation_failed_payload(execution_id: ExecutionId) -> dict[str, object]:
    return ExternalOperationFailurePayloadV1(
        execution_id=execution_id,
        operation_attempt_id="op-1",
        provider_id="provider-1",
        operation_type="llm_call",
        failure_kind=ExternalOperationFailureKind.TIMEOUT,
    ).to_envelope()


def test_external_operation_failed_evidence_canonical_position() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    failure_event_id = mint_event_id()
    events = (
        _positioned(
            _runtime_event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.STEP_COMPLETED,
                timestamp=_BASE_TS,
            ),
            4,
        ),
        _positioned(
            _runtime_event(
                event_id=failure_event_id,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.EXTERNAL_OPERATION_FAILED,
                timestamp=_BASE_TS + timedelta(seconds=1),
                payload=_external_operation_failed_payload(execution_id),
            ),
            7,
        ),
    )
    reconstruction = _reconstruction(
        task_id=task_id, run_id=run_id, positioned_events=events
    )
    assessment = _assess(reconstruction)
    story = project_operator_story(
        occurrence=_occurrence_view(
            task_id=task_id, run_id=run_id, assessment=assessment
        ),
        reconstruction=reconstruction,
        assessment=assessment,
    )
    assert story.first_failed.position is not None
    assert story.first_failed.position.value == 7
    assert _event_evidence_positions(story)[failure_event_id] == 7
    _assert_story_evidence_position_invariant(story, reconstruction)


def test_event_after_terminal_evidence_positions_match_reconstruction() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    store = InMemoryRuntimeEventStore()
    for event_type in (
        RuntimeEventType.TASK_CREATED,
        RuntimeEventType.TASK_COMPLETED,
        RuntimeEventType.RETRY_SCHEDULED,
    ):
        store.append(
            sample_runtime_event(
                tenant_id=_TENANT,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            ).model_copy(update={"event_type": event_type}),
            tenant_id=_TENANT,
        )
    reconstruction = ExecutionReconstructor(
        runtime_events=store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(_TENANT, task_id, run_id)
    assessment = _assess(reconstruction)
    story = project_operator_story(
        occurrence=_occurrence_view(
            task_id=task_id, run_id=run_id, assessment=assessment
        ),
        reconstruction=reconstruction,
        assessment=assessment,
    )
    terminal_finding = next(
        f
        for f in assessment.findings
        if f.kind is DiagnosticFindingKind.EVENT_AFTER_TERMINAL
    )
    assert len(terminal_finding.supporting_event_ids) == 2
    prior_id, violating_id = terminal_finding.supporting_event_ids
    by_event = _reconstruction_position_by_event_id(reconstruction)
    assert _event_evidence_positions(story)[prior_id] == by_event[prior_id].value
    assert (
        _event_evidence_positions(story)[violating_id] == by_event[violating_id].value
    )
    _assert_story_evidence_position_invariant(story, reconstruction)


def test_disallowed_after_failed_evidence_positions_match_reconstruction() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    store = InMemoryRuntimeEventStore()
    for event_type in (
        RuntimeEventType.TASK_CREATED,
        RuntimeEventType.TASK_FAILED,
        RuntimeEventType.PAUSE_REQUESTED,
    ):
        store.append(
            sample_runtime_event(
                tenant_id=_TENANT,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            ).model_copy(update={"event_type": event_type}),
            tenant_id=_TENANT,
        )
    reconstruction = ExecutionReconstructor(
        runtime_events=store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(_TENANT, task_id, run_id)
    assessment = _assess(reconstruction)
    story = project_operator_story(
        occurrence=_occurrence_view(
            task_id=task_id, run_id=run_id, assessment=assessment
        ),
        reconstruction=reconstruction,
        assessment=assessment,
    )
    finding = next(
        f
        for f in assessment.findings
        if f.kind is DiagnosticFindingKind.DISALLOWED_AFTER_FAILED
    )
    for event_id in finding.supporting_event_ids:
        row = next(
            r for r in reconstruction.positioned_events if r.event.event_id == event_id
        )
        assert _event_evidence_positions(story)[event_id] == row.position.value
    _assert_story_evidence_position_invariant(story, reconstruction)


def test_multiple_terminal_outcomes_evidence_positions_match_reconstruction() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    store = InMemoryRuntimeEventStore()
    for event_type in (
        RuntimeEventType.TASK_CREATED,
        RuntimeEventType.TASK_FAILED,
        RuntimeEventType.TASK_COMPLETED,
    ):
        store.append(
            sample_runtime_event(
                tenant_id=_TENANT,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            ).model_copy(update={"event_type": event_type}),
            tenant_id=_TENANT,
        )
    reconstruction = ExecutionReconstructor(
        runtime_events=store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(_TENANT, task_id, run_id)
    assessment = _assess(reconstruction)
    story = project_operator_story(
        occurrence=_occurrence_view(
            task_id=task_id, run_id=run_id, assessment=assessment
        ),
        reconstruction=reconstruction,
        assessment=assessment,
    )
    finding = next(
        f
        for f in assessment.findings
        if f.kind is DiagnosticFindingKind.MULTIPLE_TERMINAL_OUTCOMES
    )
    for event_id in finding.supporting_event_ids:
        row = next(
            r for r in reconstruction.positioned_events if r.event.event_id == event_id
        )
        assert _event_evidence_positions(story)[event_id] == row.position.value
    _assert_story_evidence_position_invariant(story, reconstruction)


def test_evidence_id_refs_do_not_inherit_failure_position() -> None:
    evidence_id = mint_event_id()
    finding = DiagnosticFinding(
        kind=DiagnosticFindingKind.EXECUTION_FAILED,
        scope=LifecycleAnomalyScope.ATTEMPT,
        attempt_id=mint_attempt_id(),
        certainty=DiagnosticCertainty.PROVEN,
        claim="test",
        source_anomaly_kind=None,
        supporting_event_ids=(),
        supporting_evidence_ids=(evidence_id,),
        supporting_positions=(ExecutionEventPosition(99),),
        execution_id=mint_execution_id(),
    )
    refs = _evidence_refs_for_finding(finding, ())
    assert len(refs) == 1
    assert refs[0].evidence_id == evidence_id
    assert refs[0].position is None


def test_duplicate_event_evidence_ref_deduped_after_position_fix() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    failure_event_id = mint_event_id()
    events = (
        _positioned(
            _runtime_event(
                event_id=failure_event_id,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                event_type=RuntimeEventType.EXECUTION_FAILED,
                timestamp=_BASE_TS,
                payload=_execution_failed_payload(),
            ),
            3,
        ),
    )
    reconstruction = _reconstruction(
        task_id=task_id, run_id=run_id, positioned_events=events
    )
    assessment = _assess(reconstruction)
    story = project_operator_story(
        occurrence=_occurrence_view(
            task_id=task_id, run_id=run_id, assessment=assessment
        ),
        reconstruction=reconstruction,
        assessment=assessment,
    )
    event_refs = [
        ref for ref in story.supporting_evidence if ref.event_id == failure_event_id
    ]
    assert len(event_refs) == 1
    assert event_refs[0].position is not None
    assert event_refs[0].position.value == 3
