# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.diagnostics.execution_reconstruction import (
    ExecutionReconstructor,
    RuntimeHistoryCompleteness,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import (
    LifecycleAnomalyAnalyzer,
    LifecycleAnomalyKind,
    LifecycleAnomalyScope,
)
from intergrax.runtime.events.asof_projection import (
    apply_lifecycle_event,
    InvalidRunExecutionHistoryError,
    RunExecutionLifecycleStatus,
    RunLifecycleViolationKind,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.causal_evidence import (
    CausalRelationKind,
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_PROVIDER = "celery"
_ANALYZER = LifecycleAnomalyAnalyzer()


def _reconstructor() -> ExecutionReconstructor:
    return ExecutionReconstructor(
        runtime_events=InMemoryRuntimeEventStore(),
        causal_evidence=InMemoryCausalEvidencePersistence(),
    )


def _transport_ref(*, tenant_id: str = _TENANT_A, transport_task_id: str = "celery-task-1") -> MessageBusTaskRef:
    return MessageBusTaskRef(
        provider=_PROVIDER,
        task_id=transport_task_id,
        tenant_id=tenant_id,
    )


def _execution_ref(
    *,
    tenant_id: str = _TENANT_A,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
) -> RuntimeExecutionRef:
    return RuntimeExecutionRef(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id=tenant_id,
    )


def _causal_evidence(
    *,
    tenant_id: str = _TENANT_A,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    transport_task_id: str = "celery-task-1",
    recorded_at: datetime | None = None,
    evidence_id: str | None = None,
) -> PlatformCausalEvidence:
    return PlatformCausalEvidence(
        evidence_id=evidence_id or mint_event_id(),
        relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
        tenant_id=tenant_id,
        source=_transport_ref(tenant_id=tenant_id, transport_task_id=transport_task_id),
        target=_execution_ref(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        ),
        recorded_at=recorded_at or datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc),
    )


def _append_event(
    store: InMemoryRuntimeEventStore,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    event_type: RuntimeEventType = RuntimeEventType.STEP_STARTED,
    timestamp: datetime | None = None,
) -> None:
    event = sample_runtime_event(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    ).model_copy(update={"event_type": event_type})
    if timestamp is not None:
        event = event.model_copy(update={"timestamp": timestamp})
    store.append(event, tenant_id=tenant_id)


def _append_sequence(
    store: InMemoryRuntimeEventStore,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    event_types: list[RuntimeEventType],
) -> None:
    for event_type in event_types:
        _append_event(
            store,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            event_type=event_type,
        )


def test_causal_only_attempt_emits_single_anomaly() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    causal_store = InMemoryCausalEvidencePersistence()
    evidence = _causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    causal_store.append(evidence)

    reconstruction = ExecutionReconstructor(
        runtime_events=InMemoryRuntimeEventStore(),
        causal_evidence=causal_store,
    ).reconstruct_execution(_TENANT_A, task_id, run_id)
    analysis = _ANALYZER.analyze(reconstruction)

    assert len(analysis.anomalies) == 1
    anomaly = analysis.anomalies[0]
    assert anomaly.kind is LifecycleAnomalyKind.CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY
    assert anomaly.scope is LifecycleAnomalyScope.ATTEMPT
    assert anomaly.attempt_id == attempt_id
    assert anomaly.supporting_evidence_ids == (evidence.evidence_id,)
    assert anomaly.supporting_event_ids == ()
    assert anomaly.supporting_positions == ()


def test_event_only_attempt_without_background_context_no_false_positive() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    _append_event(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )

    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(_TENANT_A, task_id, run_id)
    analysis = _ANALYZER.analyze(reconstruction)

    assert analysis.anomalies == ()


def test_event_only_attempt_in_background_execution_emits_anomaly() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a3 = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    causal_store.append(_causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_a1))
    _append_event(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a3,
    )

    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=causal_store,
    ).reconstruct_execution(_TENANT_A, task_id, run_id)
    analysis = _ANALYZER.analyze(reconstruction)

    missing = [
        anomaly
        for anomaly in analysis.anomalies
        if anomaly.kind is LifecycleAnomalyKind.RUNTIME_ATTEMPT_WITHOUT_CAUSAL_EVIDENCE
    ]
    assert len(missing) == 1
    assert missing[0].attempt_id == attempt_a3


def test_retry_a1_failed_a2_completed_no_contradiction() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    causal_store.append(_causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_a1))
    causal_store.append(_causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_a2))
    _append_sequence(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        event_types=[
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_FAILED,
            RuntimeEventType.RETRY_SCHEDULED,
        ],
    )
    _append_sequence(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a2,
        event_types=[
            RuntimeEventType.RETRY_STARTED,
            RuntimeEventType.STEP_STARTED,
            RuntimeEventType.TASK_COMPLETED,
        ],
    )

    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=causal_store,
    ).reconstruct_execution(_TENANT_A, task_id, run_id)
    analysis = _ANALYZER.analyze(reconstruction)

    lifecycle_kinds = {
        LifecycleAnomalyKind.MULTIPLE_TERMINAL_OUTCOMES,
        LifecycleAnomalyKind.EVENT_AFTER_TERMINAL,
        LifecycleAnomalyKind.DISALLOWED_AFTER_FAILED,
    }
    assert not any(anomaly.kind in lifecycle_kinds for anomaly in analysis.anomalies)


def test_truncated_history_emits_truncation_only() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    for _ in range(5):
        _append_event(
            runtime_store,
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        )

    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
        initial_limit=2,
        max_limit=2,
    )
    analysis = _ANALYZER.analyze(reconstruction)

    assert reconstruction.runtime_history_completeness is RuntimeHistoryCompleteness.TRUNCATED
    truncated = [
        anomaly
        for anomaly in analysis.anomalies
        if anomaly.kind is LifecycleAnomalyKind.RUNTIME_HISTORY_TRUNCATED
    ]
    assert len(truncated) == 1
    assert truncated[0].scope is LifecycleAnomalyScope.EXECUTION
    assert not any(
        anomaly.kind is LifecycleAnomalyKind.EVENT_AFTER_TERMINAL
        for anomaly in analysis.anomalies
    )


def test_determinism_same_reconstruction_same_analysis() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    evidence = _causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    causal_store.append(evidence)

    reconstructor = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=causal_store,
    )
    reconstruction = reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)

    first = _ANALYZER.analyze(reconstruction)
    second = _ANALYZER.analyze(reconstruction)

    assert first == second


def test_lifecycle_violation_uses_execution_position_not_timestamp() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    base = datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc)
    _append_event(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_type=RuntimeEventType.TASK_CREATED,
        timestamp=base + timedelta(hours=2),
    )
    _append_event(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_type=RuntimeEventType.TASK_FAILED,
        timestamp=base,
    )
    _append_event(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_type=RuntimeEventType.TASK_COMPLETED,
        timestamp=base + timedelta(hours=1),
    )

    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(_TENANT_A, task_id, run_id)
    analysis = _ANALYZER.analyze(reconstruction)

    conflicting = [
        anomaly
        for anomaly in analysis.anomalies
        if anomaly.kind is LifecycleAnomalyKind.MULTIPLE_TERMINAL_OUTCOMES
    ]
    assert len(conflicting) == 1
    assert conflicting[0].supporting_positions == (
        reconstruction.positioned_events[1].position,
        reconstruction.positioned_events[2].position,
    )
    assert conflicting[0].supporting_event_ids == (
        reconstruction.positioned_events[1].event.event_id,
        reconstruction.positioned_events[2].event.event_id,
    )


def test_disallowed_after_failed_emits_distinct_anomaly_not_event_after_terminal() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    _append_sequence(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=[
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_FAILED,
            RuntimeEventType.PAUSE_REQUESTED,
        ],
    )

    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(_TENANT_A, task_id, run_id)
    analysis = _ANALYZER.analyze(reconstruction)

    disallowed = [
        anomaly
        for anomaly in analysis.anomalies
        if anomaly.kind is LifecycleAnomalyKind.DISALLOWED_AFTER_FAILED
    ]
    assert len(disallowed) == 1
    assert not any(
        anomaly.kind is LifecycleAnomalyKind.EVENT_AFTER_TERMINAL
        for anomaly in analysis.anomalies
    )
    failed_row = reconstruction.positioned_events[1]
    violating_row = reconstruction.positioned_events[2]
    assert failed_row.event.event_type is RuntimeEventType.TASK_FAILED
    assert disallowed[0].supporting_event_ids == (
        failed_row.event.event_id,
        violating_row.event.event_id,
    )
    assert disallowed[0].supporting_positions == (
        failed_row.position,
        violating_row.position,
    )


def test_completed_run_cannot_reopen_execution_scope() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    _append_event(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        event_type=RuntimeEventType.TASK_CREATED,
    )
    _append_event(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        event_type=RuntimeEventType.TASK_COMPLETED,
    )
    _append_event(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a2,
        event_type=RuntimeEventType.RETRY_STARTED,
    )

    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(_TENANT_A, task_id, run_id)
    analysis = _ANALYZER.analyze(reconstruction)

    after_terminal = [
        anomaly
        for anomaly in analysis.anomalies
        if anomaly.kind is LifecycleAnomalyKind.EVENT_AFTER_TERMINAL
    ]
    assert len(after_terminal) == 1
    assert after_terminal[0].scope is LifecycleAnomalyScope.EXECUTION
    assert after_terminal[0].supporting_positions == (
        reconstruction.positioned_events[1].position,
        reconstruction.positioned_events[2].position,
    )
    assert after_terminal[0].supporting_event_ids == (
        reconstruction.positioned_events[1].event.event_id,
        reconstruction.positioned_events[2].event.event_id,
    )


def test_step_failure_does_not_count_as_execution_terminal() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    _append_sequence(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=[
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.STEP_STARTED,
            RuntimeEventType.STEP_FAILED,
            RuntimeEventType.TASK_COMPLETED,
        ],
    )

    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(_TENANT_A, task_id, run_id)
    analysis = _ANALYZER.analyze(reconstruction)

    assert analysis.anomalies == ()


def test_tool_failure_does_not_count_as_execution_terminal() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    _append_sequence(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=[
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TOOL_REQUESTED,
            RuntimeEventType.TOOL_FAILED,
            RuntimeEventType.TASK_COMPLETED,
        ],
    )

    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(_TENANT_A, task_id, run_id)
    analysis = _ANALYZER.analyze(reconstruction)

    assert analysis.anomalies == ()


def test_lifecycle_analysis_imports_no_private_asof_helpers() -> None:
    import intergrax.runtime.diagnostics.lifecycle_analysis as lifecycle_analysis

    module_path = Path(inspect.getfile(lifecycle_analysis))
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "intergrax.runtime.events.asof_projection":
            for alias in node.names:
                assert not alias.name.startswith("_")


def test_lifecycle_violation_classification_uses_typed_kind_not_message() -> None:
    with pytest.raises(InvalidRunExecutionHistoryError) as exc_info:
        apply_lifecycle_event(
            RunExecutionLifecycleStatus.FAILED,
            RuntimeEventType.TASK_COMPLETED,
        )
    exc = exc_info.value
    assert exc.kind is RunLifecycleViolationKind.CONFLICTING_FINAL_OUTCOME
    assert exc.current_status is RunExecutionLifecycleStatus.FAILED
    assert exc.event_type is RuntimeEventType.TASK_COMPLETED

    with pytest.raises(InvalidRunExecutionHistoryError) as exc_info:
        apply_lifecycle_event(
            RunExecutionLifecycleStatus.COMPLETED,
            RuntimeEventType.RETRY_STARTED,
        )
    exc = exc_info.value
    assert exc.kind is RunLifecycleViolationKind.EVENT_AFTER_TERMINAL

    remapped = InvalidRunExecutionHistoryError(
        "presentation-only message unrelated to semantics",
        kind=RunLifecycleViolationKind.CONFLICTING_FINAL_OUTCOME,
        current_status=RunExecutionLifecycleStatus.FAILED,
        event_type=RuntimeEventType.TASK_COMPLETED,
    )
    assert remapped.kind is RunLifecycleViolationKind.CONFLICTING_FINAL_OUTCOME


def test_analysis_preserves_execution_scope_fields() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    causal_store = InMemoryCausalEvidencePersistence()
    evidence = _causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    causal_store.append(evidence)
    reconstruction = ExecutionReconstructor(
        runtime_events=InMemoryRuntimeEventStore(),
        causal_evidence=causal_store,
    ).reconstruct_execution(_TENANT_A, task_id, run_id)

    analysis = _ANALYZER.analyze(reconstruction)

    assert analysis.tenant_id == _TENANT_A
    assert analysis.task_id == task_id
    assert analysis.run_id == run_id
    assert analysis.has_anomalies is True


def _analyze_attempt_sequence(
    event_types: list[RuntimeEventType],
) -> LifecycleAnalysis:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    _append_sequence(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=event_types,
    )
    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(_TENANT_A, task_id, run_id)
    return _ANALYZER.analyze(reconstruction)


def test_event_after_terminal_propagates_lifecycle_transition() -> None:
    analysis = _analyze_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.RETRY_SCHEDULED,
        ]
    )

    after_terminal = [
        anomaly
        for anomaly in analysis.anomalies
        if anomaly.kind is LifecycleAnomalyKind.EVENT_AFTER_TERMINAL
    ]
    assert len(after_terminal) == 1
    anomaly = after_terminal[0]
    transition = anomaly.lifecycle_transition
    assert transition is not None
    assert transition.violation_kind is RunLifecycleViolationKind.EVENT_AFTER_TERMINAL
    assert transition.prior_status is RunExecutionLifecycleStatus.COMPLETED
    assert transition.violating_event_type is RuntimeEventType.RETRY_SCHEDULED


def test_structural_collision_completed_retry_scheduled_vs_task_failed() -> None:
    analysis_a = _analyze_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.RETRY_SCHEDULED,
        ]
    )
    analysis_b = _analyze_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.TASK_FAILED,
        ]
    )

    anomaly_a = next(
        anomaly
        for anomaly in analysis_a.anomalies
        if anomaly.kind is LifecycleAnomalyKind.EVENT_AFTER_TERMINAL
    )
    anomaly_b = next(
        anomaly
        for anomaly in analysis_b.anomalies
        if anomaly.kind is LifecycleAnomalyKind.EVENT_AFTER_TERMINAL
    )

    assert anomaly_a.lifecycle_transition is not None
    assert anomaly_b.lifecycle_transition is not None
    assert anomaly_a.lifecycle_transition != anomaly_b.lifecycle_transition
    assert anomaly_a.lifecycle_transition.violating_event_type is RuntimeEventType.RETRY_SCHEDULED
    assert anomaly_b.lifecycle_transition.violating_event_type is RuntimeEventType.TASK_FAILED
    assert anomaly_a.lifecycle_transition.prior_status is RunExecutionLifecycleStatus.COMPLETED
    assert anomaly_b.lifecycle_transition.prior_status is RunExecutionLifecycleStatus.COMPLETED


def test_cancelled_vs_completed_prior_status_distinguishes_descriptors() -> None:
    cancelled_analysis = _analyze_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.CANCELLED,
            RuntimeEventType.RETRY_SCHEDULED,
        ]
    )
    completed_analysis = _analyze_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.RETRY_SCHEDULED,
        ]
    )

    cancelled_transition = next(
        anomaly.lifecycle_transition
        for anomaly in cancelled_analysis.anomalies
        if anomaly.kind is LifecycleAnomalyKind.EVENT_AFTER_TERMINAL
    )
    completed_transition = next(
        anomaly.lifecycle_transition
        for anomaly in completed_analysis.anomalies
        if anomaly.kind is LifecycleAnomalyKind.EVENT_AFTER_TERMINAL
    )

    assert cancelled_transition is not None
    assert completed_transition is not None
    assert cancelled_transition.prior_status is RunExecutionLifecycleStatus.CANCELLED
    assert completed_transition.prior_status is RunExecutionLifecycleStatus.COMPLETED
    assert cancelled_transition.violating_event_type is RuntimeEventType.RETRY_SCHEDULED
    assert completed_transition.violating_event_type is RuntimeEventType.RETRY_SCHEDULED
    assert cancelled_transition != completed_transition


def test_disallowed_after_failed_lifecycle_transition_descriptor() -> None:
    analysis = _analyze_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_FAILED,
            RuntimeEventType.PAUSE_REQUESTED,
        ]
    )

    disallowed = next(
        anomaly
        for anomaly in analysis.anomalies
        if anomaly.kind is LifecycleAnomalyKind.DISALLOWED_AFTER_FAILED
    )
    transition = disallowed.lifecycle_transition
    assert transition is not None
    assert transition.violation_kind is RunLifecycleViolationKind.DISALLOWED_AFTER_FAILED
    assert transition.prior_status is RunExecutionLifecycleStatus.FAILED
    assert transition.violating_event_type is RuntimeEventType.PAUSE_REQUESTED


def test_conflicting_final_outcome_lifecycle_transition_descriptor() -> None:
    analysis = _analyze_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_FAILED,
            RuntimeEventType.TASK_COMPLETED,
        ]
    )

    conflicting = next(
        anomaly
        for anomaly in analysis.anomalies
        if anomaly.kind is LifecycleAnomalyKind.MULTIPLE_TERMINAL_OUTCOMES
    )
    transition = conflicting.lifecycle_transition
    assert transition is not None
    assert transition.violation_kind is RunLifecycleViolationKind.CONFLICTING_FINAL_OUTCOME
    assert transition.prior_status is RunExecutionLifecycleStatus.FAILED
    assert transition.violating_event_type is RuntimeEventType.TASK_COMPLETED


def test_non_lifecycle_anomaly_lifecycle_transition_is_none() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    causal_store = InMemoryCausalEvidencePersistence()
    evidence = _causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    causal_store.append(evidence)

    reconstruction = ExecutionReconstructor(
        runtime_events=InMemoryRuntimeEventStore(),
        causal_evidence=causal_store,
    ).reconstruct_execution(_TENANT_A, task_id, run_id)
    analysis = _ANALYZER.analyze(reconstruction)

    anomaly = analysis.anomalies[0]
    assert anomaly.kind is LifecycleAnomalyKind.CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY
    assert anomaly.lifecycle_transition is None


def test_lifecycle_violation_descriptor_fails_closed_on_incomplete_typed_error() -> None:
    from intergrax.runtime.diagnostics.lifecycle_analysis import (
        LifecycleAnalysisIntegrityError,
        _lifecycle_violation_descriptor_from_exc,
    )

    incomplete = InvalidRunExecutionHistoryError(
        "incomplete typed violation",
        kind=RunLifecycleViolationKind.EVENT_AFTER_TERMINAL,
        current_status=None,
        event_type=RuntimeEventType.RETRY_SCHEDULED,
    )

    with pytest.raises(LifecycleAnalysisIntegrityError, match="current_status or event_type"):
        _lifecycle_violation_descriptor_from_exc(incomplete)


def test_terminal_publish_marker_does_not_create_event_after_terminal_finding() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    for event_type in (
        RuntimeEventType.TASK_CREATED,
        RuntimeEventType.TASK_COMPLETED,
    ):
        runtime_store.append(
            sample_runtime_event(
                tenant_id=_TENANT_A,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
            ).model_copy(update={"event_type": event_type}),
            tenant_id=_TENANT_A,
        )
    runtime_store.append(
        sample_runtime_event(
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        ).model_copy(
            update={
                "event_type": RuntimeEventType.TASK_COMPLETED,
                "payload": {
                    "source": "task_lifecycle",
                    "message": "task terminal",
                },
            },
        ),
        tenant_id=_TENANT_A,
    )

    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(_TENANT_A, task_id, run_id)
    analysis = _ANALYZER.analyze(reconstruction)

    assert analysis.anomalies == ()
