# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessmentBuilder,
    DiagnosticAssessmentIntegrityError,
    DiagnosticCertainty,
    DiagnosticFindingKind,
    DiagnosticLimitationKind,
)
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.lifecycle_analysis import (
    LifecycleAnalysis,
    LifecycleAnomalyAnalyzer,
    LifecycleAnomalyKind,
    LifecycleAnomalyScope,
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
_RECONSTRUCTOR = ExecutionReconstructor(
    runtime_events=InMemoryRuntimeEventStore(),
    causal_evidence=InMemoryCausalEvidencePersistence(),
)
_ANALYZER = LifecycleAnomalyAnalyzer()
_BUILDER = DiagnosticAssessmentBuilder()


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
) -> PlatformCausalEvidence:
    return PlatformCausalEvidence(
        evidence_id=mint_event_id(),
        relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
        tenant_id=tenant_id,
        source=_transport_ref(tenant_id=tenant_id),
        target=_execution_ref(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        ),
        recorded_at=datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc),
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


def _assess(
    *,
    runtime_store: InMemoryRuntimeEventStore | None = None,
    causal_store: InMemoryCausalEvidencePersistence | None = None,
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    initial_limit: int = 1000,
    max_limit: int = 1_000_000,
):
    task_id = task_id or mint_task_id()
    run_id = run_id or mint_run_id()
    reconstructor = ExecutionReconstructor(
        runtime_events=runtime_store or InMemoryRuntimeEventStore(),
        causal_evidence=causal_store or InMemoryCausalEvidencePersistence(),
    )
    reconstruction = reconstructor.reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
        initial_limit=initial_limit,
        max_limit=max_limit,
    )
    lifecycle = _ANALYZER.analyze(reconstruction)
    assessment = _BUILDER.assess(reconstruction, lifecycle)
    return reconstruction, lifecycle, assessment


def test_causal_without_runtime_emits_proven_finding() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    causal_store = InMemoryCausalEvidencePersistence()
    evidence = _causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    causal_store.append(evidence)

    _, _, assessment = _assess(causal_store=causal_store, task_id=task_id, run_id=run_id)

    assert len(assessment.findings) == 1
    finding = assessment.findings[0]
    assert finding.kind is DiagnosticFindingKind.CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY
    assert finding.source_anomaly_kind is LifecycleAnomalyKind.CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY
    assert finding.certainty is DiagnosticCertainty.PROVEN
    assert finding.attempt_id == attempt_id
    assert finding.supporting_evidence_ids == (evidence.evidence_id,)
    assert finding.supporting_event_ids == ()
    assert "RuntimeEvent history" in finding.claim
    assert "worker" not in finding.claim.lower()
    assert "crash" not in finding.claim.lower()
    assert assessment.limitations == ()


def test_event_after_terminal_finding_cites_terminal_and_violating_events() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
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
        attempt_id=mint_attempt_id(),
        event_type=RuntimeEventType.RETRY_STARTED,
    )

    reconstruction, _, assessment = _assess(
        runtime_store=runtime_store,
        task_id=task_id,
        run_id=run_id,
    )

    after_terminal = [
        finding
        for finding in assessment.findings
        if finding.kind is DiagnosticFindingKind.EVENT_AFTER_TERMINAL
    ]
    assert len(after_terminal) == 1
    finding = after_terminal[0]
    assert finding.certainty is DiagnosticCertainty.PROVEN
    assert finding.source_anomaly_kind is LifecycleAnomalyKind.EVENT_AFTER_TERMINAL
    assert finding.supporting_event_ids == (
        reconstruction.positioned_events[1].event.event_id,
        reconstruction.positioned_events[2].event.event_id,
    )
    assert finding.supporting_positions == (
        reconstruction.positioned_events[1].position,
        reconstruction.positioned_events[2].position,
    )


def test_disallowed_after_failed_distinguishes_from_event_after_terminal() -> None:
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

    reconstruction, _, assessment = _assess(
        runtime_store=runtime_store,
        task_id=task_id,
        run_id=run_id,
    )

    disallowed = [
        finding
        for finding in assessment.findings
        if finding.kind is DiagnosticFindingKind.DISALLOWED_AFTER_FAILED
    ]
    assert len(disallowed) == 1
    assert not any(
        finding.kind is DiagnosticFindingKind.EVENT_AFTER_TERMINAL
        for finding in assessment.findings
    )
    failed_row = reconstruction.positioned_events[1]
    violating_row = reconstruction.positioned_events[2]
    assert disallowed[0].source_anomaly_kind is LifecycleAnomalyKind.DISALLOWED_AFTER_FAILED
    assert "FAILED" in disallowed[0].claim
    assert "terminal" not in disallowed[0].claim.lower()
    assert disallowed[0].supporting_event_ids == (
        failed_row.event.event_id,
        violating_row.event.event_id,
    )
    assert disallowed[0].supporting_positions == (
        failed_row.position,
        violating_row.position,
    )


def test_truncated_history_emits_limitation_not_complete_diagnosis() -> None:
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

    _, _, assessment = _assess(
        runtime_store=runtime_store,
        task_id=task_id,
        run_id=run_id,
        initial_limit=2,
        max_limit=2,
    )

    assert len(assessment.limitations) == 1
    limitation = assessment.limitations[0]
    assert limitation.kind is DiagnosticLimitationKind.RUNTIME_HISTORY_TRUNCATED
    assert limitation.source_anomaly_kind is LifecycleAnomalyKind.RUNTIME_HISTORY_TRUNCATED
    assert "truncated" in limitation.factual_message.lower()
    assert not any(
        finding.kind is DiagnosticFindingKind.RUNTIME_HISTORY_TRUNCATED
        for finding in assessment.findings
    )


def test_no_anomalies_yields_empty_findings_and_limitations() -> None:
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

    _, _, assessment = _assess(runtime_store=runtime_store, task_id=task_id, run_id=run_id)

    assert assessment.findings == ()
    assert assessment.limitations == ()


def test_input_scope_mismatch_raises_integrity_error() -> None:
    task_id_a = mint_task_id()
    run_id_a = mint_run_id()
    task_id_b = mint_task_id()
    run_id_b = mint_run_id()
    causal_store = InMemoryCausalEvidencePersistence()
    causal_store.append(
        _causal_evidence(task_id=task_id_a, run_id=run_id_a, attempt_id=mint_attempt_id())
    )

    reconstruction = ExecutionReconstructor(
        runtime_events=InMemoryRuntimeEventStore(),
        causal_evidence=causal_store,
    ).reconstruct_execution(_TENANT_A, task_id_a, run_id_a)
    lifecycle = LifecycleAnalysis(
        tenant_id=_TENANT_A,
        task_id=task_id_b,
        run_id=run_id_b,
        anomalies=(),
    )

    with pytest.raises(DiagnosticAssessmentIntegrityError, match="task_id"):
        _BUILDER.assess(reconstruction, lifecycle)


def test_determinism_same_inputs_same_assessment() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    causal_store = InMemoryCausalEvidencePersistence()
    causal_store.append(_causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_id))

    reconstructor = ExecutionReconstructor(
        runtime_events=InMemoryRuntimeEventStore(),
        causal_evidence=causal_store,
    )
    reconstruction = reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)
    lifecycle = _ANALYZER.analyze(reconstruction)

    first = _BUILDER.assess(reconstruction, lifecycle)
    second = _BUILDER.assess(reconstruction, lifecycle)

    assert first == second


def test_all_lifecycle_anomaly_kinds_accounted_for() -> None:
    from intergrax.runtime.diagnostics import diagnostic_assessment

    mapped_kinds = set(diagnostic_assessment._ANOMALY_OUTPUT_KIND)
    assert mapped_kinds == set(LifecycleAnomalyKind)
    for kind in LifecycleAnomalyKind:
        output = diagnostic_assessment._ANOMALY_OUTPUT_KIND[kind]
        assert isinstance(output, diagnostic_assessment.DiagnosticOutputKind)
        if output is diagnostic_assessment.DiagnosticOutputKind.FINDING:
            assert kind in diagnostic_assessment._ANOMALY_TO_FINDING_KIND
        if output is diagnostic_assessment.DiagnosticOutputKind.LIMITATION:
            assert kind in diagnostic_assessment._ANOMALY_TO_LIMITATION_KIND
        if kind is LifecycleAnomalyKind.RUNTIME_HISTORY_TRUNCATED:
            assert output is diagnostic_assessment.DiagnosticOutputKind.LIMITATION
            assert (
                diagnostic_assessment._ANOMALY_TO_LIMITATION_KIND[kind]
                is DiagnosticLimitationKind.RUNTIME_HISTORY_TRUNCATED
            )


def test_finding_order_preserves_lifecycle_anomaly_order() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a3 = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    causal_store.append(_causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_a1))
    causal_store.append(_causal_evidence(task_id=task_id, run_id=run_id, attempt_id=mint_attempt_id()))
    _append_event(
        runtime_store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a3,
    )

    _, lifecycle, assessment = _assess(
        runtime_store=runtime_store,
        causal_store=causal_store,
        task_id=task_id,
        run_id=run_id,
    )

    expected_kinds = [
        anomaly.kind
        for anomaly in lifecycle.anomalies
        if anomaly.kind is not LifecycleAnomalyKind.RUNTIME_HISTORY_TRUNCATED
    ]
    actual_kinds = [finding.source_anomaly_kind for finding in assessment.findings]
    assert actual_kinds == expected_kinds


def _assess_attempt_sequence(event_types: list[RuntimeEventType]):
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
    return _assess(runtime_store=runtime_store, task_id=task_id, run_id=run_id)


def test_diagnostic_finding_passes_lifecycle_transition_from_anomaly() -> None:
    _, lifecycle, assessment = _assess_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.RETRY_SCHEDULED,
        ]
    )

    anomaly = next(
        item
        for item in lifecycle.anomalies
        if item.kind is LifecycleAnomalyKind.EVENT_AFTER_TERMINAL
    )
    finding = next(
        item
        for item in assessment.findings
        if item.kind is DiagnosticFindingKind.EVENT_AFTER_TERMINAL
    )

    assert finding.lifecycle_transition is anomaly.lifecycle_transition
    assert finding.lifecycle_transition is not None


def test_structural_collision_diagnostic_findings_differ() -> None:
    _, lifecycle_a, assessment_a = _assess_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.RETRY_SCHEDULED,
        ]
    )
    _, lifecycle_b, assessment_b = _assess_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.TASK_FAILED,
        ]
    )

    finding_a = next(
        item
        for item in assessment_a.findings
        if item.kind is DiagnosticFindingKind.EVENT_AFTER_TERMINAL
    )
    finding_b = next(
        item
        for item in assessment_b.findings
        if item.kind is DiagnosticFindingKind.EVENT_AFTER_TERMINAL
    )

    assert finding_a.lifecycle_transition != finding_b.lifecycle_transition
    assert finding_a.kind == finding_b.kind
    assert finding_a.scope == finding_b.scope
    assert finding_a.source_anomaly_kind == finding_b.source_anomaly_kind


def test_non_lifecycle_finding_lifecycle_transition_is_none() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    causal_store = InMemoryCausalEvidencePersistence()
    causal_store.append(_causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_id))

    _, lifecycle, assessment = _assess(
        causal_store=causal_store,
        task_id=task_id,
        run_id=run_id,
    )

    anomaly = lifecycle.anomalies[0]
    finding = assessment.findings[0]
    assert anomaly.lifecycle_transition is None
    assert finding.lifecycle_transition is None
