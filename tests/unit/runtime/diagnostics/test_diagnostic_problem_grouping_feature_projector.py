# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessment,
    DiagnosticCertainty,
    DiagnosticFinding,
    DiagnosticFindingKind,
)
from intergrax.runtime.diagnostics.diagnostic_problem_grouping_feature_projector import (
    DiagnosticProblemGroupingFeatureProjector,
    select_positioned_events_for_grouping,
)
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.execution_reconstruction import (
    ExecutionReconstruction,
    RuntimeHistoryCompleteness,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import (
    LifecycleAnomalyKind,
    LifecycleAnomalyScope,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingAssessmentInput,
    ProblemGroupingCandidate,
    ProblemGroupingEngine,
    ProblemGroupingInput,
    ProblemGroupingIntegrityError,
    ProblemGroupingMethod,
    ProblemGroupingProvenance,
    ProblemGroupingStrategyCharacteristics,
    ProblemGroupingStrategyId,
    ProblemGroupingStrategyRegistry,
    ProblemGroupingStrategyResult,
    ProblemGroupingStrategyVersion,
    normalize_assessment,
)
from intergrax.runtime.diagnostics.problem_grouping_features import (
    MAX_TEXT_EVIDENCE_CHARS,
    ProblemGroupingFeatureIntegrityError,
    ProblemGroupingFeatureSourceFacts,
    ProblemGroupingTextEvidenceSourceKind,
    project_assessment_features,
)
from intergrax.runtime.events.event_taxonomy import EventCategory
from intergrax.runtime.events.execution_position import ExecutionEventPosition, PositionedRuntimeEvent
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.causal_evidence import (
    CausalRelationKind,
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)
from intergrax.runtime.observability.export_attributes import ApplicationObservabilityAttributes
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_KIND_PLATFORM_TOOL_FAILURE,
    PROBLEM_SEVERITY_ERROR,
    PROBLEM_SOURCE_LAYER_TOOL,
    PlatformProblemSignal,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_PROVIDER = "celery"
_FAKE_STRATEGY_ID = ProblemGroupingStrategyId("test.feature_inspector")
_FAKE_STRATEGY_VERSION = ProblemGroupingStrategyVersion("1")


def _source_facts(
    *,
    tenant_id: str = _TENANT,
    task_id,
    run_id,
    reconstruction: ExecutionReconstruction | None = None,
    problem_signals: tuple[PlatformProblemSignal, ...] = (),
) -> ProblemGroupingFeatureSourceFacts:
    return ProblemGroupingFeatureSourceFacts(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        reconstruction=reconstruction,
        problem_signals=problem_signals,
    )


def _assessment_input(
    assessment: DiagnosticAssessment,
    *,
    source_facts: ProblemGroupingFeatureSourceFacts | None = None,
) -> ProblemGroupingAssessmentInput:
    return ProblemGroupingAssessmentInput(
        assessment=assessment,
        feature_source_facts=source_facts,
    )


def _finding(*, supporting_event_id=None) -> DiagnosticFinding:
    return DiagnosticFinding(
        kind=DiagnosticFindingKind.EVENT_AFTER_TERMINAL,
        scope=LifecycleAnomalyScope.EXECUTION,
        attempt_id=None,
        certainty=DiagnosticCertainty.PROVEN,
        claim="A lifecycle event was recorded after canonical run closure.",
        source_anomaly_kind=LifecycleAnomalyKind.EVENT_AFTER_TERMINAL,
        supporting_event_ids=((supporting_event_id or mint_event_id()),),
        supporting_evidence_ids=(),
        supporting_positions=(),
    )


def _runtime_event(
    *,
    tenant_id: str = _TENANT,
    task_id,
    run_id,
    attempt_id,
    event_type: RuntimeEventType,
    event_id=None,
    agent_id: str | None = None,
    node_id: str | None = None,
    step_id: str | None = None,
) -> RuntimeEvent:
    return sample_runtime_event(
        event_id=event_id or mint_event_id(),
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    ).model_copy(
        update={
            "event_type": event_type,
            "phase": ExecutionPhase.STEP_EXECUTION,
            "event_category": EventCategory.TOOL,
            "agent_id": agent_id,
            "node_id": node_id,
            "step_id": step_id,
        }
    )


def _positioned(event: RuntimeEvent, position: int) -> PositionedRuntimeEvent:
    return PositionedRuntimeEvent(event=event, position=ExecutionEventPosition(position))


def _causal_evidence(
    *,
    tenant_id: str = _TENANT,
    task_id,
    run_id,
    attempt_id,
    provider: str = _PROVIDER,
) -> PlatformCausalEvidence:
    return PlatformCausalEvidence(
        evidence_id=mint_event_id(),
        relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
        tenant_id=tenant_id,
        source=MessageBusTaskRef(
            provider=provider,
            task_id="transport-task-1",
            tenant_id=tenant_id,
        ),
        target=RuntimeExecutionRef(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            tenant_id=tenant_id,
        ),
        recorded_at=datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc),
    )


def _reconstruction(
    *,
    tenant_id: str = _TENANT,
    task_id,
    run_id,
    positioned_events: tuple[PositionedRuntimeEvent, ...],
    causal_evidence: tuple[PlatformCausalEvidence, ...],
) -> ExecutionReconstruction:
    return ExecutionReconstruction(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        causal_evidence=causal_evidence,
        positioned_events=positioned_events,
        attempts=(),
        runtime_history_completeness=RuntimeHistoryCompleteness.COMPLETE,
    )


class _FeatureInspectingStrategy:
    """Test-only strategy that requires and inspects projected features."""

    def __init__(self) -> None:
        self.inspected_inputs: tuple[ProblemGroupingInput, ...] = ()

    @property
    def strategy_id(self) -> ProblemGroupingStrategyId:
        return _FAKE_STRATEGY_ID

    @property
    def strategy_version(self) -> ProblemGroupingStrategyVersion:
        return _FAKE_STRATEGY_VERSION

    @property
    def characteristics(self) -> ProblemGroupingStrategyCharacteristics:
        return ProblemGroupingStrategyCharacteristics(
            method=ProblemGroupingMethod.SEMANTIC,
            deterministic=False,
            requires_features=True,
        )

    def group(
        self,
        inputs: tuple[ProblemGroupingInput, ...],
    ) -> ProblemGroupingStrategyResult:
        self.inspected_inputs = inputs
        members = tuple(input_item.subject.ref for input_item in inputs)
        return ProblemGroupingStrategyResult(
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            candidates=(
                ProblemGroupingCandidate(
                    members=members,
                    provenance=ProblemGroupingProvenance(
                        strategy_id=self.strategy_id,
                        strategy_version=self.strategy_version,
                        method=ProblemGroupingMethod.SEMANTIC,
                        supporting_subject_refs=members,
                        basis=None,
                    ),
                ),
            )
            if len(members) >= 2
            else (),
        )


def _engine_with_inspector() -> tuple[ProblemGroupingEngine, _FeatureInspectingStrategy]:
    strategy = _FeatureInspectingStrategy()
    registry = ProblemGroupingStrategyRegistry()
    registry.register(strategy)
    engine = ProblemGroupingEngine(
        registry,
        feature_projector=DiagnosticProblemGroupingFeatureProjector(),
    )
    return engine, strategy


def test_e2e_engine_projects_all_feature_categories() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    tool_failed_id = mint_event_id()
    retry_id = mint_event_id()
    signal_event_id = mint_event_id()

    tool_failed = _runtime_event(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_type=RuntimeEventType.TOOL_FAILED,
        event_id=tool_failed_id,
        agent_id="agent-1",
        node_id="node-1",
        step_id="step-1",
    )
    retry_event = _runtime_event(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_type=RuntimeEventType.RETRY_SCHEDULED,
        event_id=retry_id,
    )
    informational = _runtime_event(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_type=RuntimeEventType.STEP_STARTED,
    )
    causal = _causal_evidence(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        provider="celery",
    )
    reconstruction = _reconstruction(
        task_id=task_id,
        run_id=run_id,
        positioned_events=(
            _positioned(tool_failed, 1),
            _positioned(retry_event, 2),
            _positioned(informational, 3),
        ),
        causal_evidence=(causal,),
    )
    signal = PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_TOOL_FAILURE,
        severity=PROBLEM_SEVERITY_ERROR,
        source_layer=PROBLEM_SOURCE_LAYER_TOOL,
        source_component="invoke_tool",
        error_code="TOOL_TIMEOUT",
        safe_message="Tool invocation timed out after 30s.",
        run_id=str(run_id),
        task_id=str(task_id),
        event_id=str(signal_event_id),
        tool_id="search_docs",
        capability="knowledge.read",
        application_attributes=ApplicationObservabilityAttributes(
            namespace="local_workspace",
            operation="search",
        ),
    )
    assessment = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        findings=(_finding(supporting_event_id=tool_failed_id),),
        limitations=(),
    )
    source_facts = _source_facts(
        task_id=task_id,
        run_id=run_id,
        reconstruction=reconstruction,
        problem_signals=(signal,),
    )
    assessment_b = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        findings=(_finding(),),
        limitations=(),
    )

    engine, strategy = _engine_with_inspector()
    engine.group(
        (
            _assessment_input(assessment, source_facts=source_facts),
            _assessment_input(assessment_b),
        ),
        strategy_id=_FAKE_STRATEGY_ID,
    )

    features = strategy.inspected_inputs[0].features
    assert features is not None
    assert features.execution_context
    assert features.component_context
    assert features.operation_context
    assert features.integration_context
    assert features.failure_context
    assert features.causal_context
    assert any(
        item.source_kind is ProblemGroupingTextEvidenceSourceKind.PROBLEM_SIGNAL_SAFE_MESSAGE
        for item in features.text_evidence
    )
    assert features.execution_context[0].event_type is RuntimeEventType.TOOL_FAILED
    assert features.execution_context[1].event_type is RuntimeEventType.RETRY_SCHEDULED
    assert features.causal_context[0].relation_kind is CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION
    assert features.integration_context[0].provider == "celery"
    assert len(features.execution_context) == 2
    assert informational.event_id not in {item.supporting_event_ids[0] for item in features.execution_context}


def test_no_source_facts_yields_base_features_only() -> None:
    assessment = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        findings=(_finding(),),
        limitations=(),
    )
    subject = normalize_assessment(assessment)
    projector = DiagnosticProblemGroupingFeatureProjector()
    features = projector.project(assessment, subject)

    assert features.text_evidence
    assert features.execution_context == ()
    assert features.component_context == ()
    assert features.operation_context == ()
    assert features.integration_context == ()
    assert features.failure_context == ()
    assert features.causal_context == ()


def test_reconstruction_scope_mismatch_fails_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    assessment = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        findings=(),
        limitations=(),
    )
    other_run = mint_run_id()
    reconstruction = _reconstruction(
        task_id=task_id,
        run_id=other_run,
        positioned_events=(),
        causal_evidence=(),
    )
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    engine = ProblemGroupingEngine(
        registry,
        feature_projector=DiagnosticProblemGroupingFeatureProjector(),
    )

    with pytest.raises(ProblemGroupingIntegrityError, match="reconstruction run_id"):
        engine.group(
            (
                _assessment_input(
                    assessment,
                    source_facts=_source_facts(
                        task_id=task_id,
                        run_id=run_id,
                        reconstruction=reconstruction,
                    ),
                ),
            ),
            strategy_id=STRATEGY_ID,
        )


def test_problem_signal_task_run_mismatch_fails_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    assessment = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        findings=(),
        limitations=(),
    )
    signal = PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_TOOL_FAILURE,
        severity=PROBLEM_SEVERITY_ERROR,
        task_id=str(mint_task_id()),
        run_id=str(run_id),
    )
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    engine = ProblemGroupingEngine(
        registry,
        feature_projector=DiagnosticProblemGroupingFeatureProjector(),
    )

    with pytest.raises(ProblemGroupingIntegrityError, match="problem signal.*source_facts task_id"):
        engine.group(
            (
                _assessment_input(
                    assessment,
                    source_facts=_source_facts(
                        task_id=task_id,
                        run_id=run_id,
                        problem_signals=(signal,),
                    ),
                ),
            ),
            strategy_id=STRATEGY_ID,
        )


def test_projector_source_has_no_payload_access() -> None:
    source = Path(
        "intergrax/runtime/diagnostics/diagnostic_problem_grouping_feature_projector.py"
    ).read_text(encoding="utf-8")
    assert ".payload" not in source
    assert '["payload"]' not in source
    assert 'get("payload")' not in source


def test_causal_provider_maps_integration_and_causal_features() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    causal = _causal_evidence(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        provider="rabbitmq",
    )
    reconstruction = _reconstruction(
        task_id=task_id,
        run_id=run_id,
        positioned_events=(),
        causal_evidence=(causal,),
    )
    assessment = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        findings=(),
        limitations=(),
    )
    projector = DiagnosticProblemGroupingFeatureProjector()
    features = projector.project(
        assessment,
        normalize_assessment(assessment),
        source_facts=_source_facts(
            task_id=task_id,
            run_id=run_id,
            reconstruction=reconstruction,
        ),
    )

    assert features.causal_context[0].source_provider == "rabbitmq"
    assert features.integration_context[0].provider == "rabbitmq"


def test_runtime_event_type_and_causal_relation_remain_typed_enums() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    event = _runtime_event(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_type=RuntimeEventType.TOOL_FAILED,
    )
    causal = _causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    reconstruction = _reconstruction(
        task_id=task_id,
        run_id=run_id,
        positioned_events=(_positioned(event, 1),),
        causal_evidence=(causal,),
    )
    assessment = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        findings=(),
        limitations=(),
    )
    features = DiagnosticProblemGroupingFeatureProjector().project(
        assessment,
        normalize_assessment(assessment),
        source_facts=_source_facts(
            task_id=task_id,
            run_id=run_id,
            reconstruction=reconstruction,
        ),
    )

    assert isinstance(features.execution_context[0].event_type, RuntimeEventType)
    assert isinstance(features.causal_context[0].relation_kind, CausalRelationKind)


def test_problem_signal_safe_message_is_bounded_and_source_typed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    signal = PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_TOOL_FAILURE,
        severity=PROBLEM_SEVERITY_ERROR,
        safe_message="x" * (MAX_TEXT_EVIDENCE_CHARS + 1),
        run_id=str(run_id),
        task_id=str(task_id),
    )
    assessment = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        findings=(),
        limitations=(),
    )
    with pytest.raises(ProblemGroupingFeatureIntegrityError):
        DiagnosticProblemGroupingFeatureProjector().project(
            assessment,
            normalize_assessment(assessment),
            source_facts=_source_facts(
                task_id=task_id,
                run_id=run_id,
                problem_signals=(signal,),
            ),
        )


def test_empty_optional_problem_signal_fields_do_not_create_fake_values() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    signal = PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_TOOL_FAILURE,
        severity=PROBLEM_SEVERITY_ERROR,
        source_layer=PROBLEM_SOURCE_LAYER_TOOL,
        source_component="",
        run_id=str(run_id),
        task_id=str(task_id),
    )
    assessment = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        findings=(),
        limitations=(),
    )
    features = DiagnosticProblemGroupingFeatureProjector().project(
        assessment,
        normalize_assessment(assessment),
        source_facts=_source_facts(
            task_id=task_id,
            run_id=run_id,
            problem_signals=(signal,),
        ),
    )

    assert features.component_context == ()
    assert features.failure_context
    assert features.operation_context == ()


def test_invalid_problem_signal_event_id_fails_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    signal = PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_TOOL_FAILURE,
        severity=PROBLEM_SEVERITY_ERROR,
        source_layer=PROBLEM_SOURCE_LAYER_TOOL,
        source_component="invoke_tool",
        event_id="not-a-valid-event-id",
        run_id=str(run_id),
        task_id=str(task_id),
    )
    assessment = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        findings=(),
        limitations=(),
    )
    with pytest.raises(ProblemGroupingFeatureIntegrityError, match="event_id"):
        DiagnosticProblemGroupingFeatureProjector().project(
            assessment,
            normalize_assessment(assessment),
            source_facts=_source_facts(
                task_id=task_id,
                run_id=run_id,
                problem_signals=(signal,),
            ),
        )


def test_deterministic_grouping_unchanged_with_or_without_source_facts() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    finding = _finding()
    assessment_a = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        findings=(finding,),
        limitations=(),
    )
    assessment_b = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=mint_run_id(),
        findings=(finding,),
        limitations=(),
    )
    event = _runtime_event(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_type=RuntimeEventType.TOOL_FAILED,
    )
    reconstruction = _reconstruction(
        task_id=task_id,
        run_id=run_id,
        positioned_events=(_positioned(event, 1),),
        causal_evidence=(_causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_id),),
    )
    source_facts = _source_facts(
        task_id=task_id,
        run_id=run_id,
        reconstruction=reconstruction,
        problem_signals=(
            PlatformProblemSignal(
                problem_kind=PROBLEM_KIND_PLATFORM_TOOL_FAILURE,
                severity=PROBLEM_SEVERITY_ERROR,
                source_layer=PROBLEM_SOURCE_LAYER_TOOL,
                source_component="invoke_tool",
                run_id=str(run_id),
                task_id=str(task_id),
            ),
        ),
    )
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    engine = ProblemGroupingEngine(
        registry,
        feature_projector=DiagnosticProblemGroupingFeatureProjector(),
    )

    without_facts = engine.group(
        (
            _assessment_input(assessment_a),
            _assessment_input(assessment_b),
        ),
        strategy_id=STRATEGY_ID,
    )
    with_facts = engine.group(
        (
            _assessment_input(assessment_a, source_facts=source_facts),
            _assessment_input(assessment_b),
        ),
        strategy_id=STRATEGY_ID,
    )

    assert without_facts == with_facts


def test_grouping_input_has_no_raw_source_fact_fields() -> None:
    raw_type_names = {
        "ExecutionReconstruction",
        "RuntimeEvent",
        "PlatformProblemSignal",
        "PlatformCausalEvidence",
    }
    for field in fields(ProblemGroupingInput):
        assert field.name not in {"reconstruction", "problem_signals", "source_facts"}
        annotation = str(field.type)
        for raw_name in raw_type_names:
            assert raw_name not in annotation


def test_event_selection_includes_failure_retry_and_referenced_events() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    referenced_id = mint_event_id()
    failure = _runtime_event(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_type=RuntimeEventType.TOOL_FAILED,
    )
    retry = _runtime_event(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_type=RuntimeEventType.RETRY_STARTED,
    )
    referenced = _runtime_event(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_type=RuntimeEventType.STEP_STARTED,
        event_id=referenced_id,
    )
    noise = _runtime_event(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_type=RuntimeEventType.STEP_STARTED,
    )
    assessment = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        findings=(_finding(supporting_event_id=referenced_id),),
        limitations=(),
    )
    selected = select_positioned_events_for_grouping(
        (
            _positioned(failure, 1),
            _positioned(retry, 2),
            _positioned(referenced, 3),
            _positioned(noise, 4),
        ),
        assessment,
    )

    assert [row.event.event_id for row in selected] == [
        failure.event_id,
        retry.event_id,
        referenced.event_id,
    ]


def _deterministic_engine() -> ProblemGroupingEngine:
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    return ProblemGroupingEngine(
        registry,
        feature_projector=DiagnosticProblemGroupingFeatureProjector(),
    )


def test_bundle_scope_matching_assessment_is_valid() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    assessment = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        findings=(_finding(),),
        limitations=(),
    )
    signal = PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_TOOL_FAILURE,
        severity=PROBLEM_SEVERITY_ERROR,
        source_layer=PROBLEM_SOURCE_LAYER_TOOL,
        source_component="invoke_tool",
    )
    source_facts = _source_facts(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        problem_signals=(signal,),
    )
    engine, strategy = _engine_with_inspector()
    engine.group(
        (_assessment_input(assessment, source_facts=source_facts),),
        strategy_id=_FAKE_STRATEGY_ID,
    )
    assert strategy.inspected_inputs[0].features is not None


def test_wrong_bundle_tenant_fails_closed_before_strategy() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    assessment = DiagnosticAssessment(
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=run_id,
        findings=(),
        limitations=(),
    )
    signal = PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_TOOL_FAILURE,
        severity=PROBLEM_SEVERITY_ERROR,
        source_layer=PROBLEM_SOURCE_LAYER_TOOL,
        source_component="invoke_tool",
    )
    source_facts = _source_facts(
        tenant_id="tenant-b",
        task_id=task_id,
        run_id=run_id,
        problem_signals=(signal,),
    )
    engine, strategy = _engine_with_inspector()
    with pytest.raises(ProblemGroupingIntegrityError, match="source_facts tenant_id"):
        engine.group(
            (_assessment_input(assessment, source_facts=source_facts),),
            strategy_id=_FAKE_STRATEGY_ID,
        )
    assert strategy.inspected_inputs == ()


def test_two_tenant_source_bundle_contamination_fails_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    assessment = DiagnosticAssessment(
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=run_id,
        findings=(_finding(),),
        limitations=(),
    )
    signal = PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_TOOL_FAILURE,
        severity=PROBLEM_SEVERITY_ERROR,
        source_layer=PROBLEM_SOURCE_LAYER_TOOL,
        source_component="invoke_tool",
        error_code="TOOL_TIMEOUT",
        safe_message="Plausible cross-tenant signal payload.",
    )
    source_facts = _source_facts(
        tenant_id="tenant-b",
        task_id=task_id,
        run_id=run_id,
        problem_signals=(signal,),
    )
    engine, strategy = _engine_with_inspector()
    with pytest.raises(ProblemGroupingIntegrityError, match="source_facts tenant_id"):
        engine.group(
            (_assessment_input(assessment, source_facts=source_facts),),
            strategy_id=_FAKE_STRATEGY_ID,
        )
    assert strategy.inspected_inputs == ()


def test_wrong_bundle_task_fails_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    other_task = mint_task_id()
    assessment = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        findings=(),
        limitations=(),
    )
    engine = _deterministic_engine()
    with pytest.raises(ProblemGroupingIntegrityError, match="source_facts task_id"):
        engine.group(
            (
                _assessment_input(
                    assessment,
                    source_facts=_source_facts(task_id=other_task, run_id=run_id),
                ),
            ),
            strategy_id=STRATEGY_ID,
        )


def test_wrong_bundle_run_fails_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    other_run = mint_run_id()
    assessment = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        findings=(),
        limitations=(),
    )
    engine = _deterministic_engine()
    with pytest.raises(ProblemGroupingIntegrityError, match="source_facts run_id"):
        engine.group(
            (
                _assessment_input(
                    assessment,
                    source_facts=_source_facts(task_id=task_id, run_id=other_run),
                ),
            ),
            strategy_id=STRATEGY_ID,
        )


def test_problem_signal_run_id_mismatch_fails_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    assessment = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        findings=(),
        limitations=(),
    )
    signal = PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_TOOL_FAILURE,
        severity=PROBLEM_SEVERITY_ERROR,
        task_id=str(task_id),
        run_id=str(mint_run_id()),
    )
    engine = _deterministic_engine()
    with pytest.raises(ProblemGroupingIntegrityError, match="problem signal.*source_facts run_id"):
        engine.group(
            (
                _assessment_input(
                    assessment,
                    source_facts=_source_facts(
                        task_id=task_id,
                        run_id=run_id,
                        problem_signals=(signal,),
                    ),
                ),
            ),
            strategy_id=STRATEGY_ID,
        )


def test_problem_signal_empty_task_run_inherits_bundle_scope() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    signal = PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_TOOL_FAILURE,
        severity=PROBLEM_SEVERITY_ERROR,
        source_layer=PROBLEM_SOURCE_LAYER_TOOL,
        source_component="invoke_tool",
    )
    assessment = DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        findings=(),
        limitations=(),
    )
    features = DiagnosticProblemGroupingFeatureProjector().project(
        assessment,
        normalize_assessment(assessment),
        source_facts=_source_facts(
            task_id=task_id,
            run_id=run_id,
            problem_signals=(signal,),
        ),
    )
    assert features.failure_context
    assert features.component_context


def test_source_facts_empty_tenant_id_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="tenant_id is required"):
        ProblemGroupingFeatureSourceFacts(
            tenant_id="",
            task_id=mint_task_id(),
            run_id=mint_run_id(),
        )
