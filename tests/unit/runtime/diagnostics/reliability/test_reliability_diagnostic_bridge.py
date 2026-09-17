# © Artur Czarnecki. All rights reserved.

"""ERL-DIAG-001B — reliability diagnostic runtime bridge tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.contracts.enterprise_reliability.case_lifecycle import (
    ReliabilityCaseLifecycleState,
)
from intergrax.contracts.enterprise_reliability.diagnostics import (
    AutomationSafetyHint,
    ExternalEffectReliabilityObservation,
    ExternalEffectReliabilitySignalKind,
    NullExternalEffectReliabilityDiagnosticEmitter,
    ReliabilityDiagnosticArtifactRefs,
    ReliabilityDiagnosticCorrelation,
)
from intergrax.runtime.diagnostics.reliability.reliability_case_default_grouping_strategy import (
    ReliabilityCaseDefaultObservationGroupingStrategy,
    STRATEGY_ID as ERL_RELIABILITY_GROUPING_STRATEGY_ID,
)
from intergrax.contracts.enterprise_reliability.diagnostics.grouping import (
    reliability_diagnostic_occurrence_instance_id,
)
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticOrchestrationRequest,
    DiagnosticOrchestrationResult,
)
from intergrax.runtime.diagnostics.reliability.observation_to_problem_signal import (
    ERL_RELIABILITY_DIAGNOSTIC_SUBJECT_APPLICATION_ID,
    map_handoff_to_platform_problem_signal,
)
from intergrax.runtime.diagnostics.reliability.reliability_diagnostic_bridge import (
    ReliabilityDiagnosticBridge,
    RuntimeExternalEffectReliabilityDiagnosticEmitter,
    build_reliability_diagnostic_emitter,
)
from intergrax.runtime.diagnostics.reliability.reliability_diagnostic_handoff import (
    map_observation_to_handoff,
)
from intergrax.runtime.diagnostics.reliability.reliability_observability_attributes import (
    ExternalEffectReliabilityObservabilityAttributes,
)
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_KIND_PLATFORM_EXTERNAL_EFFECT_RELIABILITY,
    PROBLEM_SEVERITY_ERROR,
)
from testing_support.runtime.diagnostics.problem_persistence_test_support import (
    build_diagnostic_orchestrator_stack_for_tests,
    query_all_problems_for_tenant,
)

pytestmark = pytest.mark.unit

_DEFAULT_OBSERVATION_GROUPING = ReliabilityCaseDefaultObservationGroupingStrategy()

_TENANT = "tenant-erl-bridge"
_RECORDED_AT = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def _correlation(**updates: object) -> ReliabilityDiagnosticCorrelation:
    base = ReliabilityDiagnosticCorrelation(
        tenant_id=_TENANT,
        correlation_id="corr-bridge-1",
        reliability_case_id="case-bridge-1",
        external_effect_contract_id="contract-ext-1",
    )
    if not updates:
        return base
    return base.model_copy(update=updates)


def _observation(
    *,
    observation_id: str = "obs-bridge-1",
    signal_kind: ExternalEffectReliabilitySignalKind = (
        ExternalEffectReliabilitySignalKind.UNCERTAINTY_ADMITTED
    ),
    artifact_refs: ReliabilityDiagnosticArtifactRefs | None = None,
    trace_refs: tuple[str, ...] = ("trace-ref-1",),
) -> ExternalEffectReliabilityObservation:
    return ExternalEffectReliabilityObservation(
        observation_id=observation_id,
        tenant_id=_TENANT,
        signal_kind=signal_kind,
        recorded_at=_RECORDED_AT,
        reliability_case_id="case-bridge-1",
        correlation=_correlation(),
        lifecycle_state=ReliabilityCaseLifecycleState.UNKNOWN_DETECTED,
        artifact_refs=artifact_refs or ReliabilityDiagnosticArtifactRefs(),
        execution_safety_hint=AutomationSafetyHint.UNKNOWN,
        trace_refs=trace_refs,
    )


@dataclass
class _RecordingOrchestration:
    calls: list[DiagnosticOrchestrationRequest] = field(default_factory=list)

    def run(self, request: DiagnosticOrchestrationRequest) -> DiagnosticOrchestrationResult:
        self.calls.append(request)
        from intergrax.runtime.diagnostics.problem_grouping import (
            ProblemGroupingMethod,
            ProblemGroupingResult,
            ProblemGroupingStrategyVersion,
        )
        from intergrax.runtime.diagnostics.problem_lifecycle import ProblemLifecycleResult

        return DiagnosticOrchestrationResult(
            tenant_id=request.tenant_id,
            execution_results=(),
            signal_subject_results=(),
            grouping_result=ProblemGroupingResult(
                tenant_id=request.tenant_id,
                strategy_id=request.grouping_strategy_id,
                strategy_version=ProblemGroupingStrategyVersion("0"),
                method=ProblemGroupingMethod.DETERMINISTIC,
                candidates=(),
                ungrouped_subjects=(),
            ),
            lifecycle_result=ProblemLifecycleResult(created=(), updated=(), unchanged=()),
        )


def test_valid_observation_maps_and_preserves_identity_fields() -> None:
    observation = _observation(
        signal_kind=ExternalEffectReliabilitySignalKind.TRUTH_UNAVAILABLE,
        artifact_refs=ReliabilityDiagnosticArtifactRefs(evidence_ref="evidence-1"),
    )
    handoff = map_observation_to_handoff(observation)
    signal = map_handoff_to_platform_problem_signal(handoff)

    assert signal.problem_kind == PROBLEM_KIND_PLATFORM_EXTERNAL_EFFECT_RELIABILITY
    assert signal.error_code == "external_effect_reliability.TRUTH_UNAVAILABLE"
    assert signal.error_code != observation.observation_id
    assert signal.event_id == observation.observation_id
    assert signal.severity == PROBLEM_SEVERITY_ERROR
    assert signal.correlation_id == observation.correlation.correlation_id
    assert signal.exception_type is None
    assert signal.application_attributes is not None
    attrs = signal.application_attributes
    assert isinstance(attrs, ExternalEffectReliabilityObservabilityAttributes)
    assert attrs.observation_id == observation.observation_id
    assert attrs.signal_kind == observation.signal_kind.value
    assert attrs.reliability_case_id == observation.reliability_case_id
    assert attrs.trace_refs == observation.trace_refs
    assert len(signal.artifact_refs) == 1
    assert signal.artifact_refs[0].artifact_ref == "evidence-1"


def test_same_fact_type_yields_stable_error_code_and_distinct_event_ids() -> None:
    kind = ExternalEffectReliabilitySignalKind.TRUTH_UNAVAILABLE
    obs_a = _observation(observation_id="obs-1", signal_kind=kind)
    obs_b = _observation(observation_id="obs-2", signal_kind=kind)
    signal_a = map_handoff_to_platform_problem_signal(map_observation_to_handoff(obs_a))
    signal_b = map_handoff_to_platform_problem_signal(map_observation_to_handoff(obs_b))

    assert signal_a.error_code == signal_b.error_code == "external_effect_reliability.TRUTH_UNAVAILABLE"
    assert signal_a.event_id == "obs-1"
    assert signal_b.event_id == "obs-2"
    assert signal_a.error_code != signal_a.event_id


def test_bridge_does_not_derive_severity_from_signal_kind() -> None:
    severe = map_handoff_to_platform_problem_signal(
        map_observation_to_handoff(
            _observation(signal_kind=ExternalEffectReliabilitySignalKind.TRUTH_UNAVAILABLE),
        ),
    )
    mild = map_handoff_to_platform_problem_signal(
        map_observation_to_handoff(
            _observation(signal_kind=ExternalEffectReliabilitySignalKind.UNCERTAINTY_ADMITTED),
        ),
    )
    assert severe.severity == mild.severity == PROBLEM_SEVERITY_ERROR


def test_bridge_invokes_orchestration_exactly_once() -> None:
    recording = _RecordingOrchestration()
    bridge = ReliabilityDiagnosticBridge(
        recording,
        grouping_strategy_id=ERL_RELIABILITY_GROUPING_STRATEGY_ID,
        observation_grouping=_DEFAULT_OBSERVATION_GROUPING,
    )
    bridge.on_observation(_observation())

    assert len(recording.calls) == 1
    request = recording.calls[0]
    assert request.tenant_id == _TENANT
    assert len(request.signal_subjects) == 1
    scope = request.signal_subjects[0]
    assert scope.application_id == ERL_RELIABILITY_DIAGNOSTIC_SUBJECT_APPLICATION_ID
    assert scope.instance_id == reliability_diagnostic_occurrence_instance_id(
        reliability_case_id="case-bridge-1",
        observation_id="obs-bridge-1",
    )
    assert len(scope.problem_signals) == 1


def test_bridge_failure_does_not_escape() -> None:
    class _FailingOrchestration:
        def run(self, request: DiagnosticOrchestrationRequest) -> DiagnosticOrchestrationResult:
            raise RuntimeError("orchestrator down")

    bridge = ReliabilityDiagnosticBridge(
        _FailingOrchestration(),
        grouping_strategy_id=ERL_RELIABILITY_GROUPING_STRATEGY_ID,
        observation_grouping=_DEFAULT_OBSERVATION_GROUPING,
    )
    bridge.on_observation(_observation())


def test_emitter_substitution_and_null_emitter() -> None:
    recording = _RecordingOrchestration()
    emitter = build_reliability_diagnostic_emitter(
        recording,
        grouping_strategy_id=ERL_RELIABILITY_GROUPING_STRATEGY_ID,
    )
    emitter.emit(_observation())
    assert len(recording.calls) == 1

    null_emitter = NullExternalEffectReliabilityDiagnosticEmitter()
    null_emitter.emit(_observation())
    assert len(recording.calls) == 1


def test_runtime_emitter_wraps_bridge() -> None:
    recording = _RecordingOrchestration()
    bridge = ReliabilityDiagnosticBridge(
        recording,
        grouping_strategy_id=ERL_RELIABILITY_GROUPING_STRATEGY_ID,
        observation_grouping=_DEFAULT_OBSERVATION_GROUPING,
    )
    RuntimeExternalEffectReliabilityDiagnosticEmitter(bridge).emit(_observation())
    assert len(recording.calls) == 1


def test_integration_orchestrator_creates_problem_without_persistence_bypass() -> None:
    orchestrator, persistence, _, _ = build_diagnostic_orchestrator_stack_for_tests()
    emitter = build_reliability_diagnostic_emitter(
        orchestrator,
        grouping_strategy_id=ERL_RELIABILITY_GROUPING_STRATEGY_ID,
    )
    emitter.emit(_observation())

    problems = query_all_problems_for_tenant(persistence, _TENANT)
    assert len(problems) == 1
    assert problems[0].occurrence_count == 1


def test_duplicate_observation_id_does_not_duplicate_occurrence() -> None:
    orchestrator, persistence, _, _ = build_diagnostic_orchestrator_stack_for_tests()
    emitter = build_reliability_diagnostic_emitter(
        orchestrator,
        grouping_strategy_id=ERL_RELIABILITY_GROUPING_STRATEGY_ID,
    )
    observation = _observation()
    emitter.emit(observation)
    emitter.emit(observation)
    later = _observation().model_copy(
        update={"recorded_at": _RECORDED_AT + timedelta(hours=1)},
    )
    emitter.emit(later)

    problems = query_all_problems_for_tenant(persistence, _TENANT)
    assert len(problems) == 1
    assert problems[0].occurrence_count == 1
