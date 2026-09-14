# © Artur Czarnecki. All rights reserved.

"""ERL-DIAG-001C — reliability grouping and occurrence identity tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from intergrax.contracts.enterprise_reliability.case_lifecycle import (
    ReliabilityCaseLifecycleState,
)
from intergrax.contracts.enterprise_reliability.diagnostics import (
    AutomationSafetyHint,
    ExternalEffectReliabilityObservation,
    ExternalEffectReliabilityProblemGroupingStrategy,
    ExternalEffectReliabilitySignalKind,
    ReliabilityCaseSubjectRef,
    ReliabilityDiagnosticArtifactRefs,
    ReliabilityDiagnosticCorrelation,
    parse_reliability_diagnostic_occurrence_instance_id,
    reliability_case_subject_index_token,
    reliability_diagnostic_occurrence_instance_id,
)
from intergrax.runtime.diagnostics.diagnostic_subject import ApplicationDiagnosticSubjectRef
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingCandidate,
    ProblemGroupingInput,
    ProblemGroupingMethod,
    ProblemGroupingProvenance,
    ProblemGroupingStrategyCharacteristics,
    ProblemGroupingStrategyRegistry,
    ProblemGroupingStrategyResult,
    ProblemGroupingStrategyVersion,
    normalize_signal_assessment,
)
from intergrax.runtime.diagnostics.reliability.observation_to_problem_signal import (
    ERL_RELIABILITY_DIAGNOSTIC_SUBJECT_APPLICATION_ID,
    map_handoff_to_platform_problem_signal,
)
from intergrax.runtime.diagnostics.reliability.reliability_case_default_grouping_strategy import (
    ReliabilityCaseDefaultObservationGroupingStrategy,
    STRATEGY_ID,
    _parse_erl_reliability_grouping_subject,
)
from intergrax.runtime.diagnostics.reliability.reliability_case_grouping_reconciliation import (
    ReliabilityCaseProblemGroupingBasis,
)
from intergrax.runtime.diagnostics.reliability.reliability_diagnostic_bridge import (
    build_reliability_diagnostic_emitter,
)
from intergrax.runtime.diagnostics.reliability.reliability_diagnostic_handoff import (
    map_observation_to_handoff,
)
from intergrax.runtime.diagnostics.signal_diagnostic_assessment import (
    SignalDiagnosticAssessmentBuilder,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    build_diagnostic_orchestrator_stack_for_tests,
    query_all_occurrences_for_problem,
    query_all_problems_for_tenant,
)

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-erl-001c-a"
_TENANT_B = "tenant-erl-001c-b"
_RECORDED_AT = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)


def _correlation(
    *,
    tenant_id: str = _TENANT_A,
    reliability_case_id: str = "case-001",
    correlation_id: str = "corr-001",
) -> ReliabilityDiagnosticCorrelation:
    return ReliabilityDiagnosticCorrelation(
        tenant_id=tenant_id,
        correlation_id=correlation_id,
        reliability_case_id=reliability_case_id,
        external_effect_contract_id="contract-1",
    )


def _observation(
    *,
    tenant_id: str = _TENANT_A,
    observation_id: str = "obs-1",
    reliability_case_id: str = "case-001",
    signal_kind: ExternalEffectReliabilitySignalKind = (
        ExternalEffectReliabilitySignalKind.UNCERTAINTY_ADMITTED
    ),
    correlation_id: str = "corr-001",
    artifact_refs: ReliabilityDiagnosticArtifactRefs | None = None,
) -> ExternalEffectReliabilityObservation:
    return ExternalEffectReliabilityObservation(
        observation_id=observation_id,
        tenant_id=tenant_id,
        signal_kind=signal_kind,
        recorded_at=_RECORDED_AT,
        reliability_case_id=reliability_case_id,
        correlation=_correlation(
            tenant_id=tenant_id,
            reliability_case_id=reliability_case_id,
            correlation_id=correlation_id,
        ),
        lifecycle_state=ReliabilityCaseLifecycleState.UNKNOWN_DETECTED,
        artifact_refs=artifact_refs or ReliabilityDiagnosticArtifactRefs(),
        execution_safety_hint=AutomationSafetyHint.UNKNOWN,
        trace_refs=(),
    )


def _grouping_input_for(observation: ExternalEffectReliabilityObservation) -> ProblemGroupingInput:
    handoff = map_observation_to_handoff(observation)
    signal = map_handoff_to_platform_problem_signal(handoff)
    subject_ref = ApplicationDiagnosticSubjectRef(
        tenant_id=handoff.tenant_id,
        application_id=ERL_RELIABILITY_DIAGNOSTIC_SUBJECT_APPLICATION_ID,
        instance_id=reliability_diagnostic_occurrence_instance_id(
            reliability_case_id=handoff.reliability_case_id,
            observation_id=handoff.observation_id,
        ),
    )
    assessment = SignalDiagnosticAssessmentBuilder().assess(subject_ref, (signal,))
    return ProblemGroupingInput(subject=normalize_signal_assessment(assessment))


def test_default_strategy_subject_token_is_deterministic() -> None:
    strategy = ReliabilityCaseDefaultObservationGroupingStrategy()
    obs = _observation(reliability_case_id="case-x")
    subject = strategy.group(obs)
    assert isinstance(subject, ReliabilityCaseSubjectRef)
    assert subject.tenant_id == _TENANT_A
    assert subject.reliability_case_id == "case-x"
    assert subject.index_token == reliability_case_subject_index_token("case-x")


def test_occurrence_instance_id_encodes_observation_separately_from_grouping() -> None:
    instance_id = reliability_diagnostic_occurrence_instance_id(
        reliability_case_id="case-123",
        observation_id="obs-abc",
    )
    case_id, obs_id = parse_reliability_diagnostic_occurrence_instance_id(instance_id)
    assert case_id == "case-123"
    assert obs_id == "obs-abc"


def test_case1_same_case_same_observation_replay() -> None:
    orchestrator, persistence, _, _ = build_diagnostic_orchestrator_stack_for_tests()
    emitter = build_reliability_diagnostic_emitter(orchestrator)
    observation = _observation()
    emitter.emit(observation)
    emitter.emit(observation)

    problems = query_all_problems_for_tenant(persistence, _TENANT_A)
    assert len(problems) == 1
    assert problems[0].occurrence_count == 1


def test_case2_same_case_different_observations() -> None:
    orchestrator, persistence, _, occurrence_persistence = (
        build_diagnostic_orchestrator_stack_for_tests()
    )
    emitter = build_reliability_diagnostic_emitter(orchestrator)
    emitter.emit(
        _observation(
            observation_id="obs-a",
            signal_kind=ExternalEffectReliabilitySignalKind.UNCERTAINTY_ADMITTED,
        ),
    )
    emitter.emit(
        _observation(
            observation_id="obs-b",
            signal_kind=ExternalEffectReliabilitySignalKind.RECONCILIATION_ATTEMPTED,
            artifact_refs=ReliabilityDiagnosticArtifactRefs(evidence_ref="evidence-b"),
        ),
    )

    problems = query_all_problems_for_tenant(persistence, _TENANT_A)
    assert len(problems) == 1
    assert problems[0].occurrence_count == 2
    occurrences = query_all_occurrences_for_problem(
        occurrence_persistence,
        tenant_id=_TENANT_A,
        problem_id=problems[0].problem_id,
    )
    assert len(occurrences) == 2


def test_case3_different_cases_yield_different_problems() -> None:
    orchestrator, persistence, _, _ = build_diagnostic_orchestrator_stack_for_tests()
    emitter = build_reliability_diagnostic_emitter(orchestrator)
    emitter.emit(_observation(reliability_case_id="case-a", observation_id="obs-1"))
    emitter.emit(_observation(reliability_case_id="case-b", observation_id="obs-2"))

    problems = query_all_problems_for_tenant(persistence, _TENANT_A)
    assert len(problems) == 2


def test_case5_multi_tenant_isolation() -> None:
    orchestrator, persistence, _, _ = build_diagnostic_orchestrator_stack_for_tests()
    emitter = build_reliability_diagnostic_emitter(orchestrator)
    emitter.emit(
        _observation(
            tenant_id=_TENANT_A,
            reliability_case_id="shared-case",
            observation_id="obs-a",
        ),
    )
    emitter.emit(
        _observation(
            tenant_id=_TENANT_B,
            reliability_case_id="shared-case",
            observation_id="obs-b",
        ),
    )

    assert len(query_all_problems_for_tenant(persistence, _TENANT_A)) == 1
    assert len(query_all_problems_for_tenant(persistence, _TENANT_B)) == 1


class _CompositeCorrelationBatchGroupingStrategy:
    """Test-only plugin: one Problem per correlation_id across distinct case ids."""

    @property
    def strategy_id(self):
        return STRATEGY_ID

    @property
    def strategy_version(self) -> ProblemGroupingStrategyVersion:
        return ProblemGroupingStrategyVersion("test-composite-1")

    @property
    def characteristics(self) -> ProblemGroupingStrategyCharacteristics:
        return ProblemGroupingStrategyCharacteristics(
            method=ProblemGroupingMethod.DETERMINISTIC,
            deterministic=True,
        )

    def group(
        self,
        inputs: tuple[ProblemGroupingInput, ...],
    ) -> ProblemGroupingStrategyResult:
        all_members: list = []
        for input_item in inputs:
            parsed = _parse_erl_reliability_grouping_subject(input_item.subject)
            if parsed is None:
                continue
            all_members.append(parsed[1])

        members = tuple(all_members)
        candidate = ProblemGroupingCandidate(
            members=members,
            provenance=ProblemGroupingProvenance(
                strategy_id=self.strategy_id,
                strategy_version=self.strategy_version,
                method=ProblemGroupingMethod.DETERMINISTIC,
                supporting_subject_refs=members,
                basis=ReliabilityCaseProblemGroupingBasis(
                    reliability_case_id="composite:shared-corr",
                ),
            ),
        )
        return ProblemGroupingStrategyResult(
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            candidates=(candidate,),
        )


def test_case4_custom_batch_strategy_plugin_groups_two_cases() -> None:
    registry = ProblemGroupingStrategyRegistry()
    registry.register(_CompositeCorrelationBatchGroupingStrategy())
    strategy = registry.resolve(STRATEGY_ID)
    obs_case_1 = _observation(
        reliability_case_id="case-one",
        correlation_id="shared-corr",
        observation_id="obs-1",
    )
    obs_case_2 = _observation(
        reliability_case_id="case-two",
        correlation_id="shared-corr",
        observation_id="obs-2",
    )
    result = strategy.group(
        (_grouping_input_for(obs_case_1), _grouping_input_for(obs_case_2)),
    )
    assert len(result.candidates) == 1
    assert len(result.candidates[0].members) == 2


def test_public_grouping_spi_importable_without_runtime() -> None:
    import importlib

    mod = importlib.import_module(
        "intergrax.contracts.enterprise_reliability.diagnostics.grouping",
    )
    assert hasattr(mod, "ExternalEffectReliabilityProblemGroupingStrategy")


def test_custom_observation_spi_implementation() -> None:
    @dataclass(frozen=True, slots=True)
    class _ByCorrelationSubject:
        tenant_id: str
        reliability_case_id: str

        @property
        def index_token(self) -> str:
            return reliability_case_subject_index_token(self.reliability_case_id)

    class _ByCorrelationObservationGrouping:
        def group(
            self,
            observation: ExternalEffectReliabilityObservation,
        ) -> ReliabilityCaseSubjectRef:
            return _ByCorrelationSubject(
                tenant_id=observation.tenant_id,
                reliability_case_id=f"corr:{observation.correlation.correlation_id}",
            )

        @property
        def strategy_id(self):
            from intergrax.contracts.enterprise_reliability.diagnostics import (
                RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_ID,
            )

            return RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_ID

        @property
        def strategy_version(self):
            from intergrax.contracts.enterprise_reliability.diagnostics import (
                RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_VERSION,
            )

            return RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_VERSION

    impl: ExternalEffectReliabilityProblemGroupingStrategy = _ByCorrelationObservationGrouping()
    subject = impl.group(_observation(correlation_id="corr-plugin"))
    assert subject.index_token == reliability_case_subject_index_token("corr:corr-plugin")
