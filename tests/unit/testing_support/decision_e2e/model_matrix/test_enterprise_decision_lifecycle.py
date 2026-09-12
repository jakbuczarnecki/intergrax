# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle import (
    DecisionLifecycleActorRef,
    DecisionLifecycleEngine,
    DecisionLifecycleState,
    DecisionLifecycleTransitionRejectedError,
    DecisionSourceKind,
    DecisionSourceReference,
    DecisionStateTransitionProvider,
    DecisionType,
    DefaultDecisionStateTransitionProvider,
    RecordingDecisionAuditProvider,
    UtcDecisionClockProvider,
    UuidDecisionIdentityProvider,
    lifecycle_record_from_orchestration_result,
    source_references_from_orchestration_result,
)
from testing_support.decision_e2e.local_ai_incident_qualification import (
    QualificationCliExit,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationSessionState,
)
from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleEvent,
)
from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing import (
    DataSensitivityClass,
    GovernanceEvaluationEngine,
    GovernancePolicyRef,
    GovernanceRiskTier,
    GovernanceTaskContext,
    default_governance_policies,
)
from testing_support.decision_e2e.model_matrix.model_capability_baseline import (
    CapabilityDimensionId,
    CapabilityProfileBuildRequest,
    ObservationLevel,
    build_model_capability_profiles,
)
from testing_support.decision_e2e.model_matrix.model_qualification_outcome import (
    ModelQualificationOutcome,
)
from testing_support.decision_e2e.model_matrix.model_selection_recommendation import (
    CapabilitySelectionConstraints,
    ModelSelectionEngine,
    ModelSelectionRequest,
    TaskCapabilityRequirement,
    TaskRequirements,
    default_selection_strategies,
)
from testing_support.decision_e2e.model_matrix.production_decision_orchestration import (
    DecisionOrchestrationRequest,
    DecisionOrchestrator,
    EngineBackedGovernanceProvider,
    EngineBackedSelectionProvider,
    RecordingExecutionProvider,
)
from testing_support.decision_e2e.model_matrix.production_decision_orchestration.contracts import (
    DecisionOrchestrationOutcome,
)
from testing_support.decision_e2e.model_matrix.qualification_cohort_executor import (
    CohortExecutionStatus,
)
from testing_support.decision_e2e.model_matrix.registry import (
    qualification_matrix_version,
)

_ACTOR = DecisionLifecycleActorRef(
    actor_kind="test",
    reference_id="enterprise-decision-lifecycle",
)


def _outcome(profile_key: str) -> ModelQualificationOutcome:
    stamp = datetime(2026, 9, 12, 9, 0, 0, tzinfo=UTC)
    return ModelQualificationOutcome(
        profile_key=profile_key,
        provider="ollama",
        model_name=profile_key,
        matrix_version=qualification_matrix_version(),
        qualification_task_id="DS-E2E-15J-L1.R6",
        evaluated_at=stamp,
        status=CohortExecutionStatus.EXECUTED,
        exit_code=QualificationCliExit.SUCCESS,
        session_state=QualificationSessionState.FINALIZED,
    )


def _profiles(*profile_keys: str):
    built = build_model_capability_profiles(
        CapabilityProfileBuildRequest(
            matrix_version=qualification_matrix_version(),
            outcomes=tuple(_outcome(key) for key in profile_keys),
        )
    )
    return built.profiles


def _orchestration_request(profiles: tuple) -> DecisionOrchestrationRequest:
    return DecisionOrchestrationRequest(
        selection_request=ModelSelectionRequest(
            task_requirements=TaskRequirements(
                scenario_id="orchestration-scenario",
                capability_requirements=(
                    TaskCapabilityRequirement(
                        dimension_id=CapabilityDimensionId.QUALIFICATION_EXIT,
                        minimum_level=ObservationLevel.MODERATE,
                    ),
                ),
            ),
            capability_constraints=CapabilitySelectionConstraints(
                required_matrix_version=qualification_matrix_version(),
                excluded_profile_keys=(),
                require_behavioral_baseline=False,
            ),
            available_model_profiles=profiles,
        ),
        governance_task_context=GovernanceTaskContext(
            scenario_id="orchestration-scenario",
            data_sensitivity=DataSensitivityClass.PUBLIC,
            risk_tier=GovernanceRiskTier.LOW,
        ),
        applicable_policies=tuple(
            GovernancePolicyRef(
                policy_id=item.policy_id,
                policy_version=item.policy_version,
            )
            for item in default_governance_policies()
        ),
        capability_evidence=profiles,
    )


def _engine(
    audit: RecordingDecisionAuditProvider | None = None,
    *,
    transition: DecisionStateTransitionProvider | None = None,
) -> DecisionLifecycleEngine:
    return DecisionLifecycleEngine(
        transition_provider=transition or DefaultDecisionStateTransitionProvider(),
        audit_provider=audit or RecordingDecisionAuditProvider(),
        clock_provider=UtcDecisionClockProvider(),
        identity_provider=UuidDecisionIdentityProvider(),
    )


def test_decision_lifecycle_happy_path() -> None:
    audit = RecordingDecisionAuditProvider()
    engine = _engine(audit)
    record = engine.begin_decision(
        decision_type=DecisionType.PRODUCTION_MODEL_ROUTING,
        source_references=(
            DecisionSourceReference(
                source_kind=DecisionSourceKind.MODEL_SELECTION,
                reference_id="selection-1",
            ),
        ),
        actor=_ACTOR,
        reason="decision_opened",
    )
    assert record.lifecycle_state is DecisionLifecycleState.CREATED

    path = (
        DecisionLifecycleState.EVALUATING,
        DecisionLifecycleState.APPROVED,
        DecisionLifecycleState.EXECUTING,
        DecisionLifecycleState.COMPLETED,
    )
    for target in path:
        record = engine.transition(
            record,
            to_state=target,
            reason=f"advance_to_{target.value}",
            actor=_ACTOR,
        )
        assert record.lifecycle_state is target

    states = tuple(event.new_state for event in audit.events)
    assert states == (
        DecisionLifecycleState.CREATED,
        DecisionLifecycleState.EVALUATING,
        DecisionLifecycleState.APPROVED,
        DecisionLifecycleState.EXECUTING,
        DecisionLifecycleState.COMPLETED,
    )


def test_decision_lifecycle_rejects_invalid_transition() -> None:
    engine = _engine()
    record = engine.begin_decision(
        decision_type=DecisionType.PRODUCTION_MODEL_ROUTING,
        source_references=(),
        actor=_ACTOR,
        reason="decision_opened",
    )
    with pytest.raises(DecisionLifecycleTransitionRejectedError):
        engine.transition(
            record,
            to_state=DecisionLifecycleState.COMPLETED,
            reason="skip_states",
            actor=_ACTOR,
        )


@dataclass
class _CountingAuditProvider:
    recorded: list[DecisionLifecycleEvent] = field(default_factory=list)

    def record_lifecycle_event(self, event: DecisionLifecycleEvent) -> None:
        self.recorded.append(event)


def test_decision_lifecycle_works_with_swappable_audit_provider() -> None:
    counting = _CountingAuditProvider()
    engine = DecisionLifecycleEngine(
        transition_provider=DefaultDecisionStateTransitionProvider(),
        audit_provider=counting,
        clock_provider=UtcDecisionClockProvider(),
        identity_provider=UuidDecisionIdentityProvider(),
    )
    record = engine.begin_decision(
        decision_type=DecisionType.PRODUCTION_MODEL_ROUTING,
        source_references=(),
        actor=_ACTOR,
        reason="audit_plugin",
    )
    record = engine.transition(
        record,
        to_state=DecisionLifecycleState.EVALUATING,
        reason="evaluate",
        actor=_ACTOR,
    )
    assert record.lifecycle_state is DecisionLifecycleState.EVALUATING
    assert len(counting.recorded) == 2


class _EvaluatingGateTransitionProvider:
    """Alternate graph: only EVALUATING may follow CREATED; other steps use default."""

    def __init__(self) -> None:
        self._default = DefaultDecisionStateTransitionProvider()

    def assert_transition_allowed(
        self,
        *,
        from_state: DecisionLifecycleState,
        to_state: DecisionLifecycleState,
    ) -> None:
        if from_state is DecisionLifecycleState.CREATED:
            if to_state is not DecisionLifecycleState.EVALUATING:
                raise DecisionLifecycleTransitionRejectedError(
                    from_state=from_state,
                    to_state=to_state,
                )
            return
        self._default.assert_transition_allowed(
            from_state=from_state,
            to_state=to_state,
        )


def test_decision_lifecycle_accepts_swappable_transition_provider() -> None:
    engine = _engine(transition=_EvaluatingGateTransitionProvider())
    record = engine.begin_decision(
        decision_type=DecisionType.PRODUCTION_MODEL_ROUTING,
        source_references=(),
        actor=_ACTOR,
        reason="custom_transition",
    )
    record = engine.transition(
        record,
        to_state=DecisionLifecycleState.EVALUATING,
        reason="evaluate",
        actor=_ACTOR,
    )
    record = engine.transition(
        record,
        to_state=DecisionLifecycleState.APPROVED,
        reason="approve",
        actor=_ACTOR,
    )
    assert record.lifecycle_state is DecisionLifecycleState.APPROVED


def test_orchestration_result_maps_to_lifecycle_record() -> None:
    profiles = _profiles("model-a")
    orchestrator = DecisionOrchestrator(
        selection_provider=EngineBackedSelectionProvider(
            ModelSelectionEngine(strategies=default_selection_strategies())
        ),
        governance_provider=EngineBackedGovernanceProvider(
            GovernanceEvaluationEngine(evaluators=default_governance_policies())
        ),
        execution_provider=RecordingExecutionProvider(),
    )
    orchestration = orchestrator.orchestrate(
        _orchestration_request(profiles),
        orchestrated_at=datetime(2026, 9, 12, 12, 0, 0, tzinfo=UTC),
    )
    assert orchestration.outcome is DecisionOrchestrationOutcome.SUCCESS

    engine = _engine()
    record = lifecycle_record_from_orchestration_result(engine, orchestration)
    refs = source_references_from_orchestration_result(orchestration)

    assert record.lifecycle_state is DecisionLifecycleState.CREATED
    assert record.decision_type is DecisionType.PRODUCTION_MODEL_ROUTING
    assert DecisionSourceKind.MODEL_SELECTION in {r.source_kind for r in refs}
    assert DecisionSourceKind.GOVERNANCE in {r.source_kind for r in refs}
    assert DecisionSourceKind.EXECUTION in {r.source_kind for r in refs}
    assert record.source_references == refs
