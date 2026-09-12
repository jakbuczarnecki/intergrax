# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from testing_support.decision_e2e.local_ai_incident_qualification import (
    QualificationCliExit,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationSessionState,
)
from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing import (
    DataSensitivityClass,
    GovernanceDecision,
    GovernanceDisposition,
    GovernanceEvaluationEngine,
    GovernanceEvaluationRequest,
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
    ModelSelectionRecommendation,
    ModelSelectionStatus,
    TaskCapabilityRequirement,
    TaskRequirements,
    default_selection_strategies,
)
from testing_support.decision_e2e.model_matrix.production_decision_orchestration import (
    DecisionExecutionRequest,
    DecisionExecutionResultReference,
    DecisionExecutionStatus,
    DecisionOrchestrationLifecycleStage,
    DecisionOrchestrationOutcome,
    DecisionOrchestrationProviderMissingError,
    DecisionOrchestrationRequest,
    DecisionOrchestrator,
    EngineBackedGovernanceProvider,
    EngineBackedSelectionProvider,
    RecordingExecutionProvider,
)
from testing_support.decision_e2e.model_matrix.qualification_cohort_executor import (
    CohortExecutionStatus,
)
from testing_support.decision_e2e.model_matrix.registry import (
    qualification_matrix_version,
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


def _default_orchestrator() -> DecisionOrchestrator:
    return DecisionOrchestrator(
        selection_provider=EngineBackedSelectionProvider(
            ModelSelectionEngine(strategies=default_selection_strategies())
        ),
        governance_provider=EngineBackedGovernanceProvider(
            GovernanceEvaluationEngine(evaluators=default_governance_policies())
        ),
        execution_provider=RecordingExecutionProvider(),
    )


def test_orchestration_success_selection_allow_execution() -> None:
    profiles = _profiles("model-a", "model-b")
    orchestrator = _default_orchestrator()
    stamp = datetime(2026, 9, 12, 14, 0, 0, tzinfo=UTC)

    result = orchestrator.orchestrate(
        _orchestration_request(profiles),
        orchestrated_at=stamp,
    )

    assert result.outcome is DecisionOrchestrationOutcome.SUCCESS
    assert result.selection_result.status is ModelSelectionStatus.RECOMMENDED
    assert result.governance_result is not None
    assert result.governance_result.disposition is GovernanceDisposition.ALLOW
    assert result.execution_result_reference is not None
    assert result.execution_result_reference.status is DecisionExecutionStatus.EXECUTED
    assert result.lifecycle_metadata.lifecycle_stages == (
        DecisionOrchestrationLifecycleStage.SELECTED,
        DecisionOrchestrationLifecycleStage.ALLOWED,
        DecisionOrchestrationLifecycleStage.EXECUTED,
    )


@dataclass
class _SpyExecutionProvider:
    calls: int = 0

    @property
    def provider_id(self) -> str:
        return "spy-execution"

    def execute(
        self,
        request: DecisionExecutionRequest,
        *,
        executed_at: datetime | None = None,
    ) -> DecisionExecutionResultReference:
        self.calls += 1
        raise AssertionError("execution must not run when governance blocks")


@dataclass
class _BlockingGovernanceProvider:
    inner: EngineBackedGovernanceProvider

    @property
    def provider_id(self) -> str:
        return "blocking-governance"

    def evaluate(
        self,
        request: GovernanceEvaluationRequest,
        *,
        evaluated_at: datetime | None = None,
    ) -> GovernanceDecision:
        decision = self.inner.evaluate(request, evaluated_at=evaluated_at)
        if decision.disposition is not GovernanceDisposition.ALLOW:
            return decision
        blocked = GovernanceDecision(
            disposition=GovernanceDisposition.BLOCK,
            reason_references=decision.reason_references,
            policy_references=decision.policy_references,
            audit_metadata=decision.audit_metadata,
        )
        return blocked


def test_orchestration_governance_block_skips_execution() -> None:
    profiles = _profiles("model-a")
    spy = _SpyExecutionProvider()
    orchestrator = DecisionOrchestrator(
        selection_provider=EngineBackedSelectionProvider(
            ModelSelectionEngine(strategies=default_selection_strategies())
        ),
        governance_provider=_BlockingGovernanceProvider(
            EngineBackedGovernanceProvider(
                GovernanceEvaluationEngine(evaluators=default_governance_policies())
            )
        ),
        execution_provider=spy,
    )

    result = orchestrator.orchestrate(_orchestration_request(profiles))

    assert result.outcome is DecisionOrchestrationOutcome.GOVERNANCE_BLOCKED
    assert result.execution_result_reference is None
    assert spy.calls == 0
    assert DecisionOrchestrationLifecycleStage.STOPPED in (
        result.lifecycle_metadata.lifecycle_stages
    )


def test_orchestration_require_approval_stops_process() -> None:
    profiles = _profiles("model-a")
    spy = _SpyExecutionProvider()
    orchestrator = DecisionOrchestrator(
        selection_provider=EngineBackedSelectionProvider(
            ModelSelectionEngine(strategies=default_selection_strategies())
        ),
        governance_provider=EngineBackedGovernanceProvider(
            GovernanceEvaluationEngine(evaluators=default_governance_policies())
        ),
        execution_provider=spy,
    )
    request = DecisionOrchestrationRequest(
        selection_request=_orchestration_request(profiles).selection_request,
        governance_task_context=GovernanceTaskContext(
            scenario_id="orchestration-scenario",
            data_sensitivity=DataSensitivityClass.PUBLIC,
            risk_tier=GovernanceRiskTier.HIGH,
        ),
        applicable_policies=_orchestration_request(profiles).applicable_policies,
        capability_evidence=profiles,
    )

    result = orchestrator.orchestrate(request)

    assert result.outcome is DecisionOrchestrationOutcome.APPROVAL_REQUIRED
    assert result.governance_result is not None
    assert (
        result.governance_result.disposition is GovernanceDisposition.REQUIRE_APPROVAL
    )
    assert result.execution_result_reference is None
    assert spy.calls == 0
    assert result.lifecycle_metadata.lifecycle_stages == (
        DecisionOrchestrationLifecycleStage.SELECTED,
        DecisionOrchestrationLifecycleStage.REQUIRE_APPROVAL,
        DecisionOrchestrationLifecycleStage.STOPPED,
    )


@dataclass
class _TaggedSelectionProvider:
    tag: str = "custom-selection"

    @property
    def provider_id(self) -> str:
        return self.tag

    def recommend(
        self,
        request: ModelSelectionRequest,
        *,
        recommended_at: datetime | None = None,
    ) -> ModelSelectionRecommendation:
        inner = EngineBackedSelectionProvider(
            ModelSelectionEngine(strategies=default_selection_strategies())
        )
        return inner.recommend(request, recommended_at=recommended_at)


def test_orchestration_accepts_swappable_providers() -> None:
    profiles = _profiles("model-a")
    orchestrator = DecisionOrchestrator(
        selection_provider=_TaggedSelectionProvider(tag="swap-selection"),
        governance_provider=EngineBackedGovernanceProvider(
            GovernanceEvaluationEngine(evaluators=default_governance_policies())
        ),
        execution_provider=RecordingExecutionProvider(),
    )

    result = orchestrator.orchestrate(_orchestration_request(profiles))

    assert result.outcome is DecisionOrchestrationOutcome.SUCCESS
    assert result.lifecycle_metadata.selection_provider_id == "swap-selection"


def test_orchestration_missing_provider_raises_without_partial_run() -> None:
    profiles = _profiles("model-a")
    orchestrator = DecisionOrchestrator(
        selection_provider=EngineBackedSelectionProvider(
            ModelSelectionEngine(strategies=default_selection_strategies())
        ),
        governance_provider=None,
        execution_provider=RecordingExecutionProvider(),
    )

    with pytest.raises(DecisionOrchestrationProviderMissingError):
        orchestrator.orchestrate(_orchestration_request(profiles))
