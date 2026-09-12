# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

from testing_support.decision_e2e.local_ai_incident_qualification import (
    QualificationCliExit,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationSessionState,
)
from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing import (
    GOVERNANCE_TASK_ID,
    DataClassificationPolicy,
    DataClassificationPolicyConfig,
    DataSensitivityClass,
    GovernanceDisposition,
    GovernanceEvaluationEngine,
    GovernanceEvaluationRequest,
    GovernancePolicyRef,
    GovernanceReasonCode,
    GovernanceRiskTier,
    GovernanceTaskContext,
    HumanApprovalRiskPolicy,
    PolicyEvaluationResult,
    QualificationCompliancePolicy,
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
    ModelSelectionStatus,
    TaskCapabilityRequirement,
    TaskRequirements,
    default_selection_strategies,
)
from testing_support.decision_e2e.model_matrix.qualification_cohort_executor import (
    CohortExecutionStatus,
)
from testing_support.decision_e2e.model_matrix.registry import (
    qualification_matrix_version,
)


def _outcome(
    profile_key: str,
    *,
    exit_code: QualificationCliExit = QualificationCliExit.SUCCESS,
) -> ModelQualificationOutcome:
    stamp = datetime(2026, 9, 12, 9, 0, 0, tzinfo=UTC)
    return ModelQualificationOutcome(
        profile_key=profile_key,
        provider="ollama",
        model_name=profile_key,
        matrix_version=qualification_matrix_version(),
        qualification_task_id="DS-E2E-15J-L1.R6",
        evaluated_at=stamp,
        status=CohortExecutionStatus.EXECUTED,
        exit_code=exit_code,
        session_state=QualificationSessionState.FINALIZED,
    )


def _profiles(
    *profile_keys: str, exit_codes: dict[str, QualificationCliExit] | None = None
):
    matrix_version = qualification_matrix_version()
    codes = exit_codes or {}
    outcomes = tuple(
        _outcome(key, exit_code=codes.get(key, QualificationCliExit.SUCCESS))
        for key in profile_keys
    )
    built = build_model_capability_profiles(
        CapabilityProfileBuildRequest(matrix_version=matrix_version, outcomes=outcomes)
    )
    return built.profiles


def _recommendation_for(
    profiles: tuple,
    *,
    minimum: ObservationLevel = ObservationLevel.MODERATE,
):
    request = ModelSelectionRequest(
        task_requirements=TaskRequirements(
            scenario_id="governance-scenario",
            capability_requirements=(
                TaskCapabilityRequirement(
                    dimension_id=CapabilityDimensionId.QUALIFICATION_EXIT,
                    minimum_level=minimum,
                ),
            ),
        ),
        capability_constraints=CapabilitySelectionConstraints(
            required_matrix_version=qualification_matrix_version(),
            excluded_profile_keys=(),
            require_behavioral_baseline=False,
        ),
        available_model_profiles=profiles,
    )
    engine = ModelSelectionEngine(strategies=default_selection_strategies())
    return engine.recommend(
        request,
        recommended_at=datetime(2026, 9, 12, 12, 0, 0, tzinfo=UTC),
    )


def _all_policy_refs() -> tuple[GovernancePolicyRef, ...]:
    policies = default_governance_policies()
    return tuple(
        GovernancePolicyRef(
            policy_id=item.policy_id,
            policy_version=item.policy_version,
        )
        for item in policies
    )


def test_governance_allow_when_policies_permit() -> None:
    profiles = _profiles("model-a", "model-b")
    recommendation = _recommendation_for(profiles)
    assert recommendation.status is ModelSelectionStatus.RECOMMENDED

    engine = GovernanceEvaluationEngine(evaluators=default_governance_policies())
    decision = engine.evaluate(
        GovernanceEvaluationRequest(
            model_recommendation=recommendation,
            task_context=GovernanceTaskContext(
                scenario_id="governance-scenario",
                data_sensitivity=DataSensitivityClass.PUBLIC,
                risk_tier=GovernanceRiskTier.LOW,
            ),
            applicable_policies=_all_policy_refs(),
            capability_evidence=profiles,
        ),
        evaluated_at=datetime(2026, 9, 12, 12, 30, 0, tzinfo=UTC),
    )

    assert decision.disposition is GovernanceDisposition.ALLOW
    assert decision.audit_metadata.governance_task_id == GOVERNANCE_TASK_ID
    assert decision.audit_metadata.recommended_profile_key == "model-a"
    assert decision.policy_references


def test_governance_block_when_data_policy_denies() -> None:
    profiles = _profiles(
        "model-a",
        "model-b",
        exit_codes={"model-a": QualificationCliExit.CRITICAL_SAFETY_FAILURE},
    )
    recommendation = _recommendation_for(profiles)
    assert recommendation.selected_model_reference is not None
    assert recommendation.selected_model_reference.profile_key == "model-b"

    engine = GovernanceEvaluationEngine(
        evaluators=(
            DataClassificationPolicy(
                DataClassificationPolicyConfig(
                    restricted_sensitivity=DataSensitivityClass.FINANCIAL,
                    allowed_profile_keys=("model-a",),
                )
            ),
            QualificationCompliancePolicy(),
            HumanApprovalRiskPolicy(),
        )
    )
    decision = engine.evaluate(
        GovernanceEvaluationRequest(
            model_recommendation=recommendation,
            task_context=GovernanceTaskContext(
                scenario_id="governance-scenario",
                data_sensitivity=DataSensitivityClass.FINANCIAL,
                risk_tier=GovernanceRiskTier.LOW,
            ),
            applicable_policies=_all_policy_refs(),
            capability_evidence=profiles,
        )
    )

    assert decision.disposition is GovernanceDisposition.BLOCK
    assert any(
        item.reason_code is GovernanceReasonCode.DATA_CLASSIFICATION_DENIED
        for item in decision.reason_references
    )


def test_governance_require_approval_for_high_risk() -> None:
    profiles = _profiles("model-a", "model-b")
    recommendation = _recommendation_for(profiles)

    engine = GovernanceEvaluationEngine(evaluators=default_governance_policies())
    decision = engine.evaluate(
        GovernanceEvaluationRequest(
            model_recommendation=recommendation,
            task_context=GovernanceTaskContext(
                scenario_id="governance-scenario",
                data_sensitivity=DataSensitivityClass.PUBLIC,
                risk_tier=GovernanceRiskTier.HIGH,
            ),
            applicable_policies=_all_policy_refs(),
            capability_evidence=profiles,
        )
    )

    assert decision.disposition is GovernanceDisposition.REQUIRE_APPROVAL
    assert any(
        item.reason_code is GovernanceReasonCode.HIGH_RISK_REQUIRES_APPROVAL
        for item in decision.reason_references
    )


def test_custom_policy_plugs_in_without_engine_edit() -> None:
    class _BlockModelBPolicy:
        policy_id = "block_model_b_test"
        policy_version = "test-1"

        def evaluate(
            self, request: GovernanceEvaluationRequest
        ) -> PolicyEvaluationResult:
            from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing.contracts import (
                GovernanceDisposition,
                GovernanceReasonCode,
                GovernanceReasonRef,
            )

            recommendation = request.model_recommendation
            key = (
                None
                if recommendation is None
                or recommendation.selected_model_reference is None
                else recommendation.selected_model_reference.profile_key
            )
            if key == "model-b":
                return PolicyEvaluationResult(
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                    contribution=GovernanceDisposition.BLOCK,
                    reasons=(
                        GovernanceReasonRef(
                            reason_code=GovernanceReasonCode.DATA_CLASSIFICATION_DENIED,
                            summary="test plugin blocks model-b",
                            policy_id=self.policy_id,
                        ),
                    ),
                )
            return PolicyEvaluationResult(
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                contribution=GovernanceDisposition.ALLOW,
                reasons=(
                    GovernanceReasonRef(
                        reason_code=GovernanceReasonCode.POLICY_ALLOW,
                        summary="test plugin allows",
                        policy_id=self.policy_id,
                    ),
                ),
            )

    profiles = _profiles(
        "model-a",
        "model-b",
        exit_codes={"model-a": QualificationCliExit.CRITICAL_SAFETY_FAILURE},
    )
    recommendation = _recommendation_for(profiles)
    assert recommendation.selected_model_reference is not None
    assert recommendation.selected_model_reference.profile_key == "model-b"

    compliance = QualificationCompliancePolicy()
    approval = HumanApprovalRiskPolicy()
    engine = GovernanceEvaluationEngine(
        evaluators=(
            compliance,
            approval,
            _BlockModelBPolicy(),
        )
    )
    decision = engine.evaluate(
        GovernanceEvaluationRequest(
            model_recommendation=recommendation,
            task_context=GovernanceTaskContext(
                scenario_id="governance-scenario",
                data_sensitivity=DataSensitivityClass.PUBLIC,
                risk_tier=GovernanceRiskTier.LOW,
            ),
            applicable_policies=(
                GovernancePolicyRef(
                    policy_id="block_model_b_test",
                    policy_version="test-1",
                ),
                GovernancePolicyRef(
                    policy_id=compliance.policy_id,
                    policy_version=compliance.policy_version,
                ),
                GovernancePolicyRef(
                    policy_id=approval.policy_id,
                    policy_version=approval.policy_version,
                ),
            ),
            capability_evidence=profiles,
        )
    )

    assert decision.disposition is GovernanceDisposition.BLOCK
    assert "block_model_b_test" in {
        item.policy_id for item in decision.policy_references
    }


def test_governance_blocks_without_model_recommendation() -> None:
    profiles = _profiles("model-a")
    engine = GovernanceEvaluationEngine(evaluators=default_governance_policies())
    decision = engine.evaluate(
        GovernanceEvaluationRequest(
            model_recommendation=None,
            task_context=GovernanceTaskContext(
                scenario_id="governance-scenario",
                data_sensitivity=DataSensitivityClass.PUBLIC,
                risk_tier=GovernanceRiskTier.LOW,
            ),
            applicable_policies=_all_policy_refs(),
            capability_evidence=profiles,
        )
    )

    assert decision.disposition is GovernanceDisposition.BLOCK
    assert any(
        item.reason_code is GovernanceReasonCode.NO_MODEL_RECOMMENDATION
        for item in decision.reason_references
    )
    assert decision.audit_metadata.recommended_profile_key is None
