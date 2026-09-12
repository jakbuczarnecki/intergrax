# © Artur Czarnecki. All rights reserved.

"""Default governance policy plugins (DS-E2E-15J-L5)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing.contracts import (
    DataSensitivityClass,
    GovernanceDisposition,
    GovernanceEvaluationRequest,
    GovernanceReasonCode,
    GovernanceReasonRef,
    GovernanceRiskTier,
    ModelCapabilityProfile,
)
from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing.protocol import (
    PolicyEvaluationResult,
)
from testing_support.decision_e2e.model_matrix.model_capability_baseline.contracts import (
    CapabilityDimensionId,
    ObservationLevel,
)

_DATA_CLASSIFICATION_POLICY_ID = "data_classification"
_DATA_CLASSIFICATION_POLICY_VERSION = "1"
_QUALIFICATION_COMPLIANCE_POLICY_ID = "qualification_compliance"
_QUALIFICATION_COMPLIANCE_POLICY_VERSION = "1"
_HUMAN_APPROVAL_RISK_POLICY_ID = "human_approval_risk"
_HUMAN_APPROVAL_RISK_POLICY_VERSION = "1"


def _recommended_profile_key(request: GovernanceEvaluationRequest) -> str | None:
    recommendation = request.model_recommendation
    if recommendation is None or recommendation.selected_model_reference is None:
        return None
    return recommendation.selected_model_reference.profile_key


def _profile_for_key(
    profiles: tuple[ModelCapabilityProfile, ...],
    profile_key: str,
) -> ModelCapabilityProfile | None:
    for profile in profiles:
        if profile.model_identity.profile_key == profile_key:
            return profile
    return None


def _allow_reason(policy_id: str, summary: str) -> PolicyEvaluationResult:
    return PolicyEvaluationResult(
        policy_id=policy_id,
        policy_version="1",
        contribution=GovernanceDisposition.ALLOW,
        reasons=(
            GovernanceReasonRef(
                reason_code=GovernanceReasonCode.POLICY_ALLOW,
                summary=summary,
                policy_id=policy_id,
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class DataClassificationPolicyConfig:
    restricted_sensitivity: DataSensitivityClass
    allowed_profile_keys: tuple[str, ...]


class DataClassificationPolicy:
    """Blocks recommended models that are not approved for a sensitive data class."""

    def __init__(self, config: DataClassificationPolicyConfig) -> None:
        self._config = config

    @property
    def policy_id(self) -> str:
        return _DATA_CLASSIFICATION_POLICY_ID

    @property
    def policy_version(self) -> str:
        return _DATA_CLASSIFICATION_POLICY_VERSION

    def evaluate(self, request: GovernanceEvaluationRequest) -> PolicyEvaluationResult:
        key = _recommended_profile_key(request)
        if key is None:
            return PolicyEvaluationResult(
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                contribution=GovernanceDisposition.ALLOW,
                reasons=(
                    GovernanceReasonRef(
                        reason_code=GovernanceReasonCode.POLICY_ALLOW,
                        summary="no recommended model; defer to engine pre-check",
                        policy_id=self.policy_id,
                    ),
                ),
            )
        sensitivity = request.task_context.data_sensitivity
        if sensitivity is not self._config.restricted_sensitivity:
            return _allow_reason(
                self.policy_id,
                f"data sensitivity {sensitivity} not restricted by this policy",
            )
        if key in self._config.allowed_profile_keys:
            return _allow_reason(
                self.policy_id,
                f"profile {key} approved for {sensitivity}",
            )
        return PolicyEvaluationResult(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            contribution=GovernanceDisposition.BLOCK,
            reasons=(
                GovernanceReasonRef(
                    reason_code=GovernanceReasonCode.DATA_CLASSIFICATION_DENIED,
                    summary=(
                        f"profile {key} not approved for {sensitivity} data processing"
                    ),
                    policy_id=self.policy_id,
                ),
            ),
        )


class QualificationCompliancePolicy:
    """Blocks when capability evidence shows unsafe qualification observations."""

    @property
    def policy_id(self) -> str:
        return _QUALIFICATION_COMPLIANCE_POLICY_ID

    @property
    def policy_version(self) -> str:
        return _QUALIFICATION_COMPLIANCE_POLICY_VERSION

    def evaluate(self, request: GovernanceEvaluationRequest) -> PolicyEvaluationResult:
        key = _recommended_profile_key(request)
        if key is None:
            return _allow_reason(
                self.policy_id,
                "no recommended model; defer to engine pre-check",
            )
        profile = _profile_for_key(request.capability_evidence, key)
        if profile is None:
            return PolicyEvaluationResult(
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                contribution=GovernanceDisposition.BLOCK,
                reasons=(
                    GovernanceReasonRef(
                        reason_code=GovernanceReasonCode.QUALIFICATION_EVIDENCE_MISSING,
                        summary=f"no capability profile for recommended model {key}",
                        policy_id=self.policy_id,
                    ),
                ),
            )
        weak_limits = [
            item
            for item in profile.limitations
            if item.dimension_id is CapabilityDimensionId.QUALIFICATION_EXIT
            and item.level is ObservationLevel.WEAK
        ]
        if weak_limits:
            return PolicyEvaluationResult(
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                contribution=GovernanceDisposition.BLOCK,
                reasons=(
                    GovernanceReasonRef(
                        reason_code=GovernanceReasonCode.QUALIFICATION_SAFETY_LIMIT,
                        summary=weak_limits[0].factual_descriptor,
                        policy_id=self.policy_id,
                    ),
                ),
            )
        return _allow_reason(
            self.policy_id,
            f"qualification evidence acceptable for {key}",
        )


class HumanApprovalRiskPolicy:
    """Requires human approval for high-risk task contexts."""

    @property
    def policy_id(self) -> str:
        return _HUMAN_APPROVAL_RISK_POLICY_ID

    @property
    def policy_version(self) -> str:
        return _HUMAN_APPROVAL_RISK_POLICY_VERSION

    def evaluate(self, request: GovernanceEvaluationRequest) -> PolicyEvaluationResult:
        if request.task_context.risk_tier is GovernanceRiskTier.HIGH:
            return PolicyEvaluationResult(
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                contribution=GovernanceDisposition.REQUIRE_APPROVAL,
                reasons=(
                    GovernanceReasonRef(
                        reason_code=GovernanceReasonCode.HIGH_RISK_REQUIRES_APPROVAL,
                        summary="high risk tier requires human approval before execution",
                        policy_id=self.policy_id,
                    ),
                ),
            )
        return _allow_reason(
            self.policy_id,
            f"risk tier {request.task_context.risk_tier} does not require approval",
        )


def default_governance_policies() -> tuple[
    DataClassificationPolicy,
    QualificationCompliancePolicy,
    HumanApprovalRiskPolicy,
]:
    return (
        DataClassificationPolicy(
            DataClassificationPolicyConfig(
                restricted_sensitivity=DataSensitivityClass.FINANCIAL,
                allowed_profile_keys=("model-a",),
            )
        ),
        QualificationCompliancePolicy(),
        HumanApprovalRiskPolicy(),
    )


__all__ = [
    "DataClassificationPolicy",
    "DataClassificationPolicyConfig",
    "HumanApprovalRiskPolicy",
    "QualificationCompliancePolicy",
    "default_governance_policies",
]
