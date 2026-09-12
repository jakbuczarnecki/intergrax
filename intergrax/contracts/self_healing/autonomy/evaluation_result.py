# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Final autonomy evaluation result — evaluation only, not execution (SELF-HEALING R6.2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.self_healing.autonomy.approval_evaluation import HumanApprovalEvaluationResult
from intergrax.contracts.self_healing.autonomy.audit import AutonomyAuditBundle
from intergrax.contracts.self_healing.autonomy.evaluation_confidence import AutonomyEvaluationConfidence
from intergrax.contracts.self_healing.autonomy.explanation import AutonomyDecisionExplanation
from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.policy_evaluation import AutonomyPolicyEvaluationResult
from intergrax.contracts.self_healing.autonomy.risk import AutonomyRiskEvaluationResult

AUTONOMY_EVALUATION_CONTRACT_VERSION = "R6.2"


class AutonomyEvaluationVerdict(StrEnum):
    """Evaluation clearance — decision authority acts elsewhere."""

    CLEARED = "CLEARED"
    CONDITIONAL = "CONDITIONAL"
    DENIED = "DENIED"


class AutonomyEvaluationSourceKind(StrEnum):
    POLICY = "POLICY"
    RISK = "RISK"
    APPROVAL = "APPROVAL"
    RECOMMENDATION = "RECOMMENDATION"
    EVALUATOR = "EVALUATOR"


@dataclass(frozen=True, slots=True)
class AutonomyEvaluationSourceRef:
    kind: AutonomyEvaluationSourceKind
    source_id: str

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id required")


@dataclass(frozen=True, slots=True)
class AutonomyEvaluationResult:
    evaluation_id: str
    verdict: AutonomyEvaluationVerdict
    autonomy_level: AutonomyLevel
    confidence: AutonomyEvaluationConfidence
    reasons: tuple[str, ...]
    sources: tuple[AutonomyEvaluationSourceRef, ...]
    explanation: AutonomyDecisionExplanation
    policy_result: AutonomyPolicyEvaluationResult
    risk_result: AutonomyRiskEvaluationResult
    approval_result: HumanApprovalEvaluationResult
    audit_bundle: AutonomyAuditBundle
    contract_version: str
    recommendation_correlation_id: str
    evaluator_id: str

    def __post_init__(self) -> None:
        if not self.evaluation_id.strip():
            raise ValueError("evaluation_id required")
        if not self.contract_version.strip():
            raise ValueError("contract_version required")
        if not self.recommendation_correlation_id.strip():
            raise ValueError("recommendation_correlation_id required")
        if not self.evaluator_id.strip():
            raise ValueError("evaluator_id required")
        self.autonomy_level.ensure_runtime_activatable()
        if self.audit_bundle.recommendation_correlation_id != self.recommendation_correlation_id:
            raise ValueError("recommendation_correlation_id mismatch with audit_bundle")
        if not self.reasons:
            raise ValueError("reasons required")


__all__ = [
    "AUTONOMY_EVALUATION_CONTRACT_VERSION",
    "AutonomyEvaluationResult",
    "AutonomyEvaluationSourceKind",
    "AutonomyEvaluationSourceRef",
    "AutonomyEvaluationVerdict",
]
