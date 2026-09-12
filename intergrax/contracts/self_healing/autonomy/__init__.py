# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

from intergrax.contracts.self_healing.autonomy.action_kind import AutonomyRequestedActionKind
from intergrax.contracts.self_healing.autonomy.approval import (
    HumanApprovalRequirement,
    HumanApprovalRequirementResolver,
)
from intergrax.contracts.self_healing.autonomy.audit import AutonomyAuditBundle
from intergrax.contracts.self_healing.autonomy.context import AutonomyDecisionContext
from intergrax.contracts.self_healing.autonomy.decision import AutonomyControlDecision
from intergrax.contracts.self_healing.autonomy.engine import AutonomyControlEngine
from intergrax.contracts.self_healing.autonomy.authorizing_guard import AutonomyExecutionAuthorizingGuard
from intergrax.contracts.self_healing.autonomy.execution_audit import (
    AutonomyExecutionAuditRecord,
    AutonomyExecutionAuditRepository,
)
from intergrax.contracts.self_healing.autonomy.execution_authorization import (
    AutonomyExecutionAuthorization,
    AutonomyExecutionAuthorizationStatus,
)
from intergrax.contracts.self_healing.autonomy.execution_boundary import (
    AutonomyAdmissionContextSource,
    AutonomyExecutionBoundary,
)
from intergrax.contracts.self_healing.autonomy.execution_denied import AutonomyExecutionDeniedError
from intergrax.contracts.self_healing.autonomy.guard import (
    AutonomyExecutionAdmissionContext,
    AutonomyExecutionGuard,
    AutonomyGuardCheckResult,
    AutonomyGuardVerdict,
)
from intergrax.contracts.self_healing.autonomy.guard_rule import (
    AutonomyExecutionGuardRule,
    AutonomyExecutionGuardRuleVerdict,
)
from intergrax.contracts.self_healing.autonomy.approval_evaluation import (
    HumanApprovalEvaluationResult,
    HumanApprovalEvaluator,
)
from intergrax.contracts.self_healing.autonomy.decision_evaluator import AutonomyDecisionEvaluator
from intergrax.contracts.self_healing.autonomy.evaluation_audit import (
    AutonomyEvaluationAuditRecorder,
    AutonomyEvaluationAuditTrailEntry,
)
from intergrax.contracts.self_healing.autonomy.evaluation_confidence import (
    AutonomyEvaluationConfidence,
    AutonomyEvaluationConfidenceLevel,
)
from intergrax.contracts.self_healing.autonomy.evaluation_repository import AutonomyDecisionRepository
from intergrax.contracts.self_healing.autonomy.evaluation_result import (
    AUTONOMY_EVALUATION_CONTRACT_VERSION,
    AutonomyEvaluationResult,
    AutonomyEvaluationSourceKind,
    AutonomyEvaluationSourceRef,
    AutonomyEvaluationVerdict,
)
from intergrax.contracts.self_healing.autonomy.explanation import (
    AutonomyDecisionExplanation,
    AutonomyExplanationBullet,
    AutonomyRuleInfluence,
)
from intergrax.contracts.self_healing.autonomy.ids import (
    mint_autonomy_control_decision_id,
    mint_autonomy_evaluation_id,
    mint_autonomy_execution_audit_record_id,
    mint_autonomy_execution_authorization_id,
    mint_autonomy_qualification_validation_id,
    mint_autonomy_recommendation_correlation_id,
)
from intergrax.contracts.self_healing.autonomy.qualification import (
    AUTONOMY_QUALIFICATION_CONTRACT_VERSION,
    AutonomyQualificationAuditInfo,
    AutonomyQualificationRepository,
    AutonomyQualificationResult,
    AutonomyQualificationStatus,
    AutonomySafetyCheck,
    AutonomySafetyCheckResult,
    AutonomySafetyIssue,
    AutonomySafetyQualificationContext,
    AutonomySafetyValidator,
)
from intergrax.contracts.self_healing.autonomy.policy_evaluation import (
    AutonomyPolicyEvaluationResult,
    AutonomyPolicyEvaluationVerdict,
    AutonomyPolicyEvaluator,
)
from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.policy import (
    AutonomyConstraintDescriptor,
    AutonomyPolicy,
    AutonomyPolicyOutcome,
)
from intergrax.contracts.self_healing.autonomy.repository import (
    AutonomyDecisionCorrelationQuery,
    AutonomyRepository,
)
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest
from intergrax.contracts.self_healing.autonomy.risk import (
    AutonomyRiskAssessment,
    AutonomyRiskBand,
    AutonomyRiskEvaluationResult,
    AutonomyRiskEvaluator,
    AutonomyRiskFactor,
    autonomy_risk_evaluation_result_from_assessment,
)

__all__ = [
    "AUTONOMY_EVALUATION_CONTRACT_VERSION",
    "AUTONOMY_QUALIFICATION_CONTRACT_VERSION",
    "AutonomyAuditBundle",
    "AutonomyConstraintDescriptor",
    "AutonomyControlDecision",
    "AutonomyControlEngine",
    "AutonomyControlRequest",
    "AutonomyDecisionContext",
    "AutonomyDecisionCorrelationQuery",
    "AutonomyDecisionEvaluator",
    "AutonomyDecisionExplanation",
    "AutonomyDecisionRepository",
    "AutonomyEvaluationAuditRecorder",
    "AutonomyEvaluationAuditTrailEntry",
    "AutonomyEvaluationConfidence",
    "AutonomyEvaluationConfidenceLevel",
    "AutonomyEvaluationResult",
    "AutonomyEvaluationSourceKind",
    "AutonomyEvaluationSourceRef",
    "AutonomyEvaluationVerdict",
    "AutonomyAdmissionContextSource",
    "AutonomyExecutionAdmissionContext",
    "AutonomyExecutionAuditRecord",
    "AutonomyExecutionAuditRepository",
    "AutonomyExecutionAuthorization",
    "AutonomyExecutionAuthorizationStatus",
    "AutonomyExecutionAuthorizingGuard",
    "AutonomyExecutionBoundary",
    "AutonomyExecutionDeniedError",
    "AutonomyExecutionGuard",
    "AutonomyExecutionGuardRule",
    "AutonomyExecutionGuardRuleVerdict",
    "AutonomyExplanationBullet",
    "AutonomyGuardCheckResult",
    "AutonomyGuardVerdict",
    "AutonomyLevel",
    "AutonomyPolicy",
    "AutonomyPolicyEvaluationResult",
    "AutonomyPolicyEvaluationVerdict",
    "AutonomyPolicyEvaluator",
    "AutonomyPolicyOutcome",
    "AutonomyQualificationAuditInfo",
    "AutonomyQualificationRepository",
    "AutonomyQualificationResult",
    "AutonomyQualificationStatus",
    "AutonomyRepository",
    "AutonomySafetyCheck",
    "AutonomySafetyCheckResult",
    "AutonomySafetyIssue",
    "AutonomySafetyQualificationContext",
    "AutonomySafetyValidator",
    "AutonomyRequestedActionKind",
    "AutonomyRiskAssessment",
    "AutonomyRiskBand",
    "AutonomyRiskEvaluationResult",
    "AutonomyRiskEvaluator",
    "AutonomyRiskFactor",
    "AutonomyRuleInfluence",
    "HumanApprovalEvaluationResult",
    "HumanApprovalEvaluator",
    "HumanApprovalRequirement",
    "HumanApprovalRequirementResolver",
    "autonomy_risk_evaluation_result_from_assessment",
    "mint_autonomy_control_decision_id",
    "mint_autonomy_evaluation_id",
    "mint_autonomy_execution_audit_record_id",
    "mint_autonomy_execution_authorization_id",
    "mint_autonomy_qualification_validation_id",
    "mint_autonomy_recommendation_correlation_id",
]
