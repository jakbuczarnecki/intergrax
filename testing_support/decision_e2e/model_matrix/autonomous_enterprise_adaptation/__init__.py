# © Artur Czarnecki. All rights reserved.

"""Controlled enterprise adaptation — applies governed evolutions only (L13)."""

from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.adaptation_providers import (
    DefaultEnterpriseAdaptationProvider,
)
from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.audit_providers import (
    StandardAdaptationAuditProvider,
    default_adaptation_audit_provider,
)
from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.contracts import (
    AUTONOMOUS_ENTERPRISE_ADAPTATION_TASK_ID,
    AUTONOMOUS_ENTERPRISE_ADAPTATION_VERSION,
    AdaptationAuditMetadata,
    AdaptationConstraint,
    AdaptationExecutionResult,
    AdaptationExecutionStatus,
    AdaptationScope,
    ApprovedAdaptationRequest,
    EvolutionSourceReference,
    GovernanceApprovalReference,
)
from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.engine import (
    AutonomousEnterpriseAdaptationEngine,
    default_autonomous_enterprise_adaptation_engine,
)
from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.protocol import (
    AdaptationApplyOutcome,
    AdaptationAuditProvider,
    EnterpriseAdaptationProvider,
)

__all__ = [
    "AUTONOMOUS_ENTERPRISE_ADAPTATION_TASK_ID",
    "AUTONOMOUS_ENTERPRISE_ADAPTATION_VERSION",
    "AdaptationApplyOutcome",
    "AdaptationAuditMetadata",
    "AdaptationAuditProvider",
    "AdaptationConstraint",
    "AdaptationExecutionResult",
    "AdaptationExecutionStatus",
    "AdaptationScope",
    "ApprovedAdaptationRequest",
    "AutonomousEnterpriseAdaptationEngine",
    "DefaultEnterpriseAdaptationProvider",
    "EnterpriseAdaptationProvider",
    "EvolutionSourceReference",
    "GovernanceApprovalReference",
    "StandardAdaptationAuditProvider",
    "default_adaptation_audit_provider",
    "default_autonomous_enterprise_adaptation_engine",
]
