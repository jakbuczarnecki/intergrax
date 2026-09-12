# © Artur Czarnecki. All rights reserved.

"""Enterprise evolution governance framework — process control without approval (L17)."""

from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.audit_providers import (
    StandardEvolutionGovernanceFrameworkAuditProvider,
    default_evolution_governance_framework_audit_provider,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.contracts import (
    ENTERPRISE_EVOLUTION_GOVERNANCE_FRAMEWORK_TASK_ID,
    ENTERPRISE_EVOLUTION_GOVERNANCE_FRAMEWORK_VERSION,
    EnterpriseEvolutionIntelligenceResult,
    EnterpriseEvolutionStrategyResult,
    EvolutionGovernanceFrameworkAuditMetadata,
    EvolutionGovernanceFrameworkContext,
    EvolutionGovernanceFrameworkResult,
    EvolutionGovernanceFrameworkStatus,
    EvolutionGovernanceIssue,
    EvolutionGovernanceIssueSeverity,
    EvolutionGovernanceLifecycleStage,
    EvolutionGovernanceProcessReference,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.control_providers import (
    DefaultEvolutionGovernanceControlProvider,
    default_evolution_governance_control_providers,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.engine import (
    EnterpriseEvolutionGovernanceFrameworkEngine,
    run_governance_framework_evaluation,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.governance_providers import (
    DefaultEnterpriseEvolutionGovernanceProvider,
    default_enterprise_evolution_governance_provider,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.lifecycle_providers import (
    DefaultEvolutionLifecycleGovernanceProvider,
    default_evolution_lifecycle_governance_provider,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.policy_providers import (
    AuditTraceabilityPolicy,
    EvidenceCompletenessPolicy,
    LifecycleConsistencyPolicy,
    default_evolution_governance_policy_providers,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.protocol import (
    EnterpriseEvolutionGovernanceProvider,
    EvolutionGovernanceControlProvider,
    EvolutionGovernanceFrameworkAuditProvider,
    EvolutionGovernancePolicyProvider,
    EvolutionLifecycleGovernanceProvider,
)


def default_enterprise_evolution_governance_framework_engine() -> (
    EnterpriseEvolutionGovernanceFrameworkEngine
):
    return EnterpriseEvolutionGovernanceFrameworkEngine(
        governance_providers=(),
        lifecycle_provider=default_evolution_lifecycle_governance_provider(),
        policy_providers=default_evolution_governance_policy_providers(),
        control_providers=default_evolution_governance_control_providers(),
        audit_provider=default_evolution_governance_framework_audit_provider(),
    )


__all__ = [
    "ENTERPRISE_EVOLUTION_GOVERNANCE_FRAMEWORK_TASK_ID",
    "ENTERPRISE_EVOLUTION_GOVERNANCE_FRAMEWORK_VERSION",
    "AuditTraceabilityPolicy",
    "DefaultEnterpriseEvolutionGovernanceProvider",
    "DefaultEvolutionGovernanceControlProvider",
    "DefaultEvolutionLifecycleGovernanceProvider",
    "EnterpriseEvolutionGovernanceFrameworkEngine",
    "EnterpriseEvolutionGovernanceProvider",
    "EnterpriseEvolutionIntelligenceResult",
    "EnterpriseEvolutionStrategyResult",
    "EvidenceCompletenessPolicy",
    "EvolutionGovernanceControlProvider",
    "EvolutionGovernanceFrameworkAuditMetadata",
    "EvolutionGovernanceFrameworkAuditProvider",
    "EvolutionGovernanceFrameworkContext",
    "EvolutionGovernanceFrameworkResult",
    "EvolutionGovernanceFrameworkStatus",
    "EvolutionGovernanceIssue",
    "EvolutionGovernanceIssueSeverity",
    "EvolutionGovernanceLifecycleStage",
    "EvolutionGovernancePolicyProvider",
    "EvolutionGovernanceProcessReference",
    "EvolutionLifecycleGovernanceProvider",
    "LifecycleConsistencyPolicy",
    "StandardEvolutionGovernanceFrameworkAuditProvider",
    "default_enterprise_evolution_governance_framework_engine",
    "default_enterprise_evolution_governance_provider",
    "default_evolution_governance_control_providers",
    "default_evolution_governance_framework_audit_provider",
    "default_evolution_governance_policy_providers",
    "default_evolution_lifecycle_governance_provider",
    "run_governance_framework_evaluation",
]
