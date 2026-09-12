# © Artur Czarnecki. All rights reserved.

"""Enterprise evolution assurance — read-only quality validation (L18)."""

from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.assurance_providers import (
    DefaultEnterpriseEvolutionAssuranceProvider,
    default_enterprise_evolution_assurance_provider,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.audit_providers import (
    StandardEvolutionAssuranceAuditProvider,
    default_evolution_assurance_audit_provider,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.compliance_validator_providers import (
    GovernanceFrameworkAlignmentValidator,
    RequiredStagesComplianceValidator,
    default_evolution_compliance_validator_providers,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.contracts import (
    ENTERPRISE_EVOLUTION_ASSURANCE_TASK_ID,
    ENTERPRISE_EVOLUTION_ASSURANCE_VERSION,
    EvolutionAssuranceAuditMetadata,
    EvolutionAssuranceContext,
    EvolutionAssuranceFinding,
    EvolutionAssuranceFindingSeverity,
    EvolutionAssuranceLifecycleStage,
    EvolutionAssuranceResult,
    EvolutionAssuranceStatus,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.engine import (
    EnterpriseEvolutionAssuranceEngine,
    run_evolution_assurance_assessment,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.evidence_validator_providers import (
    AuditTraceEvidenceValidator,
    SourceLineageEvidenceValidator,
    default_evolution_evidence_validator_providers,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.protocol import (
    EnterpriseEvolutionAssuranceProvider,
    EvolutionAssuranceAuditProvider,
    EvolutionComplianceValidatorProvider,
    EvolutionEvidenceValidatorProvider,
    EvolutionQualityValidatorProvider,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_assurance.quality_validator_providers import (
    EvidenceAvailabilityValidator,
    LifecycleCompletenessValidator,
    VersionConsistencyValidator,
    default_evolution_quality_validator_providers,
)


def default_enterprise_evolution_assurance_engine() -> (
    EnterpriseEvolutionAssuranceEngine
):
    return EnterpriseEvolutionAssuranceEngine(
        quality_validator_providers=default_evolution_quality_validator_providers(),
        compliance_validator_providers=default_evolution_compliance_validator_providers(),
        evidence_validator_providers=default_evolution_evidence_validator_providers(),
        assurance_providers=(),
        audit_provider=default_evolution_assurance_audit_provider(),
    )


__all__ = [
    "ENTERPRISE_EVOLUTION_ASSURANCE_TASK_ID",
    "ENTERPRISE_EVOLUTION_ASSURANCE_VERSION",
    "AuditTraceEvidenceValidator",
    "DefaultEnterpriseEvolutionAssuranceProvider",
    "EnterpriseEvolutionAssuranceEngine",
    "EnterpriseEvolutionAssuranceProvider",
    "EvidenceAvailabilityValidator",
    "EvolutionAssuranceAuditMetadata",
    "EvolutionAssuranceAuditProvider",
    "EvolutionAssuranceContext",
    "EvolutionAssuranceFinding",
    "EvolutionAssuranceFindingSeverity",
    "EvolutionAssuranceLifecycleStage",
    "EvolutionAssuranceResult",
    "EvolutionAssuranceStatus",
    "EvolutionComplianceValidatorProvider",
    "EvolutionEvidenceValidatorProvider",
    "EvolutionQualityValidatorProvider",
    "GovernanceFrameworkAlignmentValidator",
    "LifecycleCompletenessValidator",
    "RequiredStagesComplianceValidator",
    "SourceLineageEvidenceValidator",
    "StandardEvolutionAssuranceAuditProvider",
    "VersionConsistencyValidator",
    "default_enterprise_evolution_assurance_engine",
    "default_enterprise_evolution_assurance_provider",
    "default_evolution_assurance_audit_provider",
    "default_evolution_compliance_validator_providers",
    "default_evolution_evidence_validator_providers",
    "default_evolution_quality_validator_providers",
    "run_evolution_assurance_assessment",
]
