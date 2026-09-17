# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Enterprise Reliability Layer runtime (ERL)."""

from intergrax.runtime.enterprise_reliability.admission_boundary import (
    ExternalEffectAdmissionCaseError,
    ExternalEffectAdmissionContextError,
    admit_external_effect_into_enterprise_reliability,
)
from intergrax.runtime.enterprise_reliability.contract_admission import (
    ExternalEffectUnknownAdmission,
    admit_external_effect_unknown_with_contract,
)
from intergrax.runtime.enterprise_reliability.plugin_gateway import (
    EnterpriseReliabilityPluginGatewayImpl,
)
from intergrax.runtime.enterprise_reliability.plugin_registry import (
    EnterpriseReliabilityPluginRegistryConfigurationError,
    InMemoryEnterpriseReliabilityPluginRegistry,
)
from intergrax.runtime.enterprise_reliability.evidence_evaluation import (
    EvidenceEvaluationOrchestrationError,
    evaluate_external_effect_evidence,
)
from intergrax.runtime.enterprise_reliability.reconciliation_evidence import (
    apply_reconciliation_evidence,
    materialize_external_effect_evidence_from_probe,
)
from intergrax.runtime.enterprise_reliability.reconciliation_execution import (
    ExternalEffectReconciliationProbeRun,
    execute_external_effect_reconciliation_probe,
)
from intergrax.runtime.enterprise_reliability.reconciliation_orchestration import (
    ExternalEffectReconciliationPlanning,
    ReconciliationOrchestrationError,
    plan_external_effect_reconciliation,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_reconciliation import (
    ProviderInvocationReconciliationRun,
    reconcile_durable_provider_invocation_unknown,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryExecutionBlockReason,
    ProviderInvocationRecoveryExecutionDisposition,
    ProviderInvocationRecoveryExecutionPorts,
    ProviderInvocationRecoveryExecutionResult,
    ProviderInvocationRecoveryHitlResult,
    ProviderInvocationRecoveryRepeatResult,
    build_recovery_escalation_context,
    decide_provider_invocation_recovery,
    execute_provider_invocation_recovery,
)
from intergrax.runtime.enterprise_reliability.default_provider_invocation_recovery_policy import (
    DEFAULT_FAIL_CLOSED_PROVIDER_INVOCATION_RECOVERY_POLICY_ID,
    DefaultFailClosedProviderInvocationRecoveryPolicy,
    default_fail_closed_provider_invocation_recovery_policy,
)
from intergrax.runtime.enterprise_reliability.resolution_execution import (
    ExternalEffectResolutionRun,
    execute_external_effect_resolution,
)
from intergrax.runtime.enterprise_reliability.compensation_execution import (
    CompensationExecutionFailure,
    ExternalEffectCompensationRun,
    execute_external_effect_compensation,
)
from intergrax.runtime.enterprise_reliability.compensation_orchestration import (
    CompensationOrchestrationError,
    ExternalEffectCompensationPlanning,
    plan_external_effect_compensation,
)
from intergrax.runtime.enterprise_reliability.governance_orchestration import (
    ExternalEffectGovernanceEvaluation,
    GovernanceOrchestrationError,
    evaluate_external_effect_governance,
)
from intergrax.runtime.enterprise_reliability.lifecycle_handoff_orchestration import (
    handoff_recovery_lifecycle_to_execution,
)
from intergrax.runtime.enterprise_reliability.recovery_orchestration import (
    ExternalEffectRecoveryRecommendation,
    RecoveryOrchestrationError,
    recommend_external_effect_recovery_lifecycle,
)
from intergrax.runtime.enterprise_reliability.resolution_orchestration import (
    ExternalEffectResolutionPlanning,
    ResolutionOrchestrationError,
    plan_external_effect_resolution,
)
from intergrax.runtime.enterprise_reliability.case_lifecycle_coordination import (
    transition_reliability_case_lifecycle,
)
from intergrax.runtime.enterprise_reliability.uncertainty_lifecycle import (
    UncertaintyResolutionError,
    admit_external_effect_unknown,
    advance_uncertainty_lifecycle,
    resolve_uncertainty,
)

__all__ = [
    "EnterpriseReliabilityPluginGatewayImpl",
    "EnterpriseReliabilityPluginRegistryConfigurationError",
    "ExternalEffectAdmissionCaseError",
    "ExternalEffectAdmissionContextError",
    "ExternalEffectUnknownAdmission",
    "InMemoryEnterpriseReliabilityPluginRegistry",
    "ExternalEffectReconciliationPlanning",
    "ExternalEffectReconciliationProbeRun",
    "CompensationExecutionFailure",
    "CompensationOrchestrationError",
    "ExternalEffectCompensationPlanning",
    "ExternalEffectCompensationRun",
    "ExternalEffectGovernanceEvaluation",
    "ExternalEffectRecoveryRecommendation",
    "ExternalEffectResolutionPlanning",
    "ExternalEffectResolutionRun",
    "GovernanceOrchestrationError",
    "ReconciliationOrchestrationError",
    "RecoveryOrchestrationError",
    "ResolutionOrchestrationError",
    "execute_external_effect_compensation",
    "plan_external_effect_compensation",
    "EvidenceEvaluationOrchestrationError",
    "evaluate_external_effect_evidence",
    "apply_reconciliation_evidence",
    "execute_external_effect_reconciliation_probe",
    "execute_external_effect_resolution",
    "plan_external_effect_resolution",
    "materialize_external_effect_evidence_from_probe",
    "evaluate_external_effect_governance",
    "handoff_recovery_lifecycle_to_execution",
    "plan_external_effect_reconciliation",
    "ProviderInvocationReconciliationRun",
    "reconcile_durable_provider_invocation_unknown",
    "DEFAULT_FAIL_CLOSED_PROVIDER_INVOCATION_RECOVERY_POLICY_ID",
    "DefaultFailClosedProviderInvocationRecoveryPolicy",
    "ProviderInvocationRecoveryExecutionBlockReason",
    "ProviderInvocationRecoveryExecutionDisposition",
    "ProviderInvocationRecoveryExecutionPorts",
    "ProviderInvocationRecoveryExecutionResult",
    "ProviderInvocationRecoveryHitlResult",
    "ProviderInvocationRecoveryRepeatResult",
    "build_recovery_escalation_context",
    "decide_provider_invocation_recovery",
    "default_fail_closed_provider_invocation_recovery_policy",
    "execute_provider_invocation_recovery",
    "recommend_external_effect_recovery_lifecycle",
    "UncertaintyResolutionError",
    "admit_external_effect_into_enterprise_reliability",
    "admit_external_effect_unknown",
    "admit_external_effect_unknown_with_contract",
    "advance_uncertainty_lifecycle",
    "resolve_uncertainty",
    "transition_reliability_case_lifecycle",
]
