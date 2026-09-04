# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Migration-only runtime helpers for Critic retirement historical evidence."""

from intergrax.runtime.migration.critic_retirement_qualification import (
    CriticRetirementEvidenceProvenance,
    CriticRetirementQualification,
    DS_MIG_03_HITL_TRANSITION_COMMIT,
    FINAL_PRE_RETIREMENT_REGRESSION_COMMIT,
    PARITY_QUALIFICATION_SOURCE_COMMIT,
    proven_critic_retirement_qualification,
)
from intergrax.runtime.migration.legacy_critic_human_evidence import (
    LegacyCriticHumanEscalationEvidence,
    LegacyCriticRetiredAction,
    LegacyCriticRetiredLayer,
    proven_retired_l2_human_escalation_evidence,
)
from intergrax.runtime.migration.decision_critic_parity import (
    CriticRetirementReadiness,
    DecisionCriticParityClassification,
    DecisionCriticParityDifferenceCode,
    DecisionCriticParityIdentity,
    DecisionCriticParityMetrics,
    DecisionCriticParityObserver,
    DecisionCriticParityResult,
    NormalizedParityOutcome,
    ParityCapabilityRequirement,
    ParityCapabilityRequirementMode,
    ParityHostScope,
    ParityVerificationCapability,
    DEFAULT_CRITIC_RETIREMENT_CAPABILITY_REQUIREMENTS,
    aggregate_parity_metrics,
    compare_decision_critic_parity,
    evaluate_critic_retirement_readiness,
    project_critic_observation,
    project_decision_observation,
)

__all__ = (
    "CriticRetirementEvidenceProvenance",
    "CriticRetirementQualification",
    "CriticRetirementReadiness",
    "DS_MIG_03_HITL_TRANSITION_COMMIT",
    "FINAL_PRE_RETIREMENT_REGRESSION_COMMIT",
    "LegacyCriticHumanEscalationEvidence",
    "LegacyCriticRetiredAction",
    "LegacyCriticRetiredLayer",
    "PARITY_QUALIFICATION_SOURCE_COMMIT",
    "proven_critic_retirement_qualification",
    "proven_retired_l2_human_escalation_evidence",
    "DEFAULT_CRITIC_RETIREMENT_CAPABILITY_REQUIREMENTS",
    "DecisionCriticParityClassification",
    "DecisionCriticParityDifferenceCode",
    "DecisionCriticParityIdentity",
    "DecisionCriticParityMetrics",
    "DecisionCriticParityObserver",
    "DecisionCriticParityResult",
    "NormalizedParityOutcome",
    "ParityCapabilityRequirement",
    "ParityCapabilityRequirementMode",
    "ParityHostScope",
    "ParityVerificationCapability",
    "aggregate_parity_metrics",
    "compare_decision_critic_parity",
    "evaluate_critic_retirement_readiness",
    "project_critic_observation",
    "project_decision_observation",
)
