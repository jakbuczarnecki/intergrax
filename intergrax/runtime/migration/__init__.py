# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Migration-only runtime helpers scheduled for removal with legacy Critic retirement."""

from intergrax.runtime.migration.critic_shadow_adapter import (
    CriticShadowAdapter,
    CriticShadowConfig,
    build_critic_shadow_adapter,
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
    "CriticRetirementReadiness",
    "LegacyCriticHumanEscalationEvidence",
    "LegacyCriticRetiredAction",
    "LegacyCriticRetiredLayer",
    "proven_retired_l2_human_escalation_evidence",
    "CriticShadowAdapter",
    "CriticShadowConfig",
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
    "build_critic_shadow_adapter",
    "compare_decision_critic_parity",
    "evaluate_critic_retirement_readiness",
    "project_critic_observation",
    "project_decision_observation",
)
