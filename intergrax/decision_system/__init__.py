# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Decision System domain package (DS-E2E-14.3).

Owns reusable Decision qualification contracts separate from Execution Engine
runtime and from provider-centric ``intergrax.core.qualification``.
"""

from intergrax.decision_system.completion_eligibility import (
    CompletionEligibilityDecision,
    CompletionEligibilityStatus,
    evaluate_completion_eligibility,
    evaluate_completion_eligibility_with_provider,
)
from intergrax.decision_system.evidence_requirements import (
    EvidenceRequirement,
    EvidenceRequirementAssessment,
    EvidenceRequirementContractError,
    EvidenceRequirementCriticality,
    EvidenceRequirementId,
    EvidenceRequirementOutcome,
    EvidenceRequirementProvider,
    EvidenceRequirementProviderAbsentError,
    EvidenceRequirementSet,
    EvidenceRequirementWaiverRef,
    EvidenceSufficiencyAssessment,
    validate_evidence_requirement_id,
    validate_evidence_requirement_set,
    validate_evidence_requirement_waiver_ref,
    validate_evidence_sufficiency_assessment,
)

__all__ = (
    "CompletionEligibilityDecision",
    "CompletionEligibilityStatus",
    "EvidenceRequirement",
    "EvidenceRequirementAssessment",
    "EvidenceRequirementContractError",
    "EvidenceRequirementCriticality",
    "EvidenceRequirementId",
    "EvidenceRequirementOutcome",
    "EvidenceRequirementProvider",
    "EvidenceRequirementProviderAbsentError",
    "EvidenceRequirementSet",
    "EvidenceRequirementWaiverRef",
    "EvidenceSufficiencyAssessment",
    "evaluate_completion_eligibility",
    "evaluate_completion_eligibility_with_provider",
    "validate_evidence_requirement_id",
    "validate_evidence_requirement_set",
    "validate_evidence_requirement_waiver_ref",
    "validate_evidence_sufficiency_assessment",
)
