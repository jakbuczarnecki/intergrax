# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Pure completion eligibility evaluation (DS-E2E-15B.2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.decision_system.evidence_requirements import (
    EvidenceRequirementCriticality,
    EvidenceRequirementId,
    EvidenceRequirementOutcome,
    EvidenceRequirementProvider,
    EvidenceRequirementProviderAbsentError,
    EvidenceRequirementSet,
    EvidenceSufficiencyAssessment,
    validate_evidence_sufficiency_assessment,
)


class CompletionEligibilityStatus(StrEnum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"


@dataclass(frozen=True, slots=True)
class CompletionEligibilityDecision:
    status: CompletionEligibilityStatus
    unresolved_mandatory_requirement_ids: tuple[EvidenceRequirementId, ...]
    satisfied_mandatory_count: int
    waived_mandatory_count: int
    optional_unsatisfied_count: int


def evaluate_completion_eligibility(
    requirement_set: EvidenceRequirementSet,
    assessment: EvidenceSufficiencyAssessment,
) -> CompletionEligibilityDecision:
    """Pure eligibility rule: mandatory requirements must be satisfied or waived."""
    validate_evidence_sufficiency_assessment(requirement_set, assessment)

    requirements_by_id = {
        requirement.requirement_id: requirement for requirement in requirement_set.requirements
    }
    assessments_by_id = {
        item.requirement_id: item for item in assessment.assessments
    }

    unresolved_mandatory: list[EvidenceRequirementId] = []
    satisfied_mandatory = 0
    waived_mandatory = 0
    optional_unsatisfied = 0

    for requirement_id, requirement in requirements_by_id.items():
        item = assessments_by_id[requirement_id]
        if requirement.criticality is EvidenceRequirementCriticality.OPTIONAL:
            if item.outcome is EvidenceRequirementOutcome.UNSATISFIED:
                optional_unsatisfied += 1
            continue

        if item.outcome is EvidenceRequirementOutcome.SATISFIED:
            satisfied_mandatory += 1
            continue
        if item.outcome is EvidenceRequirementOutcome.WAIVED:
            waived_mandatory += 1
            continue
        unresolved_mandatory.append(requirement_id)

    unresolved_tuple = tuple(unresolved_mandatory)
    if unresolved_tuple:
        return CompletionEligibilityDecision(
            status=CompletionEligibilityStatus.INELIGIBLE,
            unresolved_mandatory_requirement_ids=unresolved_tuple,
            satisfied_mandatory_count=satisfied_mandatory,
            waived_mandatory_count=waived_mandatory,
            optional_unsatisfied_count=optional_unsatisfied,
        )

    return CompletionEligibilityDecision(
        status=CompletionEligibilityStatus.ELIGIBLE,
        unresolved_mandatory_requirement_ids=(),
        satisfied_mandatory_count=satisfied_mandatory,
        waived_mandatory_count=waived_mandatory,
        optional_unsatisfied_count=optional_unsatisfied,
    )


def evaluate_completion_eligibility_with_provider(
    *,
    provider: EvidenceRequirementProvider | None,
    assessment: EvidenceSufficiencyAssessment,
) -> CompletionEligibilityDecision:
    """Fail closed when an enabled gate has no configured provider."""
    if provider is None:
        raise EvidenceRequirementProviderAbsentError(
            "completion eligibility gate requires an EvidenceRequirementProvider"
        )
    requirement_set = provider.requirements_for()
    return evaluate_completion_eligibility(requirement_set, assessment)
