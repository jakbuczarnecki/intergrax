"""Constructor injection for clarification selection."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.clarification.answerability_policy import (
    ClarificationAnswerabilityPolicy,
    DeterministicClarificationAnswerabilityPolicy,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.materiality_policy import (
    ClarificationMaterialityPolicy,
    DeterministicClarificationMaterialityPolicy,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.selection_strategy import (
    ClarificationRequirementSelectionStrategy,
    DeterministicClarificationRequirementSelectionStrategy,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.service import (
    ClarificationRequirementSelectionService,
)


def build_clarification_requirement_selection_service(
    *,
    selection_strategy: ClarificationRequirementSelectionStrategy | None = None,
    answerability_policy: ClarificationAnswerabilityPolicy | None = None,
    materiality_policy: ClarificationMaterialityPolicy | None = None,
) -> ClarificationRequirementSelectionService:
    return ClarificationRequirementSelectionService(
        selection_strategy=(
            selection_strategy
            if selection_strategy is not None
            else DeterministicClarificationRequirementSelectionStrategy()
        ),
        answerability_policy=(
            answerability_policy
            if answerability_policy is not None
            else DeterministicClarificationAnswerabilityPolicy()
        ),
        materiality_policy=(
            materiality_policy
            if materiality_policy is not None
            else DeterministicClarificationMaterialityPolicy()
        ),
    )
