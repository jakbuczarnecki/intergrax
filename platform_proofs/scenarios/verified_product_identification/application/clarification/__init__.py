"""Targeted clarification requirement selection (5C11)."""

from platform_proofs.scenarios.verified_product_identification.application.clarification.composition import (
    build_clarification_requirement_selection_service,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.contracts import (
    ClarificationRequirement,
    ClarificationRequirementKind,
    ClarificationSelectionRequest,
    ClarificationSelectionResult,
    NoClarificationReason,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.service import (
    ClarificationRequirementSelectionService,
)

__all__ = (
    "ClarificationRequirement",
    "ClarificationRequirementKind",
    "ClarificationRequirementSelectionService",
    "ClarificationSelectionRequest",
    "ClarificationSelectionResult",
    "NoClarificationReason",
    "build_clarification_requirement_selection_service",
)
