"""Authoritative product-identification request context for verification (5C10)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    StructuredAttributeConstraint,
    StructuredConstraintOperator,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifier,
)


class MissingRequirementOrigin(StrEnum):
    """Whether a missing fact comes from the user request or catalog evidence."""

    USER = "user"
    CATALOG = "catalog"


@dataclass(frozen=True, slots=True)
class MissingDistinguishingRequirement:
    """Typed missing fact — no natural-language clarification in 5C10."""

    attribute_name: str
    origin: MissingRequirementOrigin
    requirement_id: str

    def __post_init__(self) -> None:
        if not self.attribute_name.strip():
            raise ValueError("attribute_name must be non-empty")
        if not self.requirement_id.strip():
            raise ValueError("requirement_id must be non-empty")


@dataclass(frozen=True, slots=True)
class NegativeAttributeConstraint:
    """User exclusion constraint — catalog must not exhibit the excluded value."""

    attribute_name: str
    operator: StructuredConstraintOperator
    excluded_value: str

    def __post_init__(self) -> None:
        if not self.attribute_name.strip():
            raise ValueError("attribute_name must be non-empty")
        if not self.excluded_value.strip():
            raise ValueError("excluded_value must be non-empty")
        if self.operator is not StructuredConstraintOperator.EQUALS:
            raise ValueError("negative constraints support EQUALS exclusion only")


@dataclass(frozen=True, slots=True)
class ProductIdentificationQueryContext:
    """Minimum authoritative query semantics consumed by verification."""

    required_constraints: tuple[StructuredAttributeConstraint, ...] = ()
    negative_constraints: tuple[NegativeAttributeConstraint, ...] = ()
    missing_user_distinguishing_requirements: tuple[MissingDistinguishingRequirement, ...] = ()
    requested_identifiers: tuple[ProductIdentifier, ...] = ()
    soft_preferences: tuple[StructuredAttributeConstraint, ...] = ()

    def __post_init__(self) -> None:
        if type(self.required_constraints) is not tuple:
            raise TypeError("required_constraints must be a tuple")
        if type(self.negative_constraints) is not tuple:
            raise TypeError("negative_constraints must be a tuple")
        if type(self.missing_user_distinguishing_requirements) is not tuple:
            raise TypeError("missing_user_distinguishing_requirements must be a tuple")
        if type(self.requested_identifiers) is not tuple:
            raise TypeError("requested_identifiers must be a tuple")
        if type(self.soft_preferences) is not tuple:
            raise TypeError("soft_preferences must be a tuple")


@dataclass(frozen=True, slots=True)
class HypothesisRejectionEvidence:
    """Positive rejection reasoning when no ranked hypotheses remain."""

    rejection_reason_code: str
    attribute_key: str
    catalog_value: str
    source_field: str

    def __post_init__(self) -> None:
        if not self.rejection_reason_code.strip():
            raise ValueError("rejection_reason_code must be non-empty")
        if not self.attribute_key.strip():
            raise ValueError("attribute_key must be non-empty")
        if not self.catalog_value.strip():
            raise ValueError("catalog_value must be non-empty")
        if not self.source_field.strip():
            raise ValueError("source_field must be non-empty")
