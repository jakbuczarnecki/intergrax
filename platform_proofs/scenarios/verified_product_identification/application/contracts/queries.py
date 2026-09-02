"""Minimal provider-neutral catalog search query contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifier,
)


def _validate_limit(limit: int) -> None:
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive int")


@dataclass(frozen=True, slots=True)
class ExactIdentifierQuery:
    identifier: ProductIdentifier
    limit: int = 10

    def __post_init__(self) -> None:
        _validate_limit(self.limit)


@dataclass(frozen=True, slots=True)
class LexicalSearchQuery:
    query_text: str
    limit: int = 20

    def __post_init__(self) -> None:
        if not isinstance(self.query_text, str) or not self.query_text.strip():
            raise ValueError("LexicalSearchQuery.query_text must be a non-empty string")
        _validate_limit(self.limit)


class StructuredConstraintOperator(StrEnum):
    """Minimal structured constraint operators for hard material constraints."""

    EQUALS = "eq"
    CONTAINS = "contains"


@dataclass(frozen=True, slots=True)
class StructuredAttributeConstraint:
    attribute_name: str
    operator: StructuredConstraintOperator
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.attribute_name, str) or not self.attribute_name.strip():
            raise ValueError("StructuredAttributeConstraint.attribute_name must be non-empty")
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("StructuredAttributeConstraint.value must be non-empty")


@dataclass(frozen=True, slots=True)
class StructuredSearchQuery:
    constraints: tuple[StructuredAttributeConstraint, ...]
    limit: int = 20

    def __post_init__(self) -> None:
        if type(self.constraints) is not tuple:
            raise TypeError("StructuredSearchQuery.constraints must be a tuple")
        if not self.constraints:
            raise ValueError("StructuredSearchQuery.constraints must not be empty")
        _validate_limit(self.limit)


@dataclass(frozen=True, slots=True)
class VectorSearchQuery:
    query_text: str
    limit: int = 20

    def __post_init__(self) -> None:
        if not isinstance(self.query_text, str) or not self.query_text.strip():
            raise ValueError("VectorSearchQuery.query_text must be a non-empty string")
        _validate_limit(self.limit)
