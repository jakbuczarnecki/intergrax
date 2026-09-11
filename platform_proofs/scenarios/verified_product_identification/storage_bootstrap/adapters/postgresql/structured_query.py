"""Structured search query preparation — dedupe, normalize, ordinals."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.catalog.structured_attribute_normalization import (
    normalize_structured_query_attribute_name,
    normalize_structured_query_value,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
    CatalogSearchFailureKind,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    StructuredConstraintOperator,
    StructuredSearchQuery,
)


@dataclass(frozen=True, slots=True)
class PreparedStructuredConstraint:
    constraint_ordinal: int
    normalized_key: str
    operator: StructuredConstraintOperator
    normalized_value: str


@dataclass(frozen=True, slots=True)
class PreparedStructuredSearchQuery:
    constraints: tuple[PreparedStructuredConstraint, ...]
    total_constraint_count: int
    limit: int
    has_contains: bool


def prepare_structured_search_query(
    query: StructuredSearchQuery,
) -> PreparedStructuredSearchQuery | CatalogSearchFailure:
    """Normalize, deduplicate, and assign deterministic ordinals for one structured query."""
    deduped: list[PreparedStructuredConstraint] = []
    seen: set[tuple[str, StructuredConstraintOperator, str]] = set()
    has_contains = False

    for constraint in query.constraints:
        try:
            normalized_key = normalize_structured_query_attribute_name(constraint.attribute_name)
            normalized_value = normalize_structured_query_value(constraint.value)
        except ValueError as exc:
            return CatalogSearchFailure(
                kind=CatalogSearchFailureKind.INVALID_QUERY,
                message=str(exc),
            )

        dedup_key = (normalized_key, constraint.operator, normalized_value)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        if constraint.operator is StructuredConstraintOperator.CONTAINS:
            has_contains = True
        deduped.append(
            PreparedStructuredConstraint(
                constraint_ordinal=len(deduped),
                normalized_key=normalized_key,
                operator=constraint.operator,
                normalized_value=normalized_value,
            )
        )

    if not deduped:
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.INVALID_QUERY,
            message="structured query has no usable constraints after deduplication",
        )

    return PreparedStructuredSearchQuery(
        constraints=tuple(deduped),
        total_constraint_count=len(deduped),
        limit=query.limit,
        has_contains=has_contains,
    )
