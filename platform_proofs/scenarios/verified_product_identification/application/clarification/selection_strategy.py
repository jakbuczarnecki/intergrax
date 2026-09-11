"""Deterministic clarification requirement ordering."""

from __future__ import annotations

from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.application.clarification.contracts import (
    ClarificationRequirement,
    ClarificationRequirementKind,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
)


class ClarificationRequirementSelectionStrategy(Protocol):
    def select(
        self,
        candidates: tuple[ClarificationRequirement, ...],
    ) -> tuple[ClarificationRequirement | None, tuple[ClarificationRequirement, ...]]:
        """Return primary requirement and ordered alternates."""


def _authority_tier(requirement: ClarificationRequirement) -> int:
    if requirement.kind is ClarificationRequirementKind.USER_MISSING_FACT:
        return 0
    if requirement.kind is ClarificationRequirementKind.IDENTIFIER_VALUE:
        if requirement.identifier_type is ProductIdentifierType.GTIN:
            return 1
        if requirement.identifier_type is ProductIdentifierType.MPN:
            return 2
        return 5
    return 3


def _selection_category(requirement: ClarificationRequirement) -> int:
    if requirement.kind is ClarificationRequirementKind.USER_MISSING_FACT:
        return 0
    is_identifier = requirement.kind is ClarificationRequirementKind.IDENTIFIER_VALUE
    complete = requirement.discrimination.has_complete_coverage
    if not is_identifier and complete:
        return 1
    if is_identifier and complete:
        return 2
    if not is_identifier and not complete:
        return 3
    return 4


def _prefer_user_answerable_attribute_over_identifier(
    ordered: tuple[ClarificationRequirement, ...],
) -> tuple[ClarificationRequirement, ...]:
    """Prefer material attributes over technical identifiers when both fully discriminate."""

    attributes = [
        item
        for item in ordered
        if item.kind
        in (
            ClarificationRequirementKind.IDENTITY_DISCRIMINATOR,
            ClarificationRequirementKind.ATTRIBUTE_VALUE,
            ClarificationRequirementKind.USER_MISSING_FACT,
        )
        and item.discrimination.has_complete_coverage
    ]
    identifiers = [
        item
        for item in ordered
        if item.kind is ClarificationRequirementKind.IDENTIFIER_VALUE
        and item.discrimination.has_complete_coverage
    ]
    if attributes and identifiers:
        rest = [item for item in ordered if item not in attributes and item not in identifiers]
        return tuple(attributes + identifiers + rest)
    return ordered


class DeterministicClarificationRequirementSelectionStrategy:
    def select(
        self,
        candidates: tuple[ClarificationRequirement, ...],
    ) -> tuple[ClarificationRequirement | None, tuple[ClarificationRequirement, ...]]:
        if not candidates:
            return None, ()
        sorted_candidates = tuple(
            sorted(
                candidates,
                key=lambda item: (
                    _selection_category(item),
                    _authority_tier(item),
                    -item.discrimination.eliminable_hypothesis_count,
                    -item.discrimination.distinct_known_value_count,
                    -item.discrimination.known_hypothesis_count,
                    item.attribute_name.casefold(),
                    item.requirement_id,
                ),
            )
        )
        reordered = _prefer_user_answerable_attribute_over_identifier(sorted_candidates)
        primary = reordered[0]
        alternates = reordered[1:]
        return primary, alternates
