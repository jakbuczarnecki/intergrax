"""User answerability classification for clarification candidates."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
    identity_scope_for_identifier_type,
    ProductIdentifierIdentityScope,
)


class ClarificationAnswerabilityClass(StrEnum):
    USER_NATIVE = "user_native"
    TECHNICAL_BUT_REASONABLE = "technical_but_reasonable"
    SOURCE_INTERNAL = "source_internal"
    NOT_USER_ANSWERABLE = "not_user_answerable"


class ClarificationAnswerabilityPolicy(Protocol):
    def classify_attribute(self, attribute_name: str) -> ClarificationAnswerabilityClass:
        """Classify whether a structured attribute is user-answerable."""

    def classify_identifier(
        self,
        identifier_type: ProductIdentifierType,
        *,
        query_context: ProductIdentificationQueryContext,
    ) -> ClarificationAnswerabilityClass:
        """Classify whether an identifier clarification is appropriate."""

    def is_selectable(self, answerability: ClarificationAnswerabilityClass) -> bool:
        """Return whether a classified requirement may be offered to the user."""


_NON_ANSWERABLE_ATTRIBUTE_KEYS = frozenset(
    {
        "competing_identity",
        "variant",
        "identity_evidence",
        "evaluated_hypotheses",
        "title",
        "price",
        "seller",
        "semantic",
        "bm25",
        "vector",
    }
)


class DeterministicClarificationAnswerabilityPolicy:
    """Conservative deterministic answerability — no product ontology."""

    def classify_attribute(self, attribute_name: str) -> ClarificationAnswerabilityClass:
        key = attribute_name.casefold()
        if key in _NON_ANSWERABLE_ATTRIBUTE_KEYS:
            return ClarificationAnswerabilityClass.NOT_USER_ANSWERABLE
        return ClarificationAnswerabilityClass.USER_NATIVE

    def classify_identifier(
        self,
        identifier_type: ProductIdentifierType,
        *,
        query_context: ProductIdentificationQueryContext,
    ) -> ClarificationAnswerabilityClass:
        scope = identity_scope_for_identifier_type(identifier_type)
        if scope is ProductIdentifierIdentityScope.SOURCE_LOCAL:
            return ClarificationAnswerabilityClass.SOURCE_INTERNAL
        if identifier_type is ProductIdentifierType.GTIN:
            if query_context.requested_identifiers:
                for item in query_context.requested_identifiers:
                    if item.identifier_type is ProductIdentifierType.GTIN:
                        return ClarificationAnswerabilityClass.TECHNICAL_BUT_REASONABLE
            return ClarificationAnswerabilityClass.TECHNICAL_BUT_REASONABLE
        if identifier_type is ProductIdentifierType.MPN:
            return ClarificationAnswerabilityClass.USER_NATIVE
        return ClarificationAnswerabilityClass.NOT_USER_ANSWERABLE

    def is_selectable(self, answerability: ClarificationAnswerabilityClass) -> bool:
        return answerability in (
            ClarificationAnswerabilityClass.USER_NATIVE,
            ClarificationAnswerabilityClass.TECHNICAL_BUT_REASONABLE,
        )
