"""Authoritative typed product-identification query (5C12-R1)."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    ProductIdentificationQueryContext,
)


def _has_retrieval_producing_semantics(context: ProductIdentificationQueryContext) -> bool:
    return bool(context.requested_identifiers) or bool(context.required_constraints)


@dataclass(frozen=True, slots=True)
class ProductIdentificationQuery:
    """Single immutable query for retrieval derivation and verification semantics."""

    verification_context: ProductIdentificationQueryContext
    search_text: str | None = None

    def __post_init__(self) -> None:
        if type(self.verification_context) is not ProductIdentificationQueryContext:
            raise TypeError("verification_context must be ProductIdentificationQueryContext")
        if self.search_text is not None:
            if type(self.search_text) is not str:
                raise TypeError("search_text must be str or None")
            if not self.search_text.strip():
                raise ValueError("search_text must be non-empty when present")
        if not _has_retrieval_producing_semantics(self.verification_context) and self.search_text is None:
            raise ValueError(
                "ProductIdentificationQuery requires search_text, requested_identifiers, "
                "or required_constraints for retrieval"
            )
