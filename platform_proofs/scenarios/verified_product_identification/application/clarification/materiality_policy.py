"""Material attribute allow-list for VPI clarification fixtures."""

from __future__ import annotations

from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    IdentityEvidenceType,
)


class ClarificationMaterialityPolicy(Protocol):
    def is_material_attribute(self, attribute_name: str) -> bool:
        """Return whether a structured attribute may drive identity clarification."""


_MATERIAL_ATTRIBUTE_KEYS = frozenset(
    {
        "capacity",
        "interface",
        "form_factor",
        "size",
        "generation",
        "voltage",
        "memory_type",
        "ecc",
        "brand",
        "mpn",
    }
)

_WEAK_EVIDENCE_TYPES = frozenset(
    {
        IdentityEvidenceType.TITLE_TOKEN_SUPPORT,
        IdentityEvidenceType.SEMANTIC_SUPPORT,
    }
)


class DeterministicClarificationMaterialityPolicy:
    """Scenario-owned allow-list — extend only with tested fixture attributes."""

    def is_material_attribute(self, attribute_name: str) -> bool:
        return attribute_name.casefold() in _MATERIAL_ATTRIBUTE_KEYS

    def is_weak_retrieval_attribute(self, attribute_name: str) -> bool:
        key = attribute_name.casefold()
        return key in {"title", "semantic", "bm25", "vector", "price", "seller"}


def is_weak_evidence_type(evidence_type: IdentityEvidenceType) -> bool:
    return evidence_type in _WEAK_EVIDENCE_TYPES
