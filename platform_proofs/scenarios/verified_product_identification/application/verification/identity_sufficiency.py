"""Material identity evidence sufficiency — not ranking scores."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierIdentityScope,
    ProductIdentifierType,
    identity_scope_for_identifier_type,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    IdentityEvidenceStrengthClass,
    IdentityEvidenceType,
    ProductIdentityHypothesis,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    IdentityEvidenceProfile,
)


def identity_evidence_materially_sufficient(
    hypothesis: ProductIdentityHypothesis,
    evidence_profile: IdentityEvidenceProfile,
) -> bool:
    if evidence_profile.global_gtin_pair_coverage.supported_pair_count > 0:
        return True
    if evidence_profile.manufacturer_mpn_pair_coverage.supported_pair_count > 0:
        return True
    if _has_strong_brand_only(hypothesis):
        return False
    if _has_only_weak_context(evidence_profile):
        return False
    if _has_source_local_identifier_only(hypothesis):
        return False
    return False


def _has_strong_brand_only(hypothesis: ProductIdentityHypothesis) -> bool:
    strong_rows = [
        item
        for item in hypothesis.evidence
        if item.strength_class is IdentityEvidenceStrengthClass.STRONG
    ]
    if not strong_rows:
        return False
    return all(
        item.evidence_type is IdentityEvidenceType.BRAND_MATCH for item in strong_rows
    )


def _has_only_weak_context(evidence_profile: IdentityEvidenceProfile) -> bool:
    return (
        evidence_profile.weak_context_pair_coverage.supported_pair_count > 0
        and evidence_profile.global_gtin_pair_coverage.supported_pair_count == 0
        and evidence_profile.manufacturer_mpn_pair_coverage.supported_pair_count == 0
        and not evidence_profile.structured_attribute_keys
    )


def _has_source_local_identifier_only(hypothesis: ProductIdentityHypothesis) -> bool:
    identifier_rows = [
        item
        for item in hypothesis.evidence
        if item.evidence_type is IdentityEvidenceType.EXACT_IDENTIFIER_MATCH
        and item.strength_class is IdentityEvidenceStrengthClass.STRONG
        and item.identifier_type is not None
    ]
    if not identifier_rows:
        return False
    for item in identifier_rows:
        identifier_type = item.identifier_type
        if identifier_type is None:
            continue
        scope = identity_scope_for_identifier_type(identifier_type)
        if scope is ProductIdentifierIdentityScope.GLOBAL:
            return False
        if scope is ProductIdentifierIdentityScope.MANUFACTURER_SCOPED:
            return False
    return True
