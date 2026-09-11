"""Material identity evidence sufficiency — not ranking scores."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    ProductIdentificationQueryContext,
)
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
from platform_proofs.scenarios.verified_product_identification.application.verification.direct_source_evidence import (
    direct_global_gtin_supported,
    direct_manufacturer_mpn_present,
    evaluate_requested_identifier,
    facts_for_hypothesis,
)


def identity_evidence_materially_sufficient(
    hypothesis: ProductIdentityHypothesis,
    evidence_profile: IdentityEvidenceProfile,
    query_context: ProductIdentificationQueryContext,
) -> bool:
    facts = facts_for_hypothesis(hypothesis)

    requested_global = [
        item
        for item in query_context.requested_identifiers
        if identity_scope_for_identifier_type(item.identifier_type)
        is ProductIdentifierIdentityScope.GLOBAL
    ]
    requested_manufacturer = [
        item
        for item in query_context.requested_identifiers
        if identity_scope_for_identifier_type(item.identifier_type)
        is ProductIdentifierIdentityScope.MANUFACTURER_SCOPED
    ]

    if requested_global or requested_manufacturer:
        for requested in query_context.requested_identifiers:
            scope = identity_scope_for_identifier_type(requested.identifier_type)
            if scope is ProductIdentifierIdentityScope.SOURCE_LOCAL:
                continue
            status, _, bad_row, _ = evaluate_requested_identifier(
                requested=requested,
                facts=facts,
            )
            if bad_row is not None or status in ("missing", "contradicted"):
                return False
        return True

    if _has_strong_brand_only(hypothesis):
        return False
    if _has_only_weak_context(evidence_profile):
        return False
    if _has_source_local_identifier_only(hypothesis):
        return False

    for requested in query_context.requested_identifiers:
        if requested.identifier_type is ProductIdentifierType.GTIN:
            if direct_global_gtin_supported(requested_gtin=requested, facts=facts):
                return True

    if evidence_profile.global_gtin_pair_coverage.supported_pair_count > 0:
        return True
    if evidence_profile.manufacturer_mpn_pair_coverage.supported_pair_count > 0:
        return True
    if direct_manufacturer_mpn_present(facts):
        return True
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
