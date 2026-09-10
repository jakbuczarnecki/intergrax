"""Pairwise cross-offer identity evidence and contradiction derivation."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    RetrievalChannel,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
    source_ref_sort_key,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.contracts import (
    FusedOfferCandidate,
    OfferChannelEvidence,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    IdentityContradiction,
    IdentityContradictionType,
    IdentityEvidence,
    IdentityEvidenceProvenance,
    IdentityEvidenceStrengthClass,
    IdentityEvidenceType,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.profile import (
    IdentityStructuredAttribute,
    IdentityTypedIdentifier,
    SourceOfferIdentityProfile,
    brand_normalization_rule,
    identifier_normalization_rule,
    structured_normalization_rule,
)

_STRONG_IDENTIFIER_TYPES: tuple[ProductIdentifierType, ...] = (
    ProductIdentifierType.GTIN,
    ProductIdentifierType.MPN,
    ProductIdentifierType.SKU,
    ProductIdentifierType.PRODUCT_ID,
)


@dataclass(frozen=True, slots=True)
class OfferPairIdentityAssessment:
    """Bounded pairwise assessment used by grouping strategy."""

    left_source_ref: SourceRecordRef
    right_source_ref: SourceRecordRef
    evidence: tuple[IdentityEvidence, ...]
    contradictions: tuple[IdentityContradiction, ...]

    @property
    def has_strong_support(self) -> bool:
        return any(
            item.strength_class is IdentityEvidenceStrengthClass.STRONG for item in self.evidence
        )

    @property
    def has_contradiction(self) -> bool:
        return len(self.contradictions) > 0

    @property
    def is_grouping_compatible(self) -> bool:
        return self.has_strong_support and not self.has_contradiction


def assess_offer_pair(
    left_profile: SourceOfferIdentityProfile,
    right_profile: SourceOfferIdentityProfile,
    *,
    left_fused: FusedOfferCandidate,
    right_fused: FusedOfferCandidate,
) -> OfferPairIdentityAssessment:
    """Derive typed evidence and contradictions for one bounded offer pair."""

    ordered_refs = tuple(
        sorted(
            (left_profile.source_ref, right_profile.source_ref),
            key=source_ref_sort_key,
        )
    )
    left_ref, right_ref = ordered_refs
    left_is_first = left_profile.source_ref == left_ref
    left_data = left_profile if left_is_first else right_profile
    right_data = right_profile if left_is_first else left_profile
    left_candidate = left_fused if left_is_first else right_fused
    right_candidate = right_fused if left_is_first else left_fused

    evidence_items: list[IdentityEvidence] = []
    contradiction_items: list[IdentityContradiction] = []

    _assess_identifiers(
        left_data,
        right_data,
        ordered_refs=ordered_refs,
        evidence_items=evidence_items,
        contradiction_items=contradiction_items,
    )
    _assess_brand(
        left_data,
        right_data,
        ordered_refs=ordered_refs,
        evidence_items=evidence_items,
        contradiction_items=contradiction_items,
    )
    _assess_structured_attributes(
        left_data,
        right_data,
        ordered_refs=ordered_refs,
        evidence_items=evidence_items,
        contradiction_items=contradiction_items,
    )
    _assess_weak_retrieval_support(
        left_candidate,
        right_candidate,
        ordered_refs=ordered_refs,
        evidence_items=evidence_items,
    )

    return OfferPairIdentityAssessment(
        left_source_ref=ordered_refs[0],
        right_source_ref=ordered_refs[1],
        evidence=tuple(_sort_evidence(evidence_items)),
        contradictions=tuple(_sort_contradictions(contradiction_items)),
    )


def pair_has_grouping_eligibility(
    assessment: OfferPairIdentityAssessment,
    *,
    left_profile: SourceOfferIdentityProfile,
    right_profile: SourceOfferIdentityProfile,
) -> bool:
    """Discrete eligibility — strong identifier, MPN+brand, or multiple structured matches."""

    if assessment.has_contradiction:
        return False

    strong_identifiers = [
        item
        for item in assessment.evidence
        if item.evidence_type is IdentityEvidenceType.EXACT_IDENTIFIER_MATCH
        and item.strength_class is IdentityEvidenceStrengthClass.STRONG
    ]
    if strong_identifiers:
        return True

    has_mpn_match = any(
        item.evidence_type is IdentityEvidenceType.MODEL_NUMBER_MATCH
        and item.strength_class is IdentityEvidenceStrengthClass.STRONG
        for item in assessment.evidence
    )
    has_brand_conflict = any(
        item.contradiction_type is IdentityContradictionType.BRAND_CONFLICT
        for item in assessment.contradictions
    )
    has_brand_match = any(
        item.evidence_type is IdentityEvidenceType.BRAND_MATCH for item in assessment.evidence
    )
    brand_missing = left_profile.brand is None or right_profile.brand is None
    if has_mpn_match and not has_brand_conflict and (has_brand_match or brand_missing):
        return True

    structured_matches = [
        item
        for item in assessment.evidence
        if item.evidence_type is IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH
        and item.strength_class is IdentityEvidenceStrengthClass.STRONG
    ]
    unique_keys = {item.attribute_key.casefold() for item in structured_matches}
    return len(unique_keys) >= 2


def _assess_identifiers(
    left_profile: SourceOfferIdentityProfile,
    right_profile: SourceOfferIdentityProfile,
    *,
    ordered_refs: tuple[SourceRecordRef, SourceRecordRef],
    evidence_items: list[IdentityEvidence],
    contradiction_items: list[IdentityContradiction],
) -> None:
    for identifier_type in _STRONG_IDENTIFIER_TYPES:
        left_values = left_profile.identifiers_by_type(identifier_type)
        right_values = right_profile.identifiers_by_type(identifier_type)
        if not left_values and not right_values:
            continue
        if not left_values or not right_values:
            continue

        left_set = {item.normalized_value for item in left_values}
        right_set = {item.normalized_value for item in right_values}
        intersection = left_set.intersection(right_set)
        if intersection:
            matched_value = sorted(intersection)[0]
            left_field = _first_field_for_value(left_values, matched_value)
            right_field = _first_field_for_value(right_values, matched_value)
            evidence_type = (
                IdentityEvidenceType.MODEL_NUMBER_MATCH
                if identifier_type is ProductIdentifierType.MPN
                else IdentityEvidenceType.EXACT_IDENTIFIER_MATCH
            )
            evidence_items.append(
                IdentityEvidence(
                    evidence_type=evidence_type,
                    source_refs=ordered_refs,
                    attribute_key=identifier_type.value,
                    normalized_value=matched_value,
                    strength_class=IdentityEvidenceStrengthClass.STRONG,
                    identifier_type=identifier_type,
                    provenance=IdentityEvidenceProvenance(
                        left_source_ref=ordered_refs[0],
                        right_source_ref=ordered_refs[1],
                        source_field=f"{left_field}|{right_field}",
                        normalization_rule=identifier_normalization_rule(),
                    ),
                )
            )
            continue

        left_value = sorted(left_set)[0]
        right_value = sorted(right_set)[0]
        left_field = _first_field_for_value(left_values, left_value)
        right_field = _first_field_for_value(right_values, right_value)
        contradiction_type = (
            IdentityContradictionType.MODEL_NUMBER_CONFLICT
            if identifier_type is ProductIdentifierType.MPN
            else IdentityContradictionType.IDENTIFIER_CONFLICT
        )
        contradiction_items.append(
            IdentityContradiction(
                contradiction_type=contradiction_type,
                source_refs=ordered_refs,
                attribute_key=identifier_type.value,
                left_normalized_value=left_value,
                right_normalized_value=right_value,
                identifier_type=identifier_type,
                provenance=IdentityEvidenceProvenance(
                    left_source_ref=ordered_refs[0],
                    right_source_ref=ordered_refs[1],
                    source_field=f"{left_field}|{right_field}",
                    normalization_rule=identifier_normalization_rule(),
                ),
            )
        )


def _assess_brand(
    left_profile: SourceOfferIdentityProfile,
    right_profile: SourceOfferIdentityProfile,
    *,
    ordered_refs: tuple[SourceRecordRef, SourceRecordRef],
    evidence_items: list[IdentityEvidence],
    contradiction_items: list[IdentityContradiction],
) -> None:
    left_brand = left_profile.brand
    right_brand = right_profile.brand
    if left_brand is None or right_brand is None:
        return
    if left_brand == right_brand:
        evidence_items.append(
            IdentityEvidence(
                evidence_type=IdentityEvidenceType.BRAND_MATCH,
                source_refs=ordered_refs,
                attribute_key="brand",
                normalized_value=left_brand,
                strength_class=IdentityEvidenceStrengthClass.STRONG,
                provenance=IdentityEvidenceProvenance(
                    left_source_ref=ordered_refs[0],
                    right_source_ref=ordered_refs[1],
                    source_field="brand|brand",
                    normalization_rule=brand_normalization_rule(),
                ),
            )
        )
        return
    contradiction_items.append(
        IdentityContradiction(
            contradiction_type=IdentityContradictionType.BRAND_CONFLICT,
            source_refs=ordered_refs,
            attribute_key="brand",
            left_normalized_value=left_brand,
            right_normalized_value=right_brand,
            provenance=IdentityEvidenceProvenance(
                left_source_ref=ordered_refs[0],
                right_source_ref=ordered_refs[1],
                source_field="brand|brand",
                normalization_rule=brand_normalization_rule(),
            ),
        )
    )


def _assess_structured_attributes(
    left_profile: SourceOfferIdentityProfile,
    right_profile: SourceOfferIdentityProfile,
    *,
    ordered_refs: tuple[SourceRecordRef, SourceRecordRef],
    evidence_items: list[IdentityEvidence],
    contradiction_items: list[IdentityContradiction],
) -> None:
    left_by_key = _attributes_by_canonical_key(left_profile.structured_attributes)
    right_by_key = _attributes_by_canonical_key(right_profile.structured_attributes)
    shared_keys = sorted(set(left_by_key).intersection(right_by_key))
    for canonical_key in shared_keys:
        left_attribute = left_by_key[canonical_key]
        right_attribute = right_by_key[canonical_key]
        left_value = left_attribute.normalized_text_value.casefold()
        right_value = right_attribute.normalized_text_value.casefold()
        if left_value == right_value:
            evidence_items.append(
                IdentityEvidence(
                    evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                    source_refs=ordered_refs,
                    attribute_key=canonical_key,
                    normalized_value=left_attribute.normalized_text_value,
                    strength_class=IdentityEvidenceStrengthClass.STRONG,
                    provenance=IdentityEvidenceProvenance(
                        left_source_ref=ordered_refs[0],
                        right_source_ref=ordered_refs[1],
                        source_field=(
                            f"{left_attribute.source_field}/{left_attribute.source_key}|"
                            f"{right_attribute.source_field}/{right_attribute.source_key}"
                        ),
                        normalization_rule=structured_normalization_rule(),
                    ),
                )
            )
            continue
        contradiction_items.append(
            IdentityContradiction(
                contradiction_type=IdentityContradictionType.STRUCTURED_ATTRIBUTE_CONFLICT,
                source_refs=ordered_refs,
                attribute_key=canonical_key,
                left_normalized_value=left_attribute.normalized_text_value,
                right_normalized_value=right_attribute.normalized_text_value,
                provenance=IdentityEvidenceProvenance(
                    left_source_ref=ordered_refs[0],
                    right_source_ref=ordered_refs[1],
                    source_field=(
                        f"{left_attribute.source_field}/{left_attribute.source_key}|"
                        f"{right_attribute.source_field}/{right_attribute.source_key}"
                    ),
                    normalization_rule=structured_normalization_rule(),
                ),
            )
        )


def _assess_weak_retrieval_support(
    left_candidate: FusedOfferCandidate,
    right_candidate: FusedOfferCandidate,
    *,
    ordered_refs: tuple[SourceRecordRef, SourceRecordRef],
    evidence_items: list[IdentityEvidence],
) -> None:
    left_channels = {item.channel for item in left_candidate.evidence}
    right_channels = {item.channel for item in right_candidate.evidence}
    if RetrievalChannel.LEXICAL in left_channels and RetrievalChannel.LEXICAL in right_channels:
        evidence_items.append(
            IdentityEvidence(
                evidence_type=IdentityEvidenceType.TITLE_TOKEN_SUPPORT,
                source_refs=ordered_refs,
                attribute_key="retrieval_channel",
                normalized_value=RetrievalChannel.LEXICAL.value,
                strength_class=IdentityEvidenceStrengthClass.WEAK,
                provenance=IdentityEvidenceProvenance(
                    left_source_ref=ordered_refs[0],
                    right_source_ref=ordered_refs[1],
                    source_field="fusion.evidence.channel",
                    normalization_rule="fusion_context/v1",
                ),
            )
        )
    if RetrievalChannel.VECTOR in left_channels and RetrievalChannel.VECTOR in right_channels:
        evidence_items.append(
            IdentityEvidence(
                evidence_type=IdentityEvidenceType.SEMANTIC_SUPPORT,
                source_refs=ordered_refs,
                attribute_key="retrieval_channel",
                normalized_value=RetrievalChannel.VECTOR.value,
                strength_class=IdentityEvidenceStrengthClass.WEAK,
                provenance=IdentityEvidenceProvenance(
                    left_source_ref=ordered_refs[0],
                    right_source_ref=ordered_refs[1],
                    source_field="fusion.evidence.channel",
                    normalization_rule="fusion_context/v1",
                ),
            )
        )


def _attributes_by_canonical_key(
    attributes: tuple[IdentityStructuredAttribute, ...],
) -> dict[str, IdentityStructuredAttribute]:
    indexed: dict[str, IdentityStructuredAttribute] = {}
    for attribute in attributes:
        key = attribute.canonical_key.casefold()
        if key not in indexed:
            indexed[key] = attribute
    return indexed


def _first_field_for_value(
    identifiers: tuple[IdentityTypedIdentifier, ...],
    normalized_value: str,
) -> str:
    for identifier in identifiers:
        if identifier.normalized_value == normalized_value:
            return identifier.source_field
    return "unknown"


def _sort_evidence(items: list[IdentityEvidence]) -> tuple[IdentityEvidence, ...]:
    return tuple(
        sorted(
            items,
            key=lambda item: (
                source_ref_sort_key(item.source_refs[0]),
                source_ref_sort_key(item.source_refs[1]),
                item.evidence_type.value,
                item.attribute_key.casefold(),
                item.normalized_value,
            ),
        )
    )


def _sort_contradictions(
    items: list[IdentityContradiction],
) -> tuple[IdentityContradiction, ...]:
    return tuple(
        sorted(
            items,
            key=lambda item: (
                source_ref_sort_key(item.source_refs[0]),
                source_ref_sort_key(item.source_refs[1]),
                item.contradiction_type.value,
                item.attribute_key.casefold(),
                item.left_normalized_value,
                item.right_normalized_value,
            ),
        )
    )
