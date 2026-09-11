"""Derive typed internal identity evidence profiles from hypothesis rows."""

from __future__ import annotations

from collections.abc import Callable

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
    source_ref_sort_key,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    IdentityEvidence,
    IdentityEvidenceStrengthClass,
    IdentityEvidenceType,
    ProductIdentityHypothesis,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    EvidenceRelationScope,
    IdentityEvidenceProfile,
    InternalPairCoverage,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.scope import (
    classify_evidence_scope,
)


def possible_internal_pair_count(member_count: int) -> int:
    """Return M*(M-1)/2 for member count M; singleton yields zero."""

    if type(member_count) is not int or member_count < 1:
        raise ValueError("member_count must be a positive int")
    return member_count * (member_count - 1) // 2


def build_identity_evidence_profile(
    hypothesis: ProductIdentityHypothesis,
) -> IdentityEvidenceProfile:
    """Compute normalized internal evidence coverage without raw row-count bias."""

    member_refs = frozenset(member.source_ref for member in hypothesis.members)
    possible_pairs = possible_internal_pair_count(len(hypothesis.members))

    deduped_internal = _dedupe_internal_evidence(hypothesis.evidence, member_refs=member_refs)

    gtin_pairs = _supported_member_pairs(
        deduped_internal,
        predicate=_is_global_gtin_support,
    )
    mpn_pairs = _supported_member_pairs(
        deduped_internal,
        predicate=_is_manufacturer_mpn_support,
    )
    weak_pairs = _supported_member_pairs(
        deduped_internal,
        predicate=_is_weak_context_support,
    )
    structured_keys = _distinct_structured_keys(deduped_internal)

    return IdentityEvidenceProfile(
        global_gtin_pair_coverage=InternalPairCoverage(
            supported_pair_count=len(gtin_pairs),
            possible_pair_count=possible_pairs,
        ),
        manufacturer_mpn_pair_coverage=InternalPairCoverage(
            supported_pair_count=len(mpn_pairs),
            possible_pair_count=possible_pairs,
        ),
        structured_attribute_keys=structured_keys,
        weak_context_pair_coverage=InternalPairCoverage(
            supported_pair_count=len(weak_pairs),
            possible_pair_count=possible_pairs,
        ),
    )


def _dedupe_internal_evidence(
    evidence_rows: tuple[IdentityEvidence, ...],
    *,
    member_refs: frozenset[SourceRecordRef],
) -> tuple[IdentityEvidence, ...]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    deduped: list[IdentityEvidence] = []
    for item in evidence_rows:
        if classify_evidence_scope(item, member_refs=member_refs) is not EvidenceRelationScope.INTERNAL:
            continue
        key = _evidence_fact_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return tuple(
        sorted(
            deduped,
            key=lambda item: (
                source_ref_sort_key(item.source_refs[0]),
                source_ref_sort_key(item.source_refs[1]),
                item.evidence_type.value,
                item.attribute_key.casefold(),
                item.normalized_value,
            ),
        )
    )


def _evidence_fact_key(item: IdentityEvidence) -> tuple[str, str, str, str, str, str]:
    identifier_type = item.identifier_type.value if item.identifier_type is not None else ""
    return (
        item.evidence_type.value,
        source_ref_sort_key(item.source_refs[0]),
        source_ref_sort_key(item.source_refs[1]),
        item.attribute_key,
        item.normalized_value,
        identifier_type,
    )


def _member_pair_key(
    left_ref: SourceRecordRef,
    right_ref: SourceRecordRef,
) -> tuple[str, str]:
    ordered = tuple(sorted((left_ref, right_ref), key=source_ref_sort_key))
    return source_ref_sort_key(ordered[0]), source_ref_sort_key(ordered[1])


def _supported_member_pairs(
    evidence_rows: tuple[IdentityEvidence, ...],
    *,
    predicate: Callable[[IdentityEvidence], bool],
) -> frozenset[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for item in evidence_rows:
        if not predicate(item):
            continue
        pairs.add(_member_pair_key(item.source_refs[0], item.source_refs[1]))
    return frozenset(pairs)


def _is_global_gtin_support(item: IdentityEvidence) -> bool:
    return (
        item.evidence_type is IdentityEvidenceType.EXACT_IDENTIFIER_MATCH
        and item.strength_class is IdentityEvidenceStrengthClass.STRONG
        and item.identifier_type is ProductIdentifierType.GTIN
    )


def _is_manufacturer_mpn_support(item: IdentityEvidence) -> bool:
    return (
        item.evidence_type is IdentityEvidenceType.MODEL_NUMBER_MATCH
        and item.strength_class is IdentityEvidenceStrengthClass.STRONG
        and item.identifier_type is ProductIdentifierType.MPN
    )


def _is_weak_context_support(item: IdentityEvidence) -> bool:
    return item.evidence_type in (
        IdentityEvidenceType.TITLE_TOKEN_SUPPORT,
        IdentityEvidenceType.SEMANTIC_SUPPORT,
    )


def _distinct_structured_keys(evidence_rows: tuple[IdentityEvidence, ...]) -> tuple[str, ...]:
    keys: set[str] = set()
    for item in evidence_rows:
        if item.evidence_type is not IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH:
            continue
        if item.strength_class is not IdentityEvidenceStrengthClass.STRONG:
            continue
        keys.add(item.attribute_key.casefold())
    return tuple(sorted(keys))
