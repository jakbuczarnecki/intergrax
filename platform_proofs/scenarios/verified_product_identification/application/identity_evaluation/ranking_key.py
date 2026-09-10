"""Lexicographic ranking key derivation — no weighted sum."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    ProductIdentityHypothesis,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    IdentityContradictionEvaluation,
    IdentityEvidenceProfile,
    IdentityHypothesisRankingKey,
    InternalPairCoverage,
    compare_internal_pair_coverage,
)


def build_ranking_key(
    hypothesis: ProductIdentityHypothesis,
    *,
    evidence_profile: IdentityEvidenceProfile,
    contradiction_evaluation: IdentityContradictionEvaluation,
) -> IdentityHypothesisRankingKey:
    """Derive frozen lexicographic ranking basis for one hypothesis."""

    return IdentityHypothesisRankingKey(
        has_internal_blocking_contradiction=len(contradiction_evaluation.internal_blocking) > 0,
        global_gtin_supported_pairs=evidence_profile.global_gtin_pair_coverage.supported_pair_count,
        global_gtin_possible_pairs=evidence_profile.global_gtin_pair_coverage.possible_pair_count,
        manufacturer_mpn_supported_pairs=(
            evidence_profile.manufacturer_mpn_pair_coverage.supported_pair_count
        ),
        manufacturer_mpn_possible_pairs=(
            evidence_profile.manufacturer_mpn_pair_coverage.possible_pair_count
        ),
        structured_distinct_key_count=len(evidence_profile.structured_attribute_keys),
        internal_nonblocking_count=len(contradiction_evaluation.internal_nonblocking),
        best_member_fused_rank=min(member.fused_rank for member in hypothesis.members),
        member_count=len(hypothesis.members),
        hypothesis_id=hypothesis.hypothesis_id,
    )


def compare_ranking_keys(
    left: IdentityHypothesisRankingKey,
    right: IdentityHypothesisRankingKey,
) -> int:
    """Deterministic ranking comparison — negative if ``left`` ranks higher."""

    if left.has_internal_blocking_contradiction != right.has_internal_blocking_contradiction:
        return 1 if left.has_internal_blocking_contradiction else -1

    gtin_cmp = compare_internal_pair_coverage(
        InternalPairCoverage(
            supported_pair_count=left.global_gtin_supported_pairs,
            possible_pair_count=left.global_gtin_possible_pairs,
        ),
        InternalPairCoverage(
            supported_pair_count=right.global_gtin_supported_pairs,
            possible_pair_count=right.global_gtin_possible_pairs,
        ),
    )
    if gtin_cmp != 0:
        return gtin_cmp

    mpn_cmp = compare_internal_pair_coverage(
        InternalPairCoverage(
            supported_pair_count=left.manufacturer_mpn_supported_pairs,
            possible_pair_count=left.manufacturer_mpn_possible_pairs,
        ),
        InternalPairCoverage(
            supported_pair_count=right.manufacturer_mpn_supported_pairs,
            possible_pair_count=right.manufacturer_mpn_possible_pairs,
        ),
    )
    if mpn_cmp != 0:
        return mpn_cmp

    if left.structured_distinct_key_count != right.structured_distinct_key_count:
        return (
            1
            if left.structured_distinct_key_count < right.structured_distinct_key_count
            else -1
        )

    if left.internal_nonblocking_count != right.internal_nonblocking_count:
        return -1 if left.internal_nonblocking_count < right.internal_nonblocking_count else 1

    if left.best_member_fused_rank != right.best_member_fused_rank:
        return -1 if left.best_member_fused_rank < right.best_member_fused_rank else 1

    if left.member_count != right.member_count:
        return -1 if left.member_count < right.member_count else 1

    if left.hypothesis_id < right.hypothesis_id:
        return -1
    if left.hypothesis_id > right.hypothesis_id:
        return 1
    return 0
