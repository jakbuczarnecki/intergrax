"""Lexicographic ranking key derivation — no weighted sum."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    ProductIdentityHypothesis,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    IdentityContradictionEvaluation,
    IdentityEvidenceProfile,
    IdentityHypothesisRankingKey,
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


def ranking_sort_key(key: IdentityHypothesisRankingKey) -> tuple[bool, int, int, int, int, int, int, int, int, str]:
    """Canonical lexicographic ordering — lower tuple ranks higher."""

    return (
        key.has_internal_blocking_contradiction,
        -key.global_gtin_supported_pairs,
        -key.global_gtin_possible_pairs,
        -key.manufacturer_mpn_supported_pairs,
        -key.manufacturer_mpn_possible_pairs,
        -key.structured_distinct_key_count,
        key.internal_nonblocking_count,
        key.best_member_fused_rank,
        key.member_count,
        key.hypothesis_id,
    )
