"""Immutable identity hypothesis evaluation contracts — reranking, not verification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    IdentityContradiction,
    IdentityEvidence,
    ProductIdentityHypothesis,
    ProductIdentityHypothesisCollection,
)


class EvidenceRelationScope(StrEnum):
    """Whether supporting evidence relates to hypothesis members."""

    INTERNAL = "internal"
    EXTERNAL = "external"
    INVALID = "invalid"


class ContradictionRelationScope(StrEnum):
    """Whether a contradiction relates to hypothesis members."""

    INTERNAL = "internal"
    EXTERNAL = "external"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class InternalPairCoverage:
    """Deterministic pair coverage — not a confidence ratio."""

    supported_pair_count: int
    possible_pair_count: int

    def __post_init__(self) -> None:
        if type(self.supported_pair_count) is not int or self.supported_pair_count < 0:
            raise ValueError("supported_pair_count must be a non-negative int")
        if type(self.possible_pair_count) is not int or self.possible_pair_count < 0:
            raise ValueError("possible_pair_count must be a non-negative int")
        if self.supported_pair_count > self.possible_pair_count:
            raise ValueError("supported_pair_count cannot exceed possible_pair_count")


def compare_internal_pair_coverage(
    left: InternalPairCoverage,
    right: InternalPairCoverage,
) -> int:
    """Compare coverage quality using exact rational ordering.

    Returns -1 if ``left`` ranks higher, 0 if coverage tier ties, 1 if ``right`` ranks higher.
    ``possible_pair_count`` is denominator context only — never positive evidence.
    Singleton ``0/0`` carries no positive support and does not outrank actual supported coverage.
    """

    if left.supported_pair_count > 0 and right.supported_pair_count == 0:
        return -1
    if left.supported_pair_count == 0 and right.supported_pair_count > 0:
        return 1
    if left.supported_pair_count == 0 and right.supported_pair_count == 0:
        return 0

    left_cross = left.supported_pair_count * right.possible_pair_count
    right_cross = right.supported_pair_count * left.possible_pair_count
    if left_cross > right_cross:
        return -1
    if left_cross < right_cross:
        return 1

    if left.supported_pair_count > right.supported_pair_count:
        return -1
    if left.supported_pair_count < right.supported_pair_count:
        return 1
    return 0


@dataclass(frozen=True, slots=True)
class IdentityEvidenceProfile:
    """Typed internal identity evidence summary — no raw row counts."""

    global_gtin_pair_coverage: InternalPairCoverage
    manufacturer_mpn_pair_coverage: InternalPairCoverage
    structured_attribute_keys: tuple[str, ...]
    weak_context_pair_coverage: InternalPairCoverage

    def __post_init__(self) -> None:
        if not isinstance(self.structured_attribute_keys, tuple):
            raise TypeError("structured_attribute_keys must be a tuple")


@dataclass(frozen=True, slots=True)
class IdentityContradictionEvaluation:
    """Scoped contradiction summary — traceable original rows preserved."""

    internal_blocking: tuple[IdentityContradiction, ...]
    internal_nonblocking: tuple[IdentityContradiction, ...]
    external_blocking: tuple[IdentityContradiction, ...]
    external_nonblocking: tuple[IdentityContradiction, ...]
    external_separation_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.internal_blocking, tuple):
            raise TypeError("internal_blocking must be a tuple")
        if not isinstance(self.internal_nonblocking, tuple):
            raise TypeError("internal_nonblocking must be a tuple")
        if not isinstance(self.external_blocking, tuple):
            raise TypeError("external_blocking must be a tuple")
        if not isinstance(self.external_nonblocking, tuple):
            raise TypeError("external_nonblocking must be a tuple")
        if type(self.external_separation_count) is not int or self.external_separation_count < 0:
            raise ValueError("external_separation_count must be a non-negative int")


@dataclass(frozen=True, slots=True)
class IdentityHypothesisRankingKey:
    """Lexicographic ranking basis — frozen, interpretable, no scalar score."""

    has_internal_blocking_contradiction: bool
    global_gtin_supported_pairs: int
    global_gtin_possible_pairs: int
    manufacturer_mpn_supported_pairs: int
    manufacturer_mpn_possible_pairs: int
    structured_distinct_key_count: int
    internal_nonblocking_count: int
    best_member_fused_rank: int
    member_count: int
    hypothesis_id: str

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id must be non-empty")
        if type(self.best_member_fused_rank) is not int or self.best_member_fused_rank < 0:
            raise ValueError("best_member_fused_rank must be a non-negative int")
        if type(self.member_count) is not int or self.member_count < 1:
            raise ValueError("member_count must be a positive int")


@dataclass(frozen=True, slots=True)
class IdentityHypothesisEvaluationBundle:
    """Per-hypothesis evaluation inputs passed to ranking policy."""

    hypothesis: ProductIdentityHypothesis
    evidence_profile: IdentityEvidenceProfile
    contradiction_evaluation: IdentityContradictionEvaluation
    ranking_key: IdentityHypothesisRankingKey


@dataclass(frozen=True, slots=True)
class EvaluatedIdentityHypothesis:
    """One reranked hypothesis with structured evidence and contradiction state."""

    hypothesis: ProductIdentityHypothesis
    reranked_position: int
    evidence_profile: IdentityEvidenceProfile
    contradiction_evaluation: IdentityContradictionEvaluation
    ranking_key: IdentityHypothesisRankingKey

    def __post_init__(self) -> None:
        if type(self.reranked_position) is not int or self.reranked_position < 0:
            raise ValueError("reranked_position must be a non-negative int")


@dataclass(frozen=True, slots=True)
class RankedIdentityHypothesisCollection:
    """Deterministic reranked evaluated hypotheses — top rank is not verified truth."""

    hypotheses: tuple[EvaluatedIdentityHypothesis, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.hypotheses, tuple):
            raise TypeError("hypotheses must be a tuple")
        for position, evaluated in enumerate(self.hypotheses):
            if evaluated.reranked_position != position:
                raise ValueError("reranked positions must be contiguous from zero")


@dataclass(frozen=True, slots=True)
class IdentityHypothesisEvaluationRequest:
    """Validated evaluation request over 5C8 hypothesis output."""

    hypotheses: ProductIdentityHypothesisCollection

    def __post_init__(self) -> None:
        if not isinstance(self.hypotheses.hypotheses, tuple):
            raise TypeError("hypotheses must be a tuple")
