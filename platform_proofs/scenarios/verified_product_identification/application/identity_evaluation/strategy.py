"""Pluggable identity hypothesis ranking strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    EvaluatedIdentityHypothesis,
    IdentityHypothesisEvaluationBundle,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.ranking_key import (
    ranking_sort_key,
)


class IdentityHypothesisRankingStrategy(Protocol):
    """Strategy seam for deterministic hypothesis reranking."""

    def rank(
        self,
        bundles: tuple[IdentityHypothesisEvaluationBundle, ...],
    ) -> tuple[EvaluatedIdentityHypothesis, ...]:
        ...


@dataclass(frozen=True, slots=True)
class DeterministicEvidenceIdentityRankingStrategy:
    """Canonical lexicographic evidence ranking — reranking, not verification."""

    def rank(
        self,
        bundles: tuple[IdentityHypothesisEvaluationBundle, ...],
    ) -> tuple[EvaluatedIdentityHypothesis, ...]:
        ordered = tuple(
            sorted(
                bundles,
                key=lambda bundle: ranking_sort_key(bundle.ranking_key),
            )
        )
        return tuple(
            EvaluatedIdentityHypothesis(
                hypothesis=bundle.hypothesis,
                reranked_position=position,
                evidence_profile=bundle.evidence_profile,
                contradiction_evaluation=bundle.contradiction_evaluation,
                ranking_key=bundle.ranking_key,
            )
            for position, bundle in enumerate(ordered)
        )
