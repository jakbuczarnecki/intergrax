"""Composition helpers for identity hypothesis evaluation."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.service import (
    IdentityHypothesisEvaluationService,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.strategy import (
    DeterministicEvidenceIdentityRankingStrategy,
    IdentityHypothesisRankingStrategy,
)


def build_identity_hypothesis_evaluation_service(
    *,
    strategy: IdentityHypothesisRankingStrategy | None = None,
) -> IdentityHypothesisEvaluationService:
    """Construct evaluation service with injectable ranking policy."""

    resolved_strategy = (
        strategy if strategy is not None else DeterministicEvidenceIdentityRankingStrategy()
    )
    return IdentityHypothesisEvaluationService(strategy=resolved_strategy)
