"""Scenario-owned fusion composition helpers."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.fusion.contracts import (
    OfferFusionConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.service import (
    OfferCandidateFusionService,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.strategy import (
    ReciprocalRankFusionStrategy,
)


def build_offer_candidate_fusion(
    *,
    configuration: OfferFusionConfiguration | None = None,
) -> OfferCandidateFusionService:
    """Construct the canonical RRF-backed offer fusion service."""

    config = configuration or OfferFusionConfiguration()
    return OfferCandidateFusionService(
        strategy=ReciprocalRankFusionStrategy(configuration=config),
    )
