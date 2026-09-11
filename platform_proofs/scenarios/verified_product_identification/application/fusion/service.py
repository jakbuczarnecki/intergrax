"""Offer-level candidate fusion service boundary."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.fusion.contracts import (
    FusedOfferCandidateCollection,
    OfferCandidateFusionRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.strategy import (
    OfferCandidateFusionStrategy,
)


@dataclass(frozen=True, slots=True)
class OfferCandidateFusionService:
    """Validate fusion requests and delegate to an injected strategy."""

    strategy: OfferCandidateFusionStrategy

    def fuse(self, request: OfferCandidateFusionRequest) -> FusedOfferCandidateCollection:
        return self.strategy.fuse(request.candidates, limit=request.limit)
