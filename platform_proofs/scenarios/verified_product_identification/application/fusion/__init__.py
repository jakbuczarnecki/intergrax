"""Offer-level candidate fusion — ranks offers, does not verify products."""

from platform_proofs.scenarios.verified_product_identification.application.fusion.composition import (
    build_offer_candidate_fusion,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.contracts import (
    FusedOfferCandidate,
    FusedOfferCandidateCollection,
    OfferCandidateFusionRequest,
    OfferChannelEvidence,
    OfferFusionConfiguration,
    reciprocal_rank_contribution,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.errors import (
    OfferCandidateFusionError,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.service import (
    OfferCandidateFusionService,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.strategy import (
    OfferCandidateFusionStrategy,
    ReciprocalRankFusionStrategy,
)

__all__ = (
    "FusedOfferCandidate",
    "FusedOfferCandidateCollection",
    "OfferCandidateFusionError",
    "OfferCandidateFusionRequest",
    "OfferCandidateFusionService",
    "OfferCandidateFusionStrategy",
    "OfferChannelEvidence",
    "OfferFusionConfiguration",
    "ReciprocalRankFusionStrategy",
    "build_offer_candidate_fusion",
    "reciprocal_rank_contribution",
)
