# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Platform rank fusion (RRF and replaceable strategies)."""

from intergrax.rag.retrieval.fusion.contracts import (
    RECIPROCAL_RANK_FUSION_STRATEGY_ID,
    FusedRankedCandidate,
    RankFusionChannelEvidence,
    RankFusionConfiguration,
    RankFusionResult,
    RankFusionStrategyPort,
    RankedRetrievalCandidate,
    RankedRetrievalChannel,
)
from intergrax.rag.retrieval.fusion.errors import RankFusionContractError
from intergrax.rag.retrieval.fusion.reciprocal_rank import (
    reciprocal_rank_contribution,
    reciprocal_rank_fusion,
)
from intergrax.rag.retrieval.fusion.reciprocal_rank_strategy import (
    ReciprocalRankFusionStrategy,
)

__all__ = [
    "RECIPROCAL_RANK_FUSION_STRATEGY_ID",
    "FusedRankedCandidate",
    "RankFusionChannelEvidence",
    "RankFusionConfiguration",
    "RankFusionContractError",
    "RankFusionResult",
    "RankFusionStrategyPort",
    "RankedRetrievalCandidate",
    "RankedRetrievalChannel",
    "ReciprocalRankFusionStrategy",
    "reciprocal_rank_contribution",
    "reciprocal_rank_fusion",
]
