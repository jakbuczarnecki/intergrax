# © Artur Czarnecki. All rights reserved.

"""Pluggable memory strategy SPI (MEM-ENT-4)."""

from intergrax.memory.strategies.bundle import MemoryStrategySet, build_default_memory_strategies
from intergrax.memory.strategies.errors import (
    MemoryStrategyContractError,
    MemoryStrategyError,
    MemoryStrategyProviderError,
)
from intergrax.memory.strategies.models import (
    MemoryCandidate,
    MemoryDeduplicationRequest,
    MemoryDeduplicationResult,
    MemoryExtractionRequest,
    MemoryExtractionResult,
    MemoryPromotionAction,
    MemoryPromotionDecision,
    MemoryPromotionRequest,
    MemoryPromotionResult,
)
from intergrax.memory.strategies.protocols import (
    MemoryConflictDetectionStrategy,
    MemoryConflictResolutionStrategy,
    MemoryDeduplicationStrategy,
    MemoryExtractionStrategy,
    MemoryPromotionStrategy,
    MemoryRankingStrategy,
)
from intergrax.memory.strategies.recall_models import (
    MemoryConflict,
    MemoryConflictKind,
    MemoryConflictResolutionAction,
    MemoryRankingScore,
    MemoryRecallCandidate,
    MemoryRecallReasonCode,
    MemoryRetrievalSource,
    MemorySupersessionIntent,
)

__all__ = [
    "MemoryCandidate",
    "MemoryDeduplicationRequest",
    "MemoryDeduplicationResult",
    "MemoryDeduplicationStrategy",
    "MemoryExtractionRequest",
    "MemoryExtractionResult",
    "MemoryExtractionStrategy",
    "MemoryPromotionAction",
    "MemoryPromotionDecision",
    "MemoryPromotionRequest",
    "MemoryPromotionResult",
    "MemoryPromotionStrategy",
    "MemoryConflict",
    "MemoryConflictDetectionStrategy",
    "MemoryConflictKind",
    "MemoryConflictResolutionAction",
    "MemoryConflictResolutionStrategy",
    "MemoryRankingScore",
    "MemoryRankingStrategy",
    "MemoryRecallCandidate",
    "MemoryRecallReasonCode",
    "MemoryRetrievalSource",
    "MemorySupersessionIntent",
    "MemoryStrategyContractError",
    "MemoryStrategyError",
    "MemoryStrategyProviderError",
    "MemoryStrategySet",
    "build_default_memory_strategies",
]
