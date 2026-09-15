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
    MemoryDeduplicationStrategy,
    MemoryExtractionStrategy,
    MemoryPromotionStrategy,
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
    "MemoryStrategyContractError",
    "MemoryStrategyError",
    "MemoryStrategyProviderError",
    "MemoryStrategySet",
    "build_default_memory_strategies",
]
