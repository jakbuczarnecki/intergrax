# © Artur Czarnecki. All rights reserved.

"""Default memory strategy composition (MEM-ENT-4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.memory.strategies.defaults.accept_all_promotion import AcceptAllMemoryPromotionStrategy
from intergrax.memory.strategies.defaults.llm_extraction import LlmMemoryExtractionStrategy
from intergrax.memory.strategies.defaults.sequence_deduplication import (
    SequenceMatcherDeduplicationConfig,
    SequenceMatcherMemoryDeduplicationStrategy,
)
from intergrax.memory.strategies.protocols import (
    MemoryDeduplicationStrategy,
    MemoryExtractionStrategy,
    MemoryPromotionStrategy,
)


@dataclass(frozen=True)
class MemoryStrategySet:
    extraction: MemoryExtractionStrategy
    deduplication: MemoryDeduplicationStrategy
    promotion: MemoryPromotionStrategy


def build_default_memory_strategies(
    llm: LLMAdapter,
    *,
    deduplication_config: SequenceMatcherDeduplicationConfig | None = None,
) -> MemoryStrategySet:
    return MemoryStrategySet(
        extraction=LlmMemoryExtractionStrategy(llm),
        deduplication=SequenceMatcherMemoryDeduplicationStrategy(deduplication_config),
        promotion=AcceptAllMemoryPromotionStrategy(),
    )
