# © Artur Czarnecki. All rights reserved.

"""Memory strategy protocol contracts (MEM-ENT-4)."""

from __future__ import annotations

from typing import Protocol

from intergrax.memory.strategies.models import (
    MemoryDeduplicationRequest,
    MemoryDeduplicationResult,
    MemoryExtractionRequest,
    MemoryExtractionResult,
    MemoryPromotionRequest,
    MemoryPromotionResult,
)


class MemoryExtractionStrategy(Protocol):
    strategy_id: str

    async def extract(self, request: MemoryExtractionRequest) -> MemoryExtractionResult:
        """Extract memory candidates from session history."""


class MemoryDeduplicationStrategy(Protocol):
    strategy_id: str

    def deduplicate(self, request: MemoryDeduplicationRequest) -> MemoryDeduplicationResult:
        """Filter near-duplicate candidates against existing profile memory."""


class MemoryPromotionStrategy(Protocol):
    strategy_id: str

    def promote(self, request: MemoryPromotionRequest) -> MemoryPromotionResult:
        """Decide which consolidated entries should be written to long-term memory."""
