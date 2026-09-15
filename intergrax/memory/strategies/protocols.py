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
from intergrax.memory.strategies.recall_models import (
    MemoryConflictDetectionRequest,
    MemoryConflictDetectionResult,
    MemoryConflictResolutionRequest,
    MemoryConflictResolutionResult,
    MemoryRankingRequest,
    MemoryRankingResult,
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


class MemoryRankingStrategy(Protocol):
    strategy_id: str

    def rank(self, request: MemoryRankingRequest) -> MemoryRankingResult:
        """Rank recall candidates deterministically."""


class MemoryConflictDetectionStrategy(Protocol):
    strategy_id: str

    def detect(self, request: MemoryConflictDetectionRequest) -> MemoryConflictDetectionResult:
        """Detect conflicts within a bounded ranked candidate set."""


class MemoryConflictResolutionStrategy(Protocol):
    strategy_id: str

    def resolve(self, request: MemoryConflictResolutionRequest) -> MemoryConflictResolutionResult:
        """Return typed resolution decisions without mutating records."""
