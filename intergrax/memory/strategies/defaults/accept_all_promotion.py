# © Artur Czarnecki. All rights reserved.

"""Default promotion strategy: persist all deduplicated candidates (MEM-ENT-4)."""

from __future__ import annotations

from intergrax.memory.strategies.models import (
    MemoryPromotionAction,
    MemoryPromotionDecision,
    MemoryPromotionRequest,
    MemoryPromotionResult,
)


class AcceptAllMemoryPromotionStrategy:
    strategy_id = "builtin.memory.promotion.accept_all"

    def promote(self, request: MemoryPromotionRequest) -> MemoryPromotionResult:
        decisions = tuple(
            MemoryPromotionDecision(
                entry=entry,
                action=MemoryPromotionAction.PROMOTE,
                reason="accept_all_default",
            )
            for entry in request.entries
        )
        return MemoryPromotionResult(decisions=decisions)
