# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Learning repository port for adaptive healing (SELF-HEALING R4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.adaptive.context import (
    AdaptiveRollbackHistoryEntry,
    AdaptiveValidationQualitySignal,
)
from intergrax.contracts.self_healing.context import SelfHealingHistoricalOutcome


@runtime_checkable
class AdaptiveHealingLearningRepository(Protocol):
    def historical_outcomes_for_tenant(self, tenant_id: str) -> tuple[SelfHealingHistoricalOutcome, ...]: ...

    def validation_quality_for_tenant(self, tenant_id: str) -> tuple[AdaptiveValidationQualitySignal, ...]: ...

    def rollback_history_for_tenant(self, tenant_id: str) -> tuple[AdaptiveRollbackHistoryEntry, ...]: ...

    def context_similarity_refs_for_tenant(self, tenant_id: str) -> tuple[str, ...]: ...


__all__ = ["AdaptiveHealingLearningRepository"]
