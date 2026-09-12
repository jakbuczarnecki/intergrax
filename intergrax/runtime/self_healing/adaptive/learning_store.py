# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""In-memory adaptive learning repository (SELF-HEALING R4)."""

from __future__ import annotations

from intergrax.contracts.self_healing.adaptive.context import (
    AdaptiveRollbackHistoryEntry,
    AdaptiveValidationQualitySignal,
)
from intergrax.contracts.self_healing.context import SelfHealingHistoricalOutcome


class InMemoryAdaptiveHealingLearningRepository:
    def __init__(self) -> None:
        self._outcomes: dict[str, tuple[SelfHealingHistoricalOutcome, ...]] = {}
        self._validation: dict[str, tuple[AdaptiveValidationQualitySignal, ...]] = {}
        self._rollback: dict[str, tuple[AdaptiveRollbackHistoryEntry, ...]] = {}
        self._similarity: dict[str, tuple[str, ...]] = {}

    def seed_outcomes(self, tenant_id: str, outcomes: tuple[SelfHealingHistoricalOutcome, ...]) -> None:
        self._outcomes[tenant_id] = outcomes

    def seed_validation(self, tenant_id: str, signals: tuple[AdaptiveValidationQualitySignal, ...]) -> None:
        self._validation[tenant_id] = signals

    def seed_rollback(self, tenant_id: str, entries: tuple[AdaptiveRollbackHistoryEntry, ...]) -> None:
        self._rollback[tenant_id] = entries

    def seed_similarity(self, tenant_id: str, refs: tuple[str, ...]) -> None:
        self._similarity[tenant_id] = refs

    def historical_outcomes_for_tenant(self, tenant_id: str) -> tuple[SelfHealingHistoricalOutcome, ...]:
        return self._outcomes.get(tenant_id, ())

    def validation_quality_for_tenant(self, tenant_id: str) -> tuple[AdaptiveValidationQualitySignal, ...]:
        return self._validation.get(tenant_id, ())

    def rollback_history_for_tenant(self, tenant_id: str) -> tuple[AdaptiveRollbackHistoryEntry, ...]:
        return self._rollback.get(tenant_id, ())

    def context_similarity_refs_for_tenant(self, tenant_id: str) -> tuple[str, ...]:
        return self._similarity.get(tenant_id, ())


__all__ = ["InMemoryAdaptiveHealingLearningRepository"]
