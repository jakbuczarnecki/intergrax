# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Adaptive healing input context (SELF-HEALING R4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.context import SelfHealingHistoricalOutcome
from intergrax.contracts.self_healing.strategy import SelfHealingStrategy


@dataclass(frozen=True, slots=True)
class AdaptiveValidationQualitySignal:
    tenant_id: str
    strategy_id: str
    quality_score: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError("quality_score must be in [0.0, 1.0]")
        if not self.evidence_refs:
            raise ValueError("evidence_refs required for validation quality")


@dataclass(frozen=True, slots=True)
class AdaptiveRollbackHistoryEntry:
    tenant_id: str
    strategy_id: str
    rollback_count: int
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if self.rollback_count < 0:
            raise ValueError("rollback_count must be >= 0")


@dataclass(frozen=True, slots=True)
class AdaptiveHealingContext:
    """
    Read-only input to adaptive intelligence — no execution or governance bypass.
    """

    tenant_id: str
    strategy_candidates: tuple[SelfHealingStrategy, ...]
    historical_outcomes: tuple[SelfHealingHistoricalOutcome, ...]
    evidence_refs: tuple[str, ...]
    execution_context_ref: str | None
    validation_quality: tuple[AdaptiveValidationQualitySignal, ...] = ()
    rollback_history: tuple[AdaptiveRollbackHistoryEntry, ...] = ()
    context_similarity_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        for outcome in self.historical_outcomes:
            if outcome.tenant_id != self.tenant_id:
                raise ValueError("tenant isolation violation: historical_outcomes")
        for signal in self.validation_quality:
            if signal.tenant_id != self.tenant_id:
                raise ValueError("tenant isolation violation: validation_quality")
        for entry in self.rollback_history:
            if entry.tenant_id != self.tenant_id:
                raise ValueError("tenant isolation violation: rollback_history")


__all__ = [
    "AdaptiveHealingContext",
    "AdaptiveRollbackHistoryEntry",
    "AdaptiveValidationQualitySignal",
]
