# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing evaluation and execution outcomes (SELF-HEALING R1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.self_healing.decision import SelfHealingDecision


class SelfHealingStrategyEvaluationStatus(StrEnum):
    DECISION = "DECISION"
    ABSTAIN = "ABSTAIN"
    STRATEGY_FAILED = "STRATEGY_FAILED"


@dataclass(frozen=True, slots=True)
class SelfHealingStrategyEvaluationResult:
    strategy_id: str
    status: SelfHealingStrategyEvaluationStatus
    decision: SelfHealingDecision | None = None
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if self.status is SelfHealingStrategyEvaluationStatus.DECISION:
            if self.decision is None:
                raise ValueError("decision required when status is DECISION")
        if self.status is SelfHealingStrategyEvaluationStatus.STRATEGY_FAILED:
            if not self.failure_reason.strip():
                raise ValueError("failure_reason required for STRATEGY_FAILED")


class SelfHealingExecutionOutcome(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    DENIED = "DENIED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"


__all__ = [
    "SelfHealingExecutionOutcome",
    "SelfHealingStrategyEvaluationResult",
    "SelfHealingStrategyEvaluationStatus",
]
