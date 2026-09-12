# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing strategy quality profile — outcome loop inputs (SELF-HEALING R1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SelfHealingStrategyQualityProfile:
    strategy_id: str
    tenant_id: str
    successful_preventions: int = 0
    failed_actions: int = 0
    rollback_rate: float = 0.0
    false_positive_rate: float = 0.0

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not (0.0 <= self.rollback_rate <= 1.0):
            raise ValueError("rollback_rate must be in [0.0, 1.0]")
        if not (0.0 <= self.false_positive_rate <= 1.0):
            raise ValueError("false_positive_rate must be in [0.0, 1.0]")


__all__ = ["SelfHealingStrategyQualityProfile"]
