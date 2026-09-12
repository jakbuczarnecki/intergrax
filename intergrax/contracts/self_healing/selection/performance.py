# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Strategy quality feedback for selection (SELF-HEALING R3)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SelfHealingStrategyPerformance:
    strategy_id: str
    tenant_id: str
    executions: int
    success_rate: float
    rollback_rate: float
    average_recovery_time_seconds: float
    confidence_calibration: float

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if self.executions < 0:
            raise ValueError("executions must be >= 0")
        for rate in (self.success_rate, self.rollback_rate, self.confidence_calibration):
            if not (0.0 <= rate <= 1.0):
                raise ValueError("rates must be in [0.0, 1.0]")
        if self.average_recovery_time_seconds < 0.0:
            raise ValueError("average_recovery_time_seconds must be >= 0")


__all__ = ["SelfHealingStrategyPerformance"]
