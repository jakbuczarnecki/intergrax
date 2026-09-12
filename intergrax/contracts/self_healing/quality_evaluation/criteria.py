# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read criteria for strategy quality evaluation (SELF-HEALING R5.2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategyQualityEvaluationCriteria:
    tenant_id: str
    strategy_id: str

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")


__all__ = ["StrategyQualityEvaluationCriteria"]
