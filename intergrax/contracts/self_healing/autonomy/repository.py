# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Autonomy decision persistence port (SELF-HEALING R6.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.decision import AutonomyControlDecision


@dataclass(frozen=True, slots=True)
class AutonomyDecisionCorrelationQuery:
    tenant_id: str
    recommendation_correlation_id: str

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.recommendation_correlation_id.strip():
            raise ValueError("recommendation_correlation_id required")


@runtime_checkable
class AutonomyRepository(Protocol):
    def append_decision(self, decision: AutonomyControlDecision) -> AutonomyControlDecision: ...

    def get_latest_by_correlation(
        self,
        query: AutonomyDecisionCorrelationQuery,
    ) -> AutonomyControlDecision | None: ...


__all__ = ["AutonomyDecisionCorrelationQuery", "AutonomyRepository"]
