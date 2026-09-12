# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Recommendation request scope (SELF-HEALING R5.3)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategyRecommendationRequest:
    """
    Diagnostic-scoped advisory request.

    Does not trigger strategy evaluation or execution — only scopes historical quality reads.
    """

    tenant_id: str
    diagnostic_investigation_id: str
    problem_id: str
    candidate_strategy_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.diagnostic_investigation_id.strip():
            raise ValueError("diagnostic_investigation_id required")
        if not self.problem_id.strip():
            raise ValueError("problem_id required")
        if not self.candidate_strategy_ids:
            raise ValueError("candidate_strategy_ids must be non-empty")
        for strategy_id in self.candidate_strategy_ids:
            if not strategy_id.strip():
                raise ValueError("candidate strategy_id must be non-empty")


__all__ = ["StrategyRecommendationRequest"]
