# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Autonomy control evaluation request (SELF-HEALING R6.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.autonomy.action_kind import AutonomyRequestedActionKind
from intergrax.contracts.self_healing.autonomy.context import AutonomyDecisionContext
from intergrax.contracts.self_healing.strategy_recommendation.recommendation import StrategyRecommendation


@dataclass(frozen=True, slots=True)
class AutonomyControlRequest:
    """Binds an R5.3 advisory recommendation to autonomy evaluation context."""

    recommendation: StrategyRecommendation
    decision_context: AutonomyDecisionContext
    requested_action_kind: AutonomyRequestedActionKind = (
        AutonomyRequestedActionKind.CONSIDER_STRATEGY_FOR_EXECUTION
    )

    def __post_init__(self) -> None:
        rec = self.recommendation
        ctx = self.decision_context
        if rec.tenant_id != ctx.audit.tenant_id:
            raise ValueError("tenant isolation violation: recommendation vs audit")
        if rec.diagnostic_investigation_id != ctx.audit.diagnostic_investigation_id:
            raise ValueError("investigation correlation mismatch")
        if rec.problem_id != ctx.audit.problem_id:
            raise ValueError("problem correlation mismatch")


__all__ = ["AutonomyControlRequest"]
