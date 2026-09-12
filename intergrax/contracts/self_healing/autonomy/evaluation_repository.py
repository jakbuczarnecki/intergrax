# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Persistence port for autonomy evaluation results (SELF-HEALING R6.2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.evaluation_result import AutonomyEvaluationResult
from intergrax.contracts.self_healing.autonomy.repository import AutonomyDecisionCorrelationQuery


@runtime_checkable
class AutonomyDecisionRepository(Protocol):
    def append_evaluation(self, result: AutonomyEvaluationResult) -> AutonomyEvaluationResult: ...

    def get_latest_evaluation(
        self,
        query: AutonomyDecisionCorrelationQuery,
    ) -> AutonomyEvaluationResult | None: ...


__all__ = ["AutonomyDecisionRepository"]
