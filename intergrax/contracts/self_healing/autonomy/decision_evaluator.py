# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Pluggable autonomy decision evaluation port (SELF-HEALING R6.2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.evaluation_result import AutonomyEvaluationResult
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest


@runtime_checkable
class AutonomyDecisionEvaluator(Protocol):
    @property
    def evaluator_id(self) -> str: ...

    def evaluate(self, request: AutonomyControlRequest) -> AutonomyEvaluationResult:
        """
        Produce an explainable evaluation of whether autonomy posture is acceptable.

        Must not execute strategies, invoke executors, or mutate lifecycle state.
        """
        ...


__all__ = ["AutonomyDecisionEvaluator"]
