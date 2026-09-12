# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Neutral default risk evaluator — no production scoring (SELF-HEALING R6.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest
from intergrax.contracts.self_healing.autonomy.risk import (
    AutonomyRiskAssessment,
    AutonomyRiskBand,
    AutonomyRiskFactor,
)


@dataclass(frozen=True, slots=True)
class DefaultAutonomyRiskEvaluator:
    _evaluator_id: str = "platform.default_autonomy_risk"

    @property
    def evaluator_id(self) -> str:
        return self._evaluator_id

    def evaluate(
        self,
        request: AutonomyControlRequest,
        effective_level: AutonomyLevel,
    ) -> AutonomyRiskAssessment:
        _ = effective_level
        return AutonomyRiskAssessment(
            evaluator_id=self.evaluator_id,
            risk_band=AutonomyRiskBand.UNKNOWN,
            factors=(
                AutonomyRiskFactor(
                    label="neutral_default",
                    detail=(
                        f"No risk model configured for strategy "
                        f"{request.recommendation.recommended_strategy_id}."
                    ),
                ),
            ),
            rationale="Default evaluator returns neutral posture until enterprise risk plugins are configured.",
        )


__all__ = ["DefaultAutonomyRiskEvaluator"]
