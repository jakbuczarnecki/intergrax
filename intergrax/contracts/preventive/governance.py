# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Human governance envelope — automatic execution always forbidden (PREVENTIVE R6)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.predictive_risk import PredictiveRiskSeverity


@dataclass(frozen=True, slots=True)
class PreventiveRecommendationGovernance:
    risk_level: PredictiveRiskSeverity
    confidence: float
    required_approval: bool
    execution_allowed: bool = False
    priority_label: str = "NORMAL"

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        if self.execution_allowed:
            raise ValueError("execution_allowed must remain false — preventive layer is not remediation")
        if not self.priority_label.strip():
            raise ValueError("priority_label must be non-empty")


__all__ = ["PreventiveRecommendationGovernance"]
