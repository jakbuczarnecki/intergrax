# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Analyzer input bundle — readonly diagnostic + predictive context (PREVENTIVE R6)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.predictive_context import PredictiveContext
from intergrax.contracts.predictive_risk import PredictiveRiskSignal


@dataclass(frozen=True, slots=True)
class DiagnosticEvidenceContext:
    """Readonly diagnostic evidence refs — not a second evidence store."""

    tenant_id: str
    evidence_refs: tuple[str, ...]
    diagnostic_context_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")


@dataclass(frozen=True, slots=True)
class HistoricalOutcome:
    """R5-derived effectiveness inputs for recommendation confidence."""

    tenant_id: str
    historical_success_rate: float
    similar_incident_refs: tuple[str, ...] = ()
    prior_recommendation_effectiveness: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        if not (0.0 <= self.historical_success_rate <= 1.0):
            raise ValueError("historical_success_rate must be in [0.0, 1.0]")


@dataclass(frozen=True, slots=True)
class PreventiveAnalysisInput:
    predictive_context: PredictiveContext
    risk_signal: PredictiveRiskSignal
    historical_outcome: HistoricalOutcome
    diagnostic_evidence: DiagnosticEvidenceContext

    def __post_init__(self) -> None:
        tenant = self.predictive_context.tenant_id
        if tenant != self.risk_signal.tenant_id:
            raise ValueError("tenant isolation violation: context/risk_signal")
        if tenant != self.historical_outcome.tenant_id:
            raise ValueError("tenant isolation violation: context/historical_outcome")
        if tenant != self.diagnostic_evidence.tenant_id:
            raise ValueError("tenant isolation violation: context/diagnostic_evidence")


__all__ = [
    "DiagnosticEvidenceContext",
    "HistoricalOutcome",
    "PreventiveAnalysisInput",
]
