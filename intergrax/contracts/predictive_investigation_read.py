# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operator investigation attachment for predictive risk (PREDICTIVE R1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.predictive_risk import PredictiveRiskScope, PredictiveRiskSeverity


@dataclass(frozen=True, slots=True)
class RelatedPredictiveRiskSignalView:
    """
    Readonly risk signal on DiagnosticInvestigationView — not proven failure.
    """

    signal_id: str
    tenant_id: str
    scope: PredictiveRiskScope
    subject_identity: str
    risk_type: str
    severity: PredictiveRiskSeverity
    confidence: float
    evidence_refs: tuple[str, ...]
    prediction_window_label: str
    generated_at: datetime
    model_version: str
    summary: str
    recommended_actions: tuple[str, ...]
    analyzer_id: str


__all__ = ["RelatedPredictiveRiskSignalView"]
