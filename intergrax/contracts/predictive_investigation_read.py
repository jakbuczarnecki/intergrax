# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operator investigation attachment for predictive risk (PREDICTIVE R1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.predictive_history import PredictiveHistoryOutcomeStatus
from intergrax.contracts.predictive.outcome.types import (
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeType,
)
from intergrax.contracts.predictive.quality import PredictiveQualityAssessment
from intergrax.contracts.predictive.context_quality import PredictiveContextQualityReport
from intergrax.contracts.predictive_risk import PredictiveRiskScope, PredictiveRiskSeverity


@dataclass(frozen=True, slots=True)
class RelatedPredictiveHistoryEntryView:
    """Readonly historical prediction on DiagnosticInvestigationView — not incident truth."""

    risk_signal_id: str
    tenant_id: str
    generated_at: datetime
    risk_type: str
    subject_identity: str
    confidence: float
    outcome_status: PredictiveHistoryOutcomeStatus
    analyzer_id: str
    summary: str
    evidence_refs: tuple[str, ...]


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
    prediction_quality: PredictiveQualityAssessment | None = None
    context_quality: PredictiveContextQualityReport | None = None
    prediction_explanation: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RelatedPredictionOutcomeHistoryView:
    """Readonly outcome evaluation on DiagnosticInvestigationView — not incident truth."""

    prediction_signal_id: str
    tenant_id: str
    prediction_run_id: str
    risk_type: str
    analyzer_id: str
    outcome_type: PredictionOutcomeType
    evaluation_status: PredictionOutcomeEvaluationStatus
    evaluated_at: datetime
    evidence_refs: tuple[str, ...]
    rationale: str
    precision_delta_label: str = ""


__all__ = [
    "RelatedPredictiveHistoryEntryView",
    "RelatedPredictiveRiskSignalView",
    "RelatedPredictionOutcomeHistoryView",
]
