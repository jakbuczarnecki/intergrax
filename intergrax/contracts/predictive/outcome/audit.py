# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Audit envelope for outcome evaluation runs (PREDICTIVE R5)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.predictive.outcome.types import (
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeType,
)


@dataclass(frozen=True, slots=True)
class PredictionOutcomeAuditRecord:
    prediction_run_id: str
    prediction_signal_id: str
    tenant_id: str
    analyzer_id: str
    outcome_type: PredictionOutcomeType
    evaluation_status: PredictionOutcomeEvaluationStatus
    evaluated_at: datetime
    evidence_refs: tuple[str, ...]
    resolver_id: str
    resolver_outcomes: tuple[str, ...]
    rationale: str


__all__ = ["PredictionOutcomeAuditRecord"]
