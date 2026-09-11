# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Append-only preventive recommendation audit (PREVENTIVE R6 / R6-Q)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.preventive.evidence import RecommendationEvidenceReference


@dataclass(frozen=True, slots=True)
class PreventiveAuditRecord:
    """Per-recommendation reconstructability — answers why a suggestion existed."""

    recommendation_id: str
    prediction_signal_id: str
    context_snapshot_id: str
    analyzer_id: str
    analyzer_version: str
    evidence_refs: tuple[RecommendationEvidenceReference, ...]
    confidence: float
    created_at: datetime
    prediction_run_id: str = ""
    tenant_id: str = ""

    def __post_init__(self) -> None:
        if not self.recommendation_id.startswith("prrec_"):
            raise ValueError("recommendation_id must be prrec_*")
        if not self.prediction_signal_id.strip():
            raise ValueError("prediction_signal_id required")
        if not self.context_snapshot_id.strip():
            raise ValueError("context_snapshot_id required")
        if not self.evidence_refs:
            raise ValueError("evidence_refs required for audit reconstruction")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")


@dataclass(frozen=True, slots=True)
class PreventiveRecommendationAuditRecord:
    recommendation_run_id: str
    prediction_run_id: str
    risk_signal_id: str
    tenant_id: str
    recommendation_ids: tuple[str, ...]
    analyzer_outcomes: tuple[str, ...]
    degraded: bool
    recorded_at: datetime


__all__ = ["PreventiveAuditRecord", "PreventiveRecommendationAuditRecord"]
