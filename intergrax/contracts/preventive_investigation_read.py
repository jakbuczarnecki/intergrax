# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operator investigation attachment for preventive recommendations (PREVENTIVE R6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.predictive_risk import PredictiveRiskSeverity
from intergrax.contracts.preventive.evidence import RecommendationEvidenceReference


@dataclass(frozen=True, slots=True)
class RelatedPreventiveRecommendationView:
    """Readonly preventive recommendation on DiagnosticInvestigationView — not an action."""

    recommendation_id: str
    tenant_id: str
    risk_signal_id: str
    category: str
    description: str
    expected_impact: str
    confidence: float
    evidence_refs: tuple[RecommendationEvidenceReference, ...]
    risk_level: PredictiveRiskSeverity
    required_approval: bool
    execution_allowed: bool
    priority_label: str
    analyzer_id: str
    related_risk_type: str
    lifecycle_state: str = "PRESENTED"
    reasoning_summary: str = ""
    known_limitations: tuple[str, ...] = ()
    evidence_quality: str = ""


@dataclass(frozen=True, slots=True)
class RelatedPreventiveActionHistoryEntryView:
    """Readonly preventive action trail on DiagnosticInvestigationView."""

    proposal_id: str
    tenant_id: str
    action_type: str
    risk_signal_refs: tuple[str, ...]
    recommendation_refs: tuple[str, ...]
    approval_refs: tuple[str, ...]
    external_operation_id: str | None
    execution_id: str | None
    outcome: str
    recorded_at: datetime
    admission_reason: str = ""


__all__ = ["RelatedPreventiveActionHistoryEntryView", "RelatedPreventiveRecommendationView"]
