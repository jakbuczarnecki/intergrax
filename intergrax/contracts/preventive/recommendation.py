# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Preventive recommendation artifacts — never autonomous actions (PREVENTIVE R6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from intergrax.contracts.preventive.category import PreventiveRecommendationCategory
from intergrax.contracts.preventive.evidence import RecommendationEvidenceReference
from intergrax.contracts.preventive.governance import PreventiveRecommendationGovernance
from intergrax.contracts.preventive.lifecycle import PreventiveRecommendationLifecycleState
from intergrax.contracts.preventive.safety import PreventiveSafetyAssessment

PREVENTIVE_RECOMMENDATION_SCHEMA_VERSION = "preventive_recommendation.v2"


@dataclass(frozen=True, slots=True)
class PreventiveRecommendationCandidate:
    """Analyzer output — engine validates, scores, and audits before exposure."""

    category: str
    description: str
    expected_impact: str
    evidence_refs: tuple[RecommendationEvidenceReference, ...]
    analyzer_id: str
    analyzer_version: str
    raw_confidence: float = 0.75
    priority_label: str = "NORMAL"
    reasoning_summary: str = ""
    known_limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        PreventiveRecommendationCategory.validate(self.category)
        if not self.description.strip():
            raise ValueError("description must be non-empty")
        if not self.expected_impact.strip():
            raise ValueError("expected_impact must be non-empty")
        if not self.evidence_refs:
            raise ValueError("evidence_refs must be non-empty")
        if not (0.0 <= self.raw_confidence <= 1.0):
            raise ValueError("raw_confidence must be in [0.0, 1.0]")
        if not self.analyzer_id.strip() or not self.analyzer_version.strip():
            raise ValueError("analyzer identity required")


@dataclass(frozen=True, slots=True)
class PreventiveRecommendation:
    recommendation_id: str
    prediction_run_id: str
    risk_signal_id: str
    tenant_id: str
    category: str
    description: str
    expected_impact: str
    confidence: float
    evidence_refs: tuple[RecommendationEvidenceReference, ...]
    governance: PreventiveRecommendationGovernance
    created_at: datetime
    analyzer_id: str
    analyzer_version: str
    safety: PreventiveSafetyAssessment
    lifecycle_state: str
    reasoning_summary: str
    known_limitations: tuple[str, ...]
    evidence_quality: str
    context_snapshot_id: str
    schema_version: str = PREVENTIVE_RECOMMENDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.recommendation_id.startswith("prrec_"):
            raise ValueError("recommendation_id must be prrec_*")
        PreventiveRecommendationCategory.validate(self.category)
        if not self.evidence_refs:
            raise ValueError("evidence_refs must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        try:
            PreventiveRecommendationLifecycleState(self.lifecycle_state)
        except ValueError as exc:
            raise ValueError(f"invalid lifecycle_state: {self.lifecycle_state}") from exc
        if not self.reasoning_summary.strip():
            raise ValueError("reasoning_summary must be non-empty")
        if not self.evidence_quality.strip():
            raise ValueError("evidence_quality must be non-empty")
        if not self.context_snapshot_id.strip():
            raise ValueError("context_snapshot_id must be non-empty")


def mint_preventive_recommendation_id() -> str:
    return f"prrec_{uuid4().hex}"


__all__ = [
    "PREVENTIVE_RECOMMENDATION_SCHEMA_VERSION",
    "PreventiveRecommendation",
    "PreventiveRecommendationCandidate",
    "mint_preventive_recommendation_id",
]
