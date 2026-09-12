# © Artur Czarnecki. All rights reserved.

"""Typed contracts for adaptive decision intelligence (DS-E2E-15J-L10)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from testing_support.decision_e2e.model_matrix.decision_observability_analytics.contracts import (
    DecisionAnalyticsResult,
)
from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.contracts import (
    DecisionOptimizationResult,
)
from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleRecord,
)
from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing.contracts import (
    GovernanceDecision,
)
from testing_support.decision_e2e.model_matrix.model_capability_baseline.contracts import (
    ModelCapabilityProfile,
)

ADAPTIVE_INTELLIGENCE_TASK_ID = "DS-E2E-15J-L10.ADAPTIVE-DECISION-INTELLIGENCE"
ADAPTIVE_INTELLIGENCE_VERSION = "1"


class AdaptiveIntelligenceRunStatus(StrEnum):
    COMPLETE = "complete"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AdaptiveDataSourceKind(StrEnum):
    LIFECYCLE_RECORD = "lifecycle_record"
    ANALYTICS_RESULT = "analytics_result"
    OPTIMIZATION_RESULT = "optimization_result"
    CAPABILITY_PROFILE = "capability_profile"
    GOVERNANCE_DECISION = "governance_decision"


@dataclass(frozen=True, slots=True)
class AdaptiveDataSourceRef:
    source_kind: AdaptiveDataSourceKind
    reference_id: str


@dataclass(frozen=True, slots=True)
class AdaptiveDecisionContextReference:
    """Reference to the decision being assisted — not an executable decision."""

    decision_id: str
    decision_subject: str


@dataclass(frozen=True, slots=True)
class HistoricalEvidence:
    evidence_id: str
    summary: str
    source_refs: tuple[AdaptiveDataSourceRef, ...]
    context_provider_id: str
    context_provider_version: str


@dataclass(frozen=True, slots=True)
class AdaptiveReasoningInsight:
    """Intermediate reasoning output — advisory only, no side effects."""

    reasoning_insight_id: str
    reasoning_provider_id: str
    reasoning_provider_version: str
    reasoning_summary: str
    source_evidence_ids: tuple[str, ...]
    data_source_refs: tuple[AdaptiveDataSourceRef, ...]
    confidence: ConfidenceLevel
    confidence_score: float


@dataclass(frozen=True, slots=True)
class AdaptiveDecisionRecommendation:
    """Presentation-layer recommendation — human or upstream process decides."""

    recommendation_id: str
    recommendation_provider_id: str
    recommendation_provider_version: str
    recommendation_kind: str
    narrative: str
    linked_reasoning_insight_ids: tuple[str, ...]
    confidence: ConfidenceLevel
    confidence_score: float


@dataclass(frozen=True, slots=True)
class AdaptiveDecisionInsight:
    """Auditable advisory bundle — not an executable system change."""

    insight_id: str
    decision_context_reference: AdaptiveDecisionContextReference
    historical_evidence: tuple[HistoricalEvidence, ...]
    reasoning_summary: str
    recommendation: AdaptiveDecisionRecommendation
    confidence: ConfidenceLevel
    confidence_score: float
    generated_at: datetime
    context_provider_ids: tuple[str, ...]
    reasoning_provider_ids: tuple[str, ...]
    recommendation_provider_id: str
    data_source_refs: tuple[AdaptiveDataSourceRef, ...]


@dataclass(frozen=True, slots=True)
class AdaptiveDecisionIntelligenceAuditMetadata:
    engine_task_id: str
    engine_version: str
    context_provider_ids: tuple[str, ...]
    reasoning_provider_ids: tuple[str, ...]
    recommendation_provider_ids: tuple[str, ...]
    analyzed_at: datetime
    decision_context_reference: AdaptiveDecisionContextReference
    data_source_refs: tuple[AdaptiveDataSourceRef, ...]


@dataclass(frozen=True, slots=True)
class AdaptiveDecisionIntelligenceInput:
    """Read-only pipeline inputs for context plugins."""

    decision_context_reference: AdaptiveDecisionContextReference
    lifecycle_records: tuple[DecisionLifecycleRecord, ...] = ()
    analytics_results: tuple[DecisionAnalyticsResult, ...] = ()
    optimization_result: DecisionOptimizationResult | None = None
    capability_profiles: tuple[ModelCapabilityProfile, ...] = ()
    governance_decisions: tuple[GovernanceDecision, ...] = ()


@dataclass(frozen=True, slots=True)
class AdaptiveDecisionIntelligenceContext:
    """Merged read-only context for reasoning — built by context providers."""

    decision_context_reference: AdaptiveDecisionContextReference
    historical_evidence: tuple[HistoricalEvidence, ...]
    data_source_refs: tuple[AdaptiveDataSourceRef, ...]
    context_provider_ids: tuple[str, ...]
    context_provider_versions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AdaptiveDecisionIntelligenceResult:
    intelligence_task_id: str
    status: AdaptiveIntelligenceRunStatus
    audit: AdaptiveDecisionIntelligenceAuditMetadata
    reasoning_insights: tuple[AdaptiveReasoningInsight, ...]
    recommendations: tuple[AdaptiveDecisionRecommendation, ...]
    insights: tuple[AdaptiveDecisionInsight, ...]


__all__ = [
    "ADAPTIVE_INTELLIGENCE_TASK_ID",
    "ADAPTIVE_INTELLIGENCE_VERSION",
    "AdaptiveDataSourceKind",
    "AdaptiveDataSourceRef",
    "AdaptiveDecisionContextReference",
    "AdaptiveDecisionInsight",
    "AdaptiveDecisionIntelligenceAuditMetadata",
    "AdaptiveDecisionIntelligenceContext",
    "AdaptiveDecisionIntelligenceInput",
    "AdaptiveDecisionIntelligenceResult",
    "AdaptiveDecisionRecommendation",
    "AdaptiveIntelligenceRunStatus",
    "AdaptiveReasoningInsight",
    "ConfidenceLevel",
    "HistoricalEvidence",
]
