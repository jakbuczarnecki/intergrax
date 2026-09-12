# © Artur Czarnecki. All rights reserved.

"""Typed contracts for decision optimization suggestions (DS-E2E-15J-L9)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from testing_support.decision_e2e.model_matrix.decision_observability_analytics.contracts import (
    DecisionAnalyticsResult,
    DecisionObservation,
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

OPTIMIZATION_TASK_ID = "DS-E2E-15J-L9.DECISION-OPTIMIZATION-LEARNING-LOOP"
OPTIMIZATION_VERSION = "1"


class OptimizationRunStatus(StrEnum):
    COMPLETE = "complete"
    INSUFFICIENT_DATA = "insufficient_data"


class OptimizationArea(StrEnum):
    GOVERNANCE_FRICTION = "governance_friction"
    OUTCOME_RELIABILITY = "outcome_reliability"
    MODEL_CAPABILITY = "model_capability"
    CUSTOM = "custom"


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OptimizationDataSourceKind(StrEnum):
    ANALYTICS_RESULT = "analytics_result"
    OBSERVATION = "observation"
    LIFECYCLE_RECORD = "lifecycle_record"
    CAPABILITY_PROFILE = "capability_profile"
    GOVERNANCE_DECISION = "governance_decision"


@dataclass(frozen=True, slots=True)
class OptimizationDataSourceRef:
    source_kind: OptimizationDataSourceKind
    reference_id: str


@dataclass(frozen=True, slots=True)
class DetectedOptimizationPattern:
    pattern_id: str
    analyzer_id: str
    analyzer_version: str
    optimization_area: OptimizationArea
    summary: str
    source_decision_ids: tuple[str, ...]
    data_source_refs: tuple[OptimizationDataSourceRef, ...]
    confidence: ConfidenceLevel


@dataclass(frozen=True, slots=True)
class OptimizationInsight:
    insight_id: str
    generator_id: str
    generator_version: str
    optimization_area: OptimizationArea
    narrative: str
    pattern_ids: tuple[str, ...]
    source_decision_ids: tuple[str, ...]
    data_source_refs: tuple[OptimizationDataSourceRef, ...]
    confidence: ConfidenceLevel


@dataclass(frozen=True, slots=True)
class DecisionOptimizationSuggestion:
    """Describes what may be considered — not an executable system change."""

    suggestion_id: str
    source_decisions: tuple[str, ...]
    optimization_area: OptimizationArea
    observation_references: tuple[OptimizationDataSourceRef, ...]
    rationale: str
    confidence: ConfidenceLevel
    generated_at: datetime
    provider_id: str
    provider_version: str
    insight_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DecisionOptimizationAuditMetadata:
    engine_task_id: str
    engine_version: str
    pattern_analyzer_ids: tuple[str, ...]
    insight_generator_ids: tuple[str, ...]
    recommendation_provider_ids: tuple[str, ...]
    analyzed_at: datetime
    analytics_result_refs: tuple[str, ...]
    observation_decision_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DecisionOptimizationContext:
    """Read-only inputs from existing decision pipelines — no duplicate stores."""

    analytics_results: tuple[DecisionAnalyticsResult, ...]
    observations: tuple[DecisionObservation, ...] = ()
    lifecycle_records: tuple[DecisionLifecycleRecord, ...] = ()
    capability_profiles: tuple[ModelCapabilityProfile, ...] = ()
    governance_decisions: tuple[GovernanceDecision, ...] = ()


@dataclass(frozen=True, slots=True)
class DecisionOptimizationResult:
    optimization_task_id: str
    status: OptimizationRunStatus
    audit: DecisionOptimizationAuditMetadata
    patterns: tuple[DetectedOptimizationPattern, ...]
    insights: tuple[OptimizationInsight, ...]
    suggestions: tuple[DecisionOptimizationSuggestion, ...]
    aggregate_confidence: ConfidenceLevel | None


__all__ = [
    "OPTIMIZATION_TASK_ID",
    "OPTIMIZATION_VERSION",
    "ConfidenceLevel",
    "DecisionOptimizationAuditMetadata",
    "DecisionOptimizationContext",
    "DecisionOptimizationResult",
    "DecisionOptimizationSuggestion",
    "DetectedOptimizationPattern",
    "OptimizationArea",
    "OptimizationDataSourceKind",
    "OptimizationDataSourceRef",
    "OptimizationInsight",
    "OptimizationRunStatus",
]
