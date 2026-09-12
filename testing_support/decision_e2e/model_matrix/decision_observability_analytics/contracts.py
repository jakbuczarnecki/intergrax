# © Artur Czarnecki. All rights reserved.

"""Typed contracts for decision observability analytics (DS-E2E-15J-L8)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleState,
    DecisionSourceReference,
    DecisionType,
)
from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing.contracts import (
    GovernanceDisposition,
)

OBSERVABILITY_TASK_ID = "DS-E2E-15J-L8.DECISION-OBSERVABILITY-ANALYTICS"
OBSERVABILITY_VERSION = "1"


class ObservationEventKind(StrEnum):
    LIFECYCLE_RECORD = "lifecycle_record"
    LIFECYCLE_TRANSITION = "lifecycle_transition"
    GOVERNANCE_OUTCOME = "governance_outcome"


class AnalyticsResultStatus(StrEnum):
    COMPLETE = "complete"
    INSUFFICIENT_DATA = "insufficient_data"


class AnalysisPayloadKind(StrEnum):
    OUTCOME = "outcome"
    LIFECYCLE_PERFORMANCE = "lifecycle_performance"
    GOVERNANCE = "governance"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class DecisionObservationMetadata:
    """Factual metadata only — no interpretive quality labels."""

    event_kind: ObservationEventKind
    observed_at: datetime
    governance_disposition: GovernanceDisposition | None = None
    transition_previous_state: DecisionLifecycleState | None = None
    transition_new_state: DecisionLifecycleState | None = None
    transition_reason: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionObservation:
    decision_id: str
    lifecycle_state: DecisionLifecycleState
    decision_type: DecisionType
    source_references: tuple[DecisionSourceReference, ...]
    metadata: DecisionObservationMetadata


@dataclass(frozen=True, slots=True)
class DecisionAnalyticsAuditMetadata:
    analyzer_id: str
    analyzer_version: str
    decision_ids: tuple[str, ...]
    period_start: datetime | None
    period_end: datetime | None


@dataclass(frozen=True, slots=True)
class DecisionOutcomeCounts:
    completed: int
    failed: int
    blocked: int
    in_progress: int


@dataclass(frozen=True, slots=True)
class TransitionDurationFact:
    decision_id: str
    previous_state: DecisionLifecycleState | None
    new_state: DecisionLifecycleState
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class LifecyclePerformancePayload:
    transition_durations: tuple[TransitionDurationFact, ...]
    average_transition_seconds: float | None


@dataclass(frozen=True, slots=True)
class GovernanceOutcomeCounts:
    allow: int
    block: int
    require_approval: int


@dataclass(frozen=True, slots=True)
class CustomAnalysisPayload:
    summary_token: str
    numeric_fact: int


@dataclass(frozen=True, slots=True)
class DecisionAnalyticsResult:
    observability_task_id: str
    status: AnalyticsResultStatus
    audit: DecisionAnalyticsAuditMetadata
    payload_kind: AnalysisPayloadKind
    outcome_payload: DecisionOutcomeCounts | None = None
    performance_payload: LifecyclePerformancePayload | None = None
    governance_payload: GovernanceOutcomeCounts | None = None
    custom_payload: CustomAnalysisPayload | None = None


@dataclass(frozen=True, slots=True)
class DecisionMetricsSnapshot:
    metrics_provider_id: str
    metrics_provider_version: str
    observation_count: int
    unique_decision_count: int


@dataclass(frozen=True, slots=True)
class DecisionAnalyticsReport:
    report_provider_id: str
    report_provider_version: str
    generated_at: datetime
    analyzer_ids: tuple[str, ...]
    summary_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DecisionObservabilityRunResult:
    observability_task_id: str
    run_at: datetime
    observations: tuple[DecisionObservation, ...]
    analytics_results: tuple[DecisionAnalyticsResult, ...]
    metrics: tuple[DecisionMetricsSnapshot, ...]
    reports: tuple[DecisionAnalyticsReport, ...]


__all__ = [
    "OBSERVABILITY_TASK_ID",
    "OBSERVABILITY_VERSION",
    "AnalysisPayloadKind",
    "AnalyticsResultStatus",
    "CustomAnalysisPayload",
    "DecisionAnalyticsAuditMetadata",
    "DecisionAnalyticsReport",
    "DecisionAnalyticsResult",
    "DecisionMetricsSnapshot",
    "DecisionObservation",
    "DecisionObservationMetadata",
    "DecisionObservabilityRunResult",
    "DecisionOutcomeCounts",
    "GovernanceOutcomeCounts",
    "LifecyclePerformancePayload",
    "ObservationEventKind",
    "TransitionDurationFact",
]
