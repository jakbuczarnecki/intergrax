# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operator investigation read DTOs composed from DiagnosticReadService (DIAG R7)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.diagnostic_investigation import (
    DiagnosticEvidenceExplanationConfidence,
    DiagnosticImpactNodeHealth,
    DiagnosticInvestigationSeverity,
    DiagnosticRecommendationKind,
    DiagnosticRootCauseStatus,
)
from intergrax.contracts.execution_identity import EventId, ExecutionId
from intergrax.contracts.multi_agent_failure_localization import DiagnosticFailureBoundary
from intergrax.runtime.diagnostics.decision_context_read_models import DecisionContextView
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessment
from intergrax.runtime.diagnostics.diagnostic_precision import FailureBoundary
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticExecutionLineageView,
    DiagnosticProblemOccurrenceView,
    DiagnosticProblemSummary,
)
from intergrax.runtime.diagnostics.diagnostic_extension_read_models import (
    DiagnosticExtensionOccurrenceEnrichment,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemId
from intergrax.contracts.predictive_investigation_read import (
    RelatedPredictiveHistoryEntryView,
    RelatedPredictiveRiskSignalView,
    RelatedPredictionOutcomeHistoryView,
)
from intergrax.contracts.preventive_investigation_read import (
    RelatedPreventiveActionHistoryEntryView,
    RelatedPreventiveRecommendationView,
)
from intergrax.contracts.self_healing_investigation_read import RelatedSelfHealingHistoryEntryView
from intergrax.contracts.external_operations.failure import ExternalOperationFailureKind


@dataclass(frozen=True, slots=True)
class DiagnosticExternalOperationContextView:
    execution_id: ExecutionId
    operation_attempt_id: str
    provider_id: str
    operation_type: str


@dataclass(frozen=True, slots=True)
class DiagnosticExternalOperationFailureView:
    execution_id: ExecutionId
    operation_attempt_id: str
    provider_id: str
    failure_kind: ExternalOperationFailureKind
    evidence_refs: tuple[EventId, ...]


class DiagnosticTimelineEntryKind(StrEnum):
    RUNTIME_EVENT = "runtime_event"
    EXECUTION_FAILURE = "execution_failure"
    DECISION_CONTEXT = "decision_context"
    EXTENSION_EVIDENCE = "extension_evidence"
    PROBLEM_OCCURRENCE = "problem_occurrence"
    GOVERNANCE = "governance"


@dataclass(frozen=True, slots=True)
class DiagnosticEvidenceExplanation:
    """One operator-visible claim with explicit confidence — never LLM narrative."""

    headline: str
    confidence: DiagnosticEvidenceExplanationConfidence
    evidence_refs: tuple[EventId, ...]
    detail: str = ""
    is_causal_claim: bool = False


@dataclass(frozen=True, slots=True)
class DiagnosticTimelineEntry:
    """
    Chronological evidence view entry.

    Ordering is temporal / positional only — never implies causality.
    """

    kind: DiagnosticTimelineEntryKind
    label: str
    observed_at: datetime | None
    execution_id: ExecutionId | None = None
    event_id: EventId | None = None
    sort_key: tuple[int, int] = (0, 0)


@dataclass(frozen=True, slots=True)
class DiagnosticTimeline:
    entries: tuple[DiagnosticTimelineEntry, ...]
    is_truncated: bool
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiagnosticImpactGraphNode:
    execution_id: ExecutionId
    parent_execution_id: ExecutionId | None
    health: DiagnosticImpactNodeHealth


@dataclass(frozen=True, slots=True)
class DiagnosticImpactGraph:
    """Projection over execution lineage topology — no separate graph authority."""

    root_execution_id: ExecutionId | None
    nodes: tuple[DiagnosticImpactGraphNode, ...]
    completeness_limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FailureInvestigationSummary:
    """
    Failure boundary vs impact vs cause — boundary is proven; cause stays unknown unless
    explicitly proven elsewhere (never heuristic).
    """

    failure_boundaries: tuple[DiagnosticFailureBoundary, ...]
    legacy_failure_boundary: FailureBoundary | None
    impact_root_execution_id: ExecutionId | None
    root_cause_status: DiagnosticRootCauseStatus
    supporting_evidence: tuple[DiagnosticEvidenceExplanation, ...]
    related_decision_ids: tuple[str, ...]
    affected_agent_ids: tuple[str, ...]
    explicit_unknowns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiagnosticRecommendation:
    kind: DiagnosticRecommendationKind
    recommendation: str
    reason: str
    evidence_refs: tuple[EventId, ...] = ()


@dataclass(frozen=True, slots=True)
class DiagnosticExecutionContextSummary:
    task_id: str
    run_id: str
    lineage: DiagnosticExecutionLineageView | None


@dataclass(frozen=True, slots=True)
class DiagnosticStructuredInvestigationPayload:
    """
    Bounded structured payload for a future Diagnostic AI Assistant.

    LLM consumes this projection; it is not diagnostic authority.
    """

    problem_id: str
    tenant_id: str
    severity: DiagnosticInvestigationSeverity
    what_happened: str
    failure_boundary_execution_ids: tuple[str, ...]
    impact_root_execution_id: str | None
    root_cause_status: DiagnosticRootCauseStatus
    proven_evidence: tuple[str, ...]
    supported_evidence: tuple[str, ...]
    unknowns: tuple[str, ...]
    related_decision_ids: tuple[str, ...]
    recommendations: tuple[str, ...]
    timeline_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiagnosticInvestigationView:
    """
    Single operator investigation read model for one Problem occurrence.

    Composed read projection only — not a diagnostic engine.
    """

    problem: DiagnosticProblemSummary
    problem_id: ProblemId
    occurrence: DiagnosticProblemOccurrenceView
    severity: DiagnosticInvestigationSeverity
    assessment: DiagnosticAssessment | None
    failure_boundary: FailureBoundary | None
    execution_context: DiagnosticExecutionContextSummary | None
    decision_context: DecisionContextView | None
    extension_enrichment: DiagnosticExtensionOccurrenceEnrichment | None
    evidence_summary: tuple[DiagnosticEvidenceExplanation, ...]
    timeline: DiagnosticTimeline
    affected_execution_ids: tuple[ExecutionId, ...]
    failure_investigation: FailureInvestigationSummary
    impact_graph: DiagnosticImpactGraph
    recommendations: tuple[DiagnosticRecommendation, ...]
    assistant_payload: DiagnosticStructuredInvestigationPayload
    investigation_limitations: tuple[str, ...]
    related_risk_signals: tuple[RelatedPredictiveRiskSignalView, ...] = ()
    forecast_risk_signals: tuple[RelatedPredictiveRiskSignalView, ...] = ()
    prediction_history: tuple[RelatedPredictiveHistoryEntryView, ...] = ()
    prediction_outcome_history: tuple[RelatedPredictionOutcomeHistoryView, ...] = ()
    preventive_recommendations: tuple[RelatedPreventiveRecommendationView, ...] = ()
    preventive_action_history: tuple[RelatedPreventiveActionHistoryEntryView, ...] = ()
    self_healing_history: tuple[RelatedSelfHealingHistoryEntryView, ...] = ()
    external_operation_context: tuple[DiagnosticExternalOperationContextView, ...] = ()
    external_operation_failures: tuple[DiagnosticExternalOperationFailureView, ...] = ()


@dataclass(frozen=True, slots=True)
class DiagnosticInvestigationResult:
    investigation: DiagnosticInvestigationView | None
    unavailable_reason: str | None = None


__all__ = [
    "DiagnosticEvidenceExplanation",
    "DiagnosticExecutionContextSummary",
    "DiagnosticImpactGraph",
    "DiagnosticImpactGraphNode",
    "DiagnosticInvestigationResult",
    "DiagnosticInvestigationSeverity",
    "DiagnosticInvestigationView",
    "DiagnosticRecommendation",
    "DiagnosticStructuredInvestigationPayload",
    "DiagnosticTimeline",
    "DiagnosticTimelineEntry",
    "DiagnosticTimelineEntryKind",
    "FailureInvestigationSummary",
]
