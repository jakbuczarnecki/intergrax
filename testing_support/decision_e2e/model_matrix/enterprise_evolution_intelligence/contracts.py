# © Artur Czarnecki. All rights reserved.

"""Typed contracts for enterprise evolution intelligence (DS-E2E-15J-L15)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.contracts import (
    ControlledEvolutionRecord,
)
from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.contracts import (
    AdaptationExecutionResult,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_operations.contracts import (
    EvolutionHealthObservation,
    EvolutionOperationRecord,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.contracts import (
    SelfImprovementGovernanceDecision,
    SelfImprovementGovernanceStatus,
)

ENTERPRISE_EVOLUTION_INTELLIGENCE_TASK_ID = (
    "DS-E2E-15J-L15.ENTERPRISE-EVOLUTION-INTELLIGENCE"
)
ENTERPRISE_EVOLUTION_INTELLIGENCE_VERSION = "1"


class EvolutionIntelligenceRunStatus(StrEnum):
    COMPLETE = "complete"
    INSUFFICIENT_INPUT = "insufficient_input"


class EvolutionRecommendationKind(StrEnum):
    REVIEW_ADAPTATION = "review_adaptation"
    COLLECT_MORE_EVIDENCE = "collect_more_evidence"
    CONTINUE_OBSERVATION = "continue_observation"


@dataclass(frozen=True, slots=True)
class EvolutionGovernanceReference:
    """Read-only governance trace — not an approval or change command."""

    governance_decision_reference: str
    governance_status: SelfImprovementGovernanceStatus
    approval_id: str | None


@dataclass(frozen=True, slots=True)
class EvolutionHistoryReference:
    """Pointer to evolution history scope under analysis."""

    adaptation_id: str
    version: str
    proposal_id: str | None
    controlled_evolution_record_id: str | None


@dataclass(frozen=True, slots=True)
class EvolutionIntelligenceDataSourceRef:
    source_id: str
    source_kind: str
    description: str


@dataclass(frozen=True, slots=True)
class EvolutionIntelligenceContext:
    """Read-only analysis input — no runtime or execution engine references."""

    adaptation_id: str
    version: str
    history_reference: EvolutionHistoryReference
    operation_records: tuple[EvolutionOperationRecord, ...]
    execution_results: tuple[AdaptationExecutionResult, ...]
    health_observations: tuple[EvolutionHealthObservation, ...]
    controlled_evolution: ControlledEvolutionRecord | None
    governance_reference: EvolutionGovernanceReference | None
    governance_decision: SelfImprovementGovernanceDecision | None


@dataclass(frozen=True, slots=True)
class EvolutionAnalysisFinding:
    finding_id: str
    analyzer_id: str
    analyzer_version: str
    category: str
    summary: str
    data_source_refs: tuple[EvolutionIntelligenceDataSourceRef, ...]


@dataclass(frozen=True, slots=True)
class EvolutionMetricSnapshot:
    metric_id: str
    metric_provider_id: str
    metric_provider_version: str
    metric_name: str
    metric_value: str
    unit_label: str
    data_source_refs: tuple[EvolutionIntelligenceDataSourceRef, ...]


@dataclass(frozen=True, slots=True)
class EvolutionIntelligenceInsight:
    insight_id: str
    observation: str
    insight_provider_id: str
    insight_provider_version: str
    linked_finding_ids: tuple[str, ...]
    linked_metric_ids: tuple[str, ...]
    evidence_source_refs: tuple[EvolutionIntelligenceDataSourceRef, ...]


@dataclass(frozen=True, slots=True)
class EvolutionIntelligenceRecommendation:
    recommendation_id: str
    kind: EvolutionRecommendationKind
    summary: str
    recommendation_provider_id: str
    recommendation_provider_version: str
    linked_insight_ids: tuple[str, ...]
    rationale: str


@dataclass(frozen=True, slots=True)
class EvolutionIntelligenceAuditMetadata:
    intelligence_task_id: str
    intelligence_layer_version: str
    adaptation_id: str
    adaptation_version: str
    history_reference: EvolutionHistoryReference
    intelligence_provider_ids: tuple[str, ...]
    intelligence_provider_versions: tuple[str, ...]
    analyzer_ids: tuple[str, ...]
    analyzer_versions: tuple[str, ...]
    metric_provider_ids: tuple[str, ...]
    metric_provider_versions: tuple[str, ...]
    insight_provider_ids: tuple[str, ...]
    insight_provider_versions: tuple[str, ...]
    recommendation_provider_ids: tuple[str, ...]
    recommendation_provider_versions: tuple[str, ...]
    data_source_refs: tuple[EvolutionIntelligenceDataSourceRef, ...]
    analysis_scope_summary: str
    analyzed_at: datetime


@dataclass(frozen=True, slots=True)
class EvolutionIntelligenceResult:
    intelligence_task_id: str
    status: EvolutionIntelligenceRunStatus
    audit: EvolutionIntelligenceAuditMetadata
    analysis_findings: tuple[EvolutionAnalysisFinding, ...]
    metric_snapshots: tuple[EvolutionMetricSnapshot, ...]
    insights: tuple[EvolutionIntelligenceInsight, ...]
    recommendations: tuple[EvolutionIntelligenceRecommendation, ...]


__all__ = [
    "ENTERPRISE_EVOLUTION_INTELLIGENCE_TASK_ID",
    "ENTERPRISE_EVOLUTION_INTELLIGENCE_VERSION",
    "AdaptationExecutionResult",
    "ControlledEvolutionRecord",
    "EvolutionAnalysisFinding",
    "EvolutionGovernanceReference",
    "EvolutionHealthObservation",
    "EvolutionHistoryReference",
    "EvolutionIntelligenceAuditMetadata",
    "EvolutionIntelligenceContext",
    "EvolutionIntelligenceDataSourceRef",
    "EvolutionIntelligenceInsight",
    "EvolutionIntelligenceRecommendation",
    "EvolutionIntelligenceResult",
    "EvolutionIntelligenceRunStatus",
    "EvolutionMetricSnapshot",
    "EvolutionOperationRecord",
    "EvolutionRecommendationKind",
    "SelfImprovementGovernanceDecision",
]
