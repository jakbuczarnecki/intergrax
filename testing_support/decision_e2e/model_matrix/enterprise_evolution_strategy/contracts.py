# © Artur Czarnecki. All rights reserved.

"""Typed contracts for enterprise evolution strategy (DS-E2E-15J-L16)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.contracts import (
    AdaptationExecutionResult,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.contracts import (
    EvolutionGovernanceReference,
    EvolutionHistoryReference,
    EvolutionIntelligenceResult,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_operations.contracts import (
    EvolutionOperationRecord,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.contracts import (
    SelfImprovementGovernanceDecision,
)

ENTERPRISE_EVOLUTION_STRATEGY_TASK_ID = "DS-E2E-15J-L16.ENTERPRISE-EVOLUTION-STRATEGY"
ENTERPRISE_EVOLUTION_STRATEGY_VERSION = "1"


class EvolutionStrategyRunStatus(StrEnum):
    COMPLETE = "complete"
    INSUFFICIENT_INPUT = "insufficient_input"


class EvolutionStrategyRecommendationKind(StrEnum):
    EXPLORE = "explore"
    REVIEW = "review"
    COLLECT_MORE_DATA = "collect_more_data"
    NO_ACTION = "no_action"


@dataclass(frozen=True, slots=True)
class EvolutionStrategyDataSourceRef:
    source_id: str
    source_kind: str
    description: str


@dataclass(frozen=True, slots=True)
class EvolutionCapabilityObservation:
    """Read-only capability trend observation — not a change directive."""

    observation_id: str
    capability_name: str
    trend_direction: str
    summary: str
    data_source_refs: tuple[EvolutionStrategyDataSourceRef, ...]


@dataclass(frozen=True, slots=True)
class EvolutionStrategicConstraint:
    """Boundary for strategic analysis — does not enforce runtime policy."""

    constraint_id: str
    constraint_kind: str
    description: str


@dataclass(frozen=True, slots=True)
class EvolutionStrategyContext:
    """Read-only strategic analysis input — no runtime or execution engine references."""

    scope_id: str
    version: str
    intelligence_result: EvolutionIntelligenceResult | None
    operation_records: tuple[EvolutionOperationRecord, ...]
    execution_results: tuple[AdaptationExecutionResult, ...]
    adaptation_history_refs: tuple[EvolutionHistoryReference, ...]
    governance_reference: EvolutionGovernanceReference | None
    governance_decision: SelfImprovementGovernanceDecision | None
    capability_observations: tuple[EvolutionCapabilityObservation, ...]
    strategic_constraints: tuple[EvolutionStrategicConstraint, ...]


@dataclass(frozen=True, slots=True)
class EvolutionStrategyDirectionFinding:
    finding_id: str
    analyzer_id: str
    analyzer_version: str
    direction_theme: str
    summary: str
    data_source_refs: tuple[EvolutionStrategyDataSourceRef, ...]


@dataclass(frozen=True, slots=True)
class EvolutionFutureScenario:
    scenario_id: str
    title: str
    summary: str
    scenario_provider_id: str
    scenario_provider_version: str
    linked_finding_ids: tuple[str, ...]
    data_source_refs: tuple[EvolutionStrategyDataSourceRef, ...]


@dataclass(frozen=True, slots=True)
class EvolutionScenarioImpactAssessment:
    """Per-scenario impact dimensions — not a single winning strategy score."""

    assessment_id: str
    scenario_id: str
    impact_analyzer_id: str
    impact_analyzer_version: str
    cost_assessment: str
    risk_assessment: str
    complexity_assessment: str
    value_assessment: str
    data_source_refs: tuple[EvolutionStrategyDataSourceRef, ...]


@dataclass(frozen=True, slots=True)
class EvolutionStrategicRecommendation:
    recommendation_id: str
    kind: EvolutionStrategyRecommendationKind
    summary: str
    recommendation_provider_id: str
    recommendation_provider_version: str
    linked_scenario_ids: tuple[str, ...]
    linked_assessment_ids: tuple[str, ...]
    rationale: str


@dataclass(frozen=True, slots=True)
class EvolutionStrategyAuditMetadata:
    strategy_task_id: str
    strategy_layer_version: str
    scope_id: str
    scope_version: str
    strategy_provider_ids: tuple[str, ...]
    strategy_provider_versions: tuple[str, ...]
    strategy_analyzer_ids: tuple[str, ...]
    strategy_analyzer_versions: tuple[str, ...]
    scenario_provider_ids: tuple[str, ...]
    scenario_provider_versions: tuple[str, ...]
    impact_analyzer_ids: tuple[str, ...]
    impact_analyzer_versions: tuple[str, ...]
    recommendation_provider_ids: tuple[str, ...]
    recommendation_provider_versions: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    recommendation_ids: tuple[str, ...]
    data_source_refs: tuple[EvolutionStrategyDataSourceRef, ...]
    analysis_scope_summary: str
    analyzed_at: datetime


@dataclass(frozen=True, slots=True)
class EvolutionStrategyResult:
    strategy_task_id: str
    status: EvolutionStrategyRunStatus
    audit: EvolutionStrategyAuditMetadata
    direction_findings: tuple[EvolutionStrategyDirectionFinding, ...]
    scenarios: tuple[EvolutionFutureScenario, ...]
    impact_assessments: tuple[EvolutionScenarioImpactAssessment, ...]
    recommendations: tuple[EvolutionStrategicRecommendation, ...]


__all__ = [
    "ENTERPRISE_EVOLUTION_STRATEGY_TASK_ID",
    "ENTERPRISE_EVOLUTION_STRATEGY_VERSION",
    "AdaptationExecutionResult",
    "EvolutionCapabilityObservation",
    "EvolutionFutureScenario",
    "EvolutionGovernanceReference",
    "EvolutionHistoryReference",
    "EvolutionIntelligenceResult",
    "EvolutionOperationRecord",
    "EvolutionScenarioImpactAssessment",
    "EvolutionStrategicConstraint",
    "EvolutionStrategicRecommendation",
    "EvolutionStrategyAuditMetadata",
    "EvolutionStrategyContext",
    "EvolutionStrategyDataSourceRef",
    "EvolutionStrategyDirectionFinding",
    "EvolutionStrategyRecommendationKind",
    "EvolutionStrategyResult",
    "EvolutionStrategyRunStatus",
    "SelfImprovementGovernanceDecision",
]
