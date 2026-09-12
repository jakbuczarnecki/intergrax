# © Artur Czarnecki. All rights reserved.

"""Typed contracts for enterprise evolution governance framework (DS-E2E-15J-L17)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.contracts import (
    AdaptationExecutionResult,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.contracts import (
    EvolutionGovernanceReference,
    EvolutionIntelligenceResult,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_operations.contracts import (
    EvolutionOperationRecord,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.contracts import (
    EvolutionStrategyResult,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.contracts import (
    SelfImprovementGovernanceDecision,
)

ENTERPRISE_EVOLUTION_GOVERNANCE_FRAMEWORK_TASK_ID = (
    "DS-E2E-15J-L17.ENTERPRISE-EVOLUTION-GOVERNANCE-FRAMEWORK"
)
ENTERPRISE_EVOLUTION_GOVERNANCE_FRAMEWORK_VERSION = "1"

# Documentation aliases — same types as L15/L16 results.
EnterpriseEvolutionIntelligenceResult = EvolutionIntelligenceResult
EnterpriseEvolutionStrategyResult = EvolutionStrategyResult


class EvolutionGovernanceFrameworkStatus(StrEnum):
    CONSISTENT = "consistent"
    REVIEW_REQUIRED = "review_required"
    INCOMPLETE = "incomplete"
    BLOCKED = "blocked"


class EvolutionGovernanceIssueSeverity(StrEnum):
    REVIEW = "review"
    INCOMPLETE = "incomplete"
    BLOCKED = "blocked"


class EvolutionGovernanceLifecycleStage(StrEnum):
    ANALYSIS = "analysis"
    STRATEGY = "strategy"
    GOVERNANCE = "governance"
    ADAPTATION = "adaptation"
    OPERATIONS = "operations"


@dataclass(frozen=True, slots=True)
class EvolutionGovernanceProcessReference:
    """Immutable pointer to an evolution process under governance review."""

    process_id: str
    version: str
    description: str


@dataclass(frozen=True, slots=True)
class EvolutionGovernanceFrameworkContext:
    """Read-only evolution process snapshot — no runtime or execution handles."""

    scope_id: str
    version: str
    process_reference: EvolutionGovernanceProcessReference | None
    intelligence_result: EvolutionIntelligenceResult | None
    strategy_result: EvolutionStrategyResult | None
    governance_decision: SelfImprovementGovernanceDecision | None
    governance_reference: EvolutionGovernanceReference | None
    execution_results: tuple[AdaptationExecutionResult, ...]
    operation_records: tuple[EvolutionOperationRecord, ...]


@dataclass(frozen=True, slots=True)
class EvolutionGovernanceIssue:
    issue_id: str
    severity: EvolutionGovernanceIssueSeverity
    issue_code: str
    summary: str
    lifecycle_stage: EvolutionGovernanceLifecycleStage | None
    provider_id: str
    provider_version: str


@dataclass(frozen=True, slots=True)
class EvolutionGovernanceFrameworkAuditMetadata:
    framework_task_id: str
    framework_layer_version: str
    scope_id: str
    scope_version: str
    lifecycle_provider_id: str
    lifecycle_provider_version: str
    policy_provider_ids: tuple[str, ...]
    policy_provider_versions: tuple[str, ...]
    control_provider_ids: tuple[str, ...]
    control_provider_versions: tuple[str, ...]
    governance_provider_ids: tuple[str, ...]
    governance_provider_versions: tuple[str, ...]
    process_reference: EvolutionGovernanceProcessReference | None
    data_source_refs: tuple[str, ...]
    evaluation_scope_summary: str
    evaluated_at: datetime


@dataclass(frozen=True, slots=True)
class EvolutionGovernanceFrameworkResult:
    framework_task_id: str
    status: EvolutionGovernanceFrameworkStatus
    issues: tuple[EvolutionGovernanceIssue, ...]
    lifecycle_stages_present: tuple[EvolutionGovernanceLifecycleStage, ...]
    audit: EvolutionGovernanceFrameworkAuditMetadata


__all__ = [
    "ENTERPRISE_EVOLUTION_GOVERNANCE_FRAMEWORK_TASK_ID",
    "ENTERPRISE_EVOLUTION_GOVERNANCE_FRAMEWORK_VERSION",
    "AdaptationExecutionResult",
    "EnterpriseEvolutionIntelligenceResult",
    "EnterpriseEvolutionStrategyResult",
    "EvolutionGovernanceFrameworkAuditMetadata",
    "EvolutionGovernanceFrameworkContext",
    "EvolutionGovernanceFrameworkResult",
    "EvolutionGovernanceFrameworkStatus",
    "EvolutionGovernanceIssue",
    "EvolutionGovernanceIssueSeverity",
    "EvolutionGovernanceLifecycleStage",
    "EvolutionGovernanceProcessReference",
    "EvolutionGovernanceReference",
    "EvolutionIntelligenceResult",
    "EvolutionOperationRecord",
    "EvolutionStrategyResult",
    "SelfImprovementGovernanceDecision",
]
