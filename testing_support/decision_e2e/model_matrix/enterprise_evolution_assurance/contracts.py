# © Artur Czarnecki. All rights reserved.

"""Typed contracts for enterprise evolution assurance (DS-E2E-15J-L18)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.contracts import (
    AdaptationExecutionResult,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.contracts import (
    EvolutionGovernanceFrameworkResult,
    EvolutionGovernanceProcessReference,
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

ENTERPRISE_EVOLUTION_ASSURANCE_TASK_ID = "DS-E2E-15J-L18.ENTERPRISE-EVOLUTION-ASSURANCE"
ENTERPRISE_EVOLUTION_ASSURANCE_VERSION = "1"


class EvolutionAssuranceStatus(StrEnum):
    PASSED = "passed"
    WARNING = "warning"
    REVIEW_REQUIRED = "review_required"
    FAILED = "failed"


class EvolutionAssuranceFindingSeverity(StrEnum):
    WARNING = "warning"
    REVIEW = "review"
    FAILED = "failed"


class EvolutionAssuranceLifecycleStage(StrEnum):
    ANALYSIS = "analysis"
    STRATEGY = "strategy"
    GOVERNANCE = "governance"
    ADAPTATION = "adaptation"
    OPERATIONS = "operations"
    GOVERNANCE_FRAMEWORK = "governance_framework"


@dataclass(frozen=True, slots=True)
class EvolutionAssuranceContext:
    """Read-only evolution lifecycle snapshot for quality assurance — no runtime handles."""

    scope_id: str
    version: str
    process_reference: EvolutionGovernanceProcessReference | None
    governance_framework_result: EvolutionGovernanceFrameworkResult | None
    intelligence_result: EvolutionIntelligenceResult | None
    strategy_result: EvolutionStrategyResult | None
    governance_decision: SelfImprovementGovernanceDecision | None
    governance_reference: EvolutionGovernanceReference | None
    execution_results: tuple[AdaptationExecutionResult, ...]
    operation_records: tuple[EvolutionOperationRecord, ...]


@dataclass(frozen=True, slots=True)
class EvolutionAssuranceFinding:
    finding_id: str
    severity: EvolutionAssuranceFindingSeverity
    finding_code: str
    summary: str
    lifecycle_stage: EvolutionAssuranceLifecycleStage | None
    provider_id: str
    provider_version: str
    evidence_ref: str | None = None


@dataclass(frozen=True, slots=True)
class EvolutionAssuranceAuditMetadata:
    assurance_task_id: str
    assurance_layer_version: str
    scope_id: str
    scope_version: str
    quality_validator_ids: tuple[str, ...]
    quality_validator_versions: tuple[str, ...]
    compliance_validator_ids: tuple[str, ...]
    compliance_validator_versions: tuple[str, ...]
    evidence_validator_ids: tuple[str, ...]
    evidence_validator_versions: tuple[str, ...]
    assurance_provider_ids: tuple[str, ...]
    assurance_provider_versions: tuple[str, ...]
    process_reference: EvolutionGovernanceProcessReference | None
    evidence_refs: tuple[str, ...]
    finding_ids: tuple[str, ...]
    assessment_scope_summary: str
    assessed_at: datetime


@dataclass(frozen=True, slots=True)
class EvolutionAssuranceResult:
    assurance_task_id: str
    status: EvolutionAssuranceStatus
    findings: tuple[EvolutionAssuranceFinding, ...]
    evidence_refs: tuple[str, ...]
    audit: EvolutionAssuranceAuditMetadata


__all__ = [
    "ENTERPRISE_EVOLUTION_ASSURANCE_TASK_ID",
    "ENTERPRISE_EVOLUTION_ASSURANCE_VERSION",
    "AdaptationExecutionResult",
    "EvolutionAssuranceAuditMetadata",
    "EvolutionAssuranceContext",
    "EvolutionAssuranceFinding",
    "EvolutionAssuranceFindingSeverity",
    "EvolutionAssuranceLifecycleStage",
    "EvolutionAssuranceResult",
    "EvolutionAssuranceStatus",
    "EvolutionGovernanceFrameworkResult",
    "EvolutionGovernanceProcessReference",
    "EvolutionGovernanceReference",
    "EvolutionIntelligenceResult",
    "EvolutionOperationRecord",
    "EvolutionStrategyResult",
    "SelfImprovementGovernanceDecision",
]
