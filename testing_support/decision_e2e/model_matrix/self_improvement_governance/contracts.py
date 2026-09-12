# © Artur Czarnecki. All rights reserved.

"""Typed contracts for enterprise self-improvement governance (DS-E2E-15J-L12)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.contracts import (
    DecisionEvolutionProposal,
    EvolutionEvaluationFinding,
    EvolutionExperimentSpec,
)
from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleRecord,
)

SELF_IMPROVEMENT_GOVERNANCE_TASK_ID = (
    "DS-E2E-15J-L12.ENTERPRISE-SELF-IMPROVEMENT-GOVERNANCE"
)
SELF_IMPROVEMENT_GOVERNANCE_VERSION = "1"


class SelfImprovementGovernanceStatus(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"
    REQUIRES_MORE_EVIDENCE = "requires_more_evidence"


class EvolutionRiskImpactLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class EvolutionRiskFinding:
    finding_id: str
    evaluator_id: str
    evaluator_version: str
    impact_level: EvolutionRiskImpactLevel
    summary: str
    consequence_description: str
    required_controls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvolutionRiskContext:
    proposal_id: str
    findings: tuple[EvolutionRiskFinding, ...]


@dataclass(frozen=True, slots=True)
class SelfImprovementGovernanceReason:
    reason_code: str
    summary: str
    policy_id: str | None


@dataclass(frozen=True, slots=True)
class SelfImprovementPolicyRef:
    policy_id: str
    policy_version: str


@dataclass(frozen=True, slots=True)
class SelfImprovementGovernanceRequest:
    evolution_proposal: DecisionEvolutionProposal
    experiment_result: EvolutionExperimentSpec
    evaluation_findings: tuple[EvolutionEvaluationFinding, ...]
    risk_context: EvolutionRiskContext
    lifecycle_records: tuple[DecisionLifecycleRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class SelfImprovementGovernanceAuditMetadata:
    governance_task_id: str
    governance_version: str
    proposal_reference: str
    experiment_reference: str
    evaluation_references: tuple[str, ...]
    policy_references: tuple[SelfImprovementPolicyRef, ...]
    risk_evaluator_ids: tuple[str, ...]
    risk_evaluator_versions: tuple[str, ...]
    policy_evaluator_ids: tuple[str, ...]
    policy_evaluator_versions: tuple[str, ...]
    approval_provider_id: str | None
    approval_provider_version: str | None
    evaluated_at: datetime


@dataclass(frozen=True, slots=True)
class SelfImprovementGovernanceDecision:
    status: SelfImprovementGovernanceStatus
    reasons: tuple[SelfImprovementGovernanceReason, ...]
    required_actions: tuple[str, ...]
    policy_references: tuple[SelfImprovementPolicyRef, ...]
    audit_metadata: SelfImprovementGovernanceAuditMetadata


__all__ = [
    "SELF_IMPROVEMENT_GOVERNANCE_TASK_ID",
    "SELF_IMPROVEMENT_GOVERNANCE_VERSION",
    "DecisionLifecycleRecord",
    "EvolutionRiskContext",
    "EvolutionRiskFinding",
    "EvolutionRiskImpactLevel",
    "SelfImprovementGovernanceAuditMetadata",
    "SelfImprovementGovernanceDecision",
    "SelfImprovementGovernanceReason",
    "SelfImprovementGovernanceRequest",
    "SelfImprovementGovernanceStatus",
    "SelfImprovementPolicyRef",
]
