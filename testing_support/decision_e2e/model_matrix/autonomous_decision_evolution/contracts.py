# © Artur Czarnecki. All rights reserved.

"""Typed contracts for autonomous decision evolution (DS-E2E-15J-L11)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.contracts import (
    AdaptiveDecisionInsight,
)

AUTONOMOUS_EVOLUTION_TASK_ID = "DS-E2E-15J-L11.AUTONOMOUS-DECISION-EVOLUTION"
AUTONOMOUS_EVOLUTION_VERSION = "1"


class EvolutionRunStatus(StrEnum):
    COMPLETE = "complete"
    INSUFFICIENT_INPUT = "insufficient_input"


class EvolutionTargetArea(StrEnum):
    MODEL_SELECTION = "model_selection"
    POLICY = "policy"
    PROCESS = "process"


class EvolutionApprovalOutcome(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"


class EvaluationCriterionKind(StrEnum):
    QUALITY = "quality"
    SAFETY = "safety"
    COST = "cost"
    RELIABILITY = "reliability"


@dataclass(frozen=True, slots=True)
class EvolutionRiskInformation:
    risk_summary: str
    mitigation_notes: str


@dataclass(frozen=True, slots=True)
class DecisionEvolutionProposal:
    """Proposed opportunity for improvement — not an executable change."""

    proposal_id: str
    generator_id: str
    generator_version: str
    source_insight_ids: tuple[str, ...]
    target_area: EvolutionTargetArea
    proposed_change: str
    expected_benefit: str
    risk_information: EvolutionRiskInformation
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ExperimentVariant:
    variant_id: str
    variant_label: str
    variant_description: str


@dataclass(frozen=True, slots=True)
class EvaluationCriterion:
    criterion_id: str
    criterion_kind: EvaluationCriterionKind
    description: str


@dataclass(frozen=True, slots=True)
class EvolutionExperimentSpec:
    """Controlled experiment design — does not deploy to production."""

    experiment_id: str
    proposal_id: str
    experiment_provider_id: str
    experiment_provider_version: str
    variants: tuple[ExperimentVariant, ...]
    evaluation_criteria: tuple[EvaluationCriterion, ...]
    experiment_summary: str


@dataclass(frozen=True, slots=True)
class EvolutionEvaluationFinding:
    """Single evaluator output — no aggregated universal score."""

    evaluation_id: str
    experiment_id: str
    evaluator_id: str
    evaluator_version: str
    criterion_kind: EvaluationCriterionKind
    finding_summary: str
    supports_proposal: bool


@dataclass(frozen=True, slots=True)
class EvolutionApprovalDecision:
    approval_id: str
    proposal_id: str
    experiment_id: str
    outcome: EvolutionApprovalOutcome
    approval_provider_id: str
    approval_provider_version: str
    rationale: str


@dataclass(frozen=True, slots=True)
class ControlledEvolutionRecord:
    """Auditable trace: insight → proposal → experiment → evaluation → approval."""

    record_id: str
    source_insight_ids: tuple[str, ...]
    proposal: DecisionEvolutionProposal
    experiment: EvolutionExperimentSpec | None
    evaluation_findings: tuple[EvolutionEvaluationFinding, ...]
    approval: EvolutionApprovalDecision | None
    proposal_generator_ids: tuple[str, ...]
    experiment_provider_ids: tuple[str, ...]
    evaluation_provider_ids: tuple[str, ...]
    approval_provider_id: str | None


@dataclass(frozen=True, slots=True)
class AutonomousDecisionEvolutionAuditMetadata:
    engine_task_id: str
    engine_version: str
    proposal_generator_ids: tuple[str, ...]
    experiment_provider_ids: tuple[str, ...]
    evaluation_provider_ids: tuple[str, ...]
    approval_provider_ids: tuple[str, ...]
    source_insight_ids: tuple[str, ...]
    planned_at: datetime


@dataclass(frozen=True, slots=True)
class AutonomousDecisionEvolutionInput:
    """Read-only insights from adaptive intelligence — no runtime mutation."""

    insights: tuple[AdaptiveDecisionInsight, ...] = ()


@dataclass(frozen=True, slots=True)
class AutonomousDecisionEvolutionResult:
    evolution_task_id: str
    status: EvolutionRunStatus
    audit: AutonomousDecisionEvolutionAuditMetadata
    proposals: tuple[DecisionEvolutionProposal, ...]
    experiments: tuple[EvolutionExperimentSpec, ...]
    evaluation_findings: tuple[EvolutionEvaluationFinding, ...]
    approval_decisions: tuple[EvolutionApprovalDecision, ...]
    evolution_records: tuple[ControlledEvolutionRecord, ...]


__all__ = [
    "AUTONOMOUS_EVOLUTION_TASK_ID",
    "AUTONOMOUS_EVOLUTION_VERSION",
    "AutonomousDecisionEvolutionAuditMetadata",
    "AutonomousDecisionEvolutionInput",
    "AutonomousDecisionEvolutionResult",
    "ControlledEvolutionRecord",
    "DecisionEvolutionProposal",
    "EvaluationCriterion",
    "EvaluationCriterionKind",
    "EvolutionApprovalDecision",
    "EvolutionApprovalOutcome",
    "EvolutionEvaluationFinding",
    "EvolutionExperimentSpec",
    "EvolutionRiskInformation",
    "EvolutionRunStatus",
    "EvolutionTargetArea",
    "ExperimentVariant",
]
