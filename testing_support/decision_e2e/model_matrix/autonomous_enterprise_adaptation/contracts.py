# © Artur Czarnecki. All rights reserved.

"""Typed contracts for controlled enterprise adaptation (DS-E2E-15J-L13)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.contracts import (
    ControlledEvolutionRecord,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.contracts import (
    SelfImprovementGovernanceStatus,
)

AUTONOMOUS_ENTERPRISE_ADAPTATION_TASK_ID = (
    "DS-E2E-15J-L13.AUTONOMOUS-ENTERPRISE-ADAPTATION"
)
AUTONOMOUS_ENTERPRISE_ADAPTATION_VERSION = "1"


class AdaptationExecutionStatus(StrEnum):
    APPLIED = "applied"
    REJECTED = "rejected"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"


@dataclass(frozen=True, slots=True)
class EvolutionSourceReference:
    controlled_evolution_record_id: str
    proposal_id: str
    source_insight_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GovernanceApprovalReference:
    approval_id: str
    governance_status: SelfImprovementGovernanceStatus
    approver_identity: str
    governance_decision_reference: str


@dataclass(frozen=True, slots=True)
class AdaptationScope:
    scope_id: str
    target_system_area: str
    scope_label: str


@dataclass(frozen=True, slots=True)
class AdaptationConstraint:
    constraint_id: str
    scope_id: str
    system_area: str
    adaptation_version: str
    valid_until: datetime | None = None


@dataclass(frozen=True, slots=True)
class ApprovedAdaptationRequest:
    """Governed adaptation input — not a raw executable change command."""

    adaptation_id: str
    version: str
    source_reference: EvolutionSourceReference
    governance_approval: GovernanceApprovalReference | None
    scope: AdaptationScope
    constraints: tuple[AdaptationConstraint, ...]
    controlled_evolution: ControlledEvolutionRecord | None = None


@dataclass(frozen=True, slots=True)
class AdaptationAuditMetadata:
    adaptation_task_id: str
    adaptation_layer_version: str
    adaptation_id: str
    adaptation_version: str
    source_reference: EvolutionSourceReference
    governance_approval_id: str | None
    approver_identity: str | None
    provider_id: str
    provider_version: str
    applied_change_reference: str | None
    outcome_status: AdaptationExecutionStatus
    executed_at: datetime
    outcome_summary: str


@dataclass(frozen=True, slots=True)
class AdaptationExecutionResult:
    status: AdaptationExecutionStatus
    provider_id: str
    provider_version: str
    applied_change_reference: str | None
    audit_metadata: AdaptationAuditMetadata
    adaptation_id: str
    version: str
    source_reference: EvolutionSourceReference


__all__ = [
    "AUTONOMOUS_ENTERPRISE_ADAPTATION_TASK_ID",
    "AUTONOMOUS_ENTERPRISE_ADAPTATION_VERSION",
    "AdaptationAuditMetadata",
    "AdaptationConstraint",
    "AdaptationExecutionResult",
    "AdaptationExecutionStatus",
    "AdaptationScope",
    "ApprovedAdaptationRequest",
    "ControlledEvolutionRecord",
    "EvolutionSourceReference",
    "GovernanceApprovalReference",
    "SelfImprovementGovernanceStatus",
]
