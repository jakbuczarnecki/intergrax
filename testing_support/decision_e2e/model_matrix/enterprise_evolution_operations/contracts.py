# © Artur Czarnecki. All rights reserved.

"""Typed contracts for enterprise evolution operations (DS-E2E-15J-L14)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.contracts import (
    ControlledEvolutionRecord,
)
from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.contracts import (
    AdaptationExecutionResult,
    ApprovedAdaptationRequest,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.contracts import (
    SelfImprovementGovernanceDecision,
)

ENTERPRISE_EVOLUTION_OPERATIONS_TASK_ID = (
    "DS-E2E-15J-L14.ENTERPRISE-EVOLUTION-OPERATIONS"
)
ENTERPRISE_EVOLUTION_OPERATIONS_VERSION = "1"


class EvolutionOperationStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    REVIEW_REQUIRED = "review_required"
    DISABLED = "disabled"
    FAILED = "failed"


class EvolutionOperationType(StrEnum):
    OBSERVE = "observe"
    PAUSE = "pause"
    RESUME = "resume"
    MARK_REVIEW_REQUIRED = "mark_review_required"
    DISABLE = "disable"


class EvolutionOperationalEventKind(StrEnum):
    CREATED = "created"
    APPROVED = "approved"
    ACTIVATED = "activated"
    OBSERVED = "observed"
    REVIEWED = "reviewed"
    PAUSED = "paused"
    RESUMED = "resumed"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class AdaptationOperationalReference:
    """Reference to an applied adaptation under operational management — not a change command."""

    adaptation_id: str
    version: str
    applied_change_reference: str | None
    scope_id: str
    proposal_id: str
    controlled_evolution_record_id: str | None = None


@dataclass(frozen=True, slots=True)
class EvolutionOperationConstraint:
    constraint_id: str
    allowed_operation_types: tuple[EvolutionOperationType, ...]


@dataclass(frozen=True, slots=True)
class EvolutionOperationRequestMetadata:
    operator_identity: str | None = None
    reason_code: str | None = None
    ticket_reference: str | None = None


@dataclass(frozen=True, slots=True)
class EvolutionOperationRequest:
    adaptation_reference: AdaptationOperationalReference
    operation_type: EvolutionOperationType
    constraints: tuple[EvolutionOperationConstraint, ...]
    request_metadata: EvolutionOperationRequestMetadata


@dataclass(frozen=True, slots=True)
class EvolutionHealthObservation:
    adaptation_id: str
    version: str
    operational_status: EvolutionOperationStatus
    quality_indicator: str
    risk_indicator: str
    summary: str


@dataclass(frozen=True, slots=True)
class EvolutionOperationRecord:
    record_id: str
    adaptation_id: str
    version: str
    event_kind: EvolutionOperationalEventKind
    operational_status: EvolutionOperationStatus
    occurred_at: datetime
    operator_identity: str | None
    summary: str


@dataclass(frozen=True, slots=True)
class EvolutionOperationsAuditMetadata:
    operations_task_id: str
    operations_layer_version: str
    adaptation_id: str
    adaptation_version: str
    adaptation_reference: AdaptationOperationalReference
    operations_provider_id: str
    operations_provider_version: str
    health_provider_id: str
    health_provider_version: str
    operation_type: EvolutionOperationType
    outcome_status: EvolutionOperationStatus
    executed_at: datetime
    outcome_summary: str


@dataclass(frozen=True, slots=True)
class EvolutionOperationResult:
    operational_status: EvolutionOperationStatus
    provider_id: str
    provider_version: str
    operation_record: EvolutionOperationRecord
    health_observation: EvolutionHealthObservation
    audit_metadata: EvolutionOperationsAuditMetadata
    adaptation_reference: AdaptationOperationalReference
    operation_type: EvolutionOperationType


__all__ = [
    "ENTERPRISE_EVOLUTION_OPERATIONS_TASK_ID",
    "ENTERPRISE_EVOLUTION_OPERATIONS_VERSION",
    "AdaptationExecutionResult",
    "AdaptationOperationalReference",
    "ApprovedAdaptationRequest",
    "ControlledEvolutionRecord",
    "EvolutionHealthObservation",
    "EvolutionOperationConstraint",
    "EvolutionOperationRecord",
    "EvolutionOperationRequest",
    "EvolutionOperationRequestMetadata",
    "EvolutionOperationResult",
    "EvolutionOperationStatus",
    "EvolutionOperationType",
    "EvolutionOperationalEventKind",
    "EvolutionOperationsAuditMetadata",
    "SelfImprovementGovernanceDecision",
]
