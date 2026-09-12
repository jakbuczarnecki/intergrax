# © Artur Czarnecki. All rights reserved.

"""Typed contracts for production decision orchestration (DS-E2E-15J-L6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing.contracts import (
    GovernanceDecision,
    GovernancePolicyRef,
    GovernanceTaskContext,
    ModelCapabilityProfile,
)
from testing_support.decision_e2e.model_matrix.model_selection_recommendation.contracts import (
    ModelSelectionRecommendation,
    ModelSelectionRequest,
)

ORCHESTRATION_TASK_ID = "DS-E2E-15J-L6.PRODUCTION-DECISION-ORCHESTRATION"
ORCHESTRATION_VERSION = "1"


class DecisionOrchestrationLifecycleStage(StrEnum):
    SELECTED = "selected"
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    REQUIRE_APPROVAL = "require_approval"
    EXECUTED = "executed"
    STOPPED = "stopped"


class DecisionOrchestrationOutcome(StrEnum):
    SUCCESS = "success"
    GOVERNANCE_BLOCKED = "governance_blocked"
    APPROVAL_REQUIRED = "approval_required"
    PROVIDER_MISSING = "provider_missing"


class DecisionExecutionStatus(StrEnum):
    EXECUTED = "executed"


@dataclass(frozen=True, slots=True)
class DecisionOrchestrationRequest:
    selection_request: ModelSelectionRequest
    governance_task_context: GovernanceTaskContext
    applicable_policies: tuple[GovernancePolicyRef, ...]
    capability_evidence: tuple[ModelCapabilityProfile, ...]


@dataclass(frozen=True, slots=True)
class DecisionExecutionRequest:
    selection_result: ModelSelectionRecommendation
    governance_decision: GovernanceDecision


@dataclass(frozen=True, slots=True)
class DecisionExecutionResultReference:
    execution_reference_id: str
    provider_id: str
    status: DecisionExecutionStatus
    profile_key: str | None
    governance_decision_id: str
    selection_task_id: str


@dataclass(frozen=True, slots=True)
class DecisionOrchestrationLifecycleMetadata:
    orchestration_task_id: str
    orchestration_version: str
    orchestrated_at: datetime
    scenario_id: str
    lifecycle_stages: tuple[DecisionOrchestrationLifecycleStage, ...]
    selection_provider_id: str
    governance_provider_id: str
    execution_provider_id: str | None


@dataclass(frozen=True, slots=True)
class DecisionOrchestrationResult:
    outcome: DecisionOrchestrationOutcome
    selection_result: ModelSelectionRecommendation
    governance_result: GovernanceDecision | None
    execution_result_reference: DecisionExecutionResultReference | None
    lifecycle_metadata: DecisionOrchestrationLifecycleMetadata


__all__ = [
    "ORCHESTRATION_TASK_ID",
    "ORCHESTRATION_VERSION",
    "DecisionExecutionRequest",
    "DecisionExecutionResultReference",
    "DecisionExecutionStatus",
    "DecisionOrchestrationLifecycleMetadata",
    "DecisionOrchestrationLifecycleStage",
    "DecisionOrchestrationOutcome",
    "DecisionOrchestrationRequest",
    "DecisionOrchestrationResult",
    "GovernanceDecision",
    "ModelSelectionRecommendation",
]
