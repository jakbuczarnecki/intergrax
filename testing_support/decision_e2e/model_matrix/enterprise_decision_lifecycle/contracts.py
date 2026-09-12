# © Artur Czarnecki. All rights reserved.

"""Typed contracts for enterprise decision lifecycle (DS-E2E-15J-L7)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

LIFECYCLE_TASK_ID = "DS-E2E-15J-L7.ENTERPRISE-DECISION-LIFECYCLE"
LIFECYCLE_VERSION = "1"


class DecisionLifecycleState(StrEnum):
    CREATED = "created"
    EVALUATING = "evaluating"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class DecisionType(StrEnum):
    PRODUCTION_MODEL_ROUTING = "production_model_routing"


class DecisionSourceKind(StrEnum):
    MODEL_SELECTION = "model_selection"
    GOVERNANCE = "governance"
    EXECUTION = "execution"
    ORCHESTRATION = "orchestration"


@dataclass(frozen=True, slots=True)
class DecisionSourceReference:
    source_kind: DecisionSourceKind
    reference_id: str


@dataclass(frozen=True, slots=True)
class DecisionLifecycleActorRef:
    actor_kind: str
    reference_id: str


@dataclass(frozen=True, slots=True)
class DecisionLifecycleRecord:
    decision_id: str
    decision_type: DecisionType
    lifecycle_state: DecisionLifecycleState
    created_at: datetime
    source_references: tuple[DecisionSourceReference, ...]


@dataclass(frozen=True, slots=True)
class DecisionLifecycleEvent:
    decision_id: str
    previous_state: DecisionLifecycleState | None
    new_state: DecisionLifecycleState
    reason: str
    timestamp: datetime
    actor: DecisionLifecycleActorRef


__all__ = [
    "LIFECYCLE_TASK_ID",
    "LIFECYCLE_VERSION",
    "DecisionLifecycleActorRef",
    "DecisionLifecycleEvent",
    "DecisionLifecycleRecord",
    "DecisionLifecycleState",
    "DecisionSourceKind",
    "DecisionSourceReference",
    "DecisionType",
]
