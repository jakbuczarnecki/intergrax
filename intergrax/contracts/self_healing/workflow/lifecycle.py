# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing workflow lifecycle (SELF-HEALING R2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import uuid4


class SelfHealingWorkflowState(StrEnum):
    CREATED = "CREATED"
    PLANNED = "PLANNED"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    APPROVED = "APPROVED"
    EXECUTING = "EXECUTING"
    VALIDATING = "VALIDATING"
    SUCCEEDED = "SUCCEEDED"
    ROLLBACK_REQUIRED = "ROLLBACK_REQUIRED"
    ROLLING_BACK = "ROLLING_BACK"
    FAILED = "FAILED"
    ESCALATED = "ESCALATED"


_ALLOWED: dict[SelfHealingWorkflowState, frozenset[SelfHealingWorkflowState]] = {
    SelfHealingWorkflowState.CREATED: frozenset({SelfHealingWorkflowState.PLANNED}),
    SelfHealingWorkflowState.PLANNED: frozenset(
        {
            SelfHealingWorkflowState.WAITING_APPROVAL,
            SelfHealingWorkflowState.APPROVED,
            SelfHealingWorkflowState.FAILED,
        },
    ),
    SelfHealingWorkflowState.WAITING_APPROVAL: frozenset(
        {
            SelfHealingWorkflowState.APPROVED,
            SelfHealingWorkflowState.FAILED,
            SelfHealingWorkflowState.ESCALATED,
        },
    ),
    SelfHealingWorkflowState.APPROVED: frozenset({SelfHealingWorkflowState.EXECUTING}),
    SelfHealingWorkflowState.EXECUTING: frozenset(
        {
            SelfHealingWorkflowState.VALIDATING,
            SelfHealingWorkflowState.FAILED,
        },
    ),
    SelfHealingWorkflowState.VALIDATING: frozenset(
        {
            SelfHealingWorkflowState.SUCCEEDED,
            SelfHealingWorkflowState.ROLLBACK_REQUIRED,
            SelfHealingWorkflowState.FAILED,
        },
    ),
    SelfHealingWorkflowState.ROLLBACK_REQUIRED: frozenset(
        {SelfHealingWorkflowState.ROLLING_BACK},
    ),
    SelfHealingWorkflowState.ROLLING_BACK: frozenset(
        {
            SelfHealingWorkflowState.FAILED,
            SelfHealingWorkflowState.SUCCEEDED,
        },
    ),
    SelfHealingWorkflowState.SUCCEEDED: frozenset(),
    SelfHealingWorkflowState.FAILED: frozenset(),
    SelfHealingWorkflowState.ESCALATED: frozenset(),
}


def mint_self_healing_workflow_id() -> str:
    return f"sh_wf_{uuid4().hex}"


def assert_workflow_transition(
    current: SelfHealingWorkflowState,
    target: SelfHealingWorkflowState,
) -> None:
    allowed = _ALLOWED.get(current, frozenset())
    if target not in allowed:
        raise ValueError(f"illegal workflow transition {current.value} -> {target.value}")


@dataclass(frozen=True, slots=True)
class SelfHealingWorkflowAuditEntry:
    workflow_id: str
    tenant_id: str
    from_state: SelfHealingWorkflowState
    to_state: SelfHealingWorkflowState
    actor: str
    reason: str
    recorded_at: datetime

    def __post_init__(self) -> None:
        if not self.workflow_id.startswith("sh_wf_"):
            raise ValueError("workflow_id must be sh_wf_*")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.actor.strip():
            raise ValueError("actor required")


__all__ = [
    "SelfHealingWorkflowAuditEntry",
    "SelfHealingWorkflowState",
    "assert_workflow_transition",
    "mint_self_healing_workflow_id",
]
