# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Enterprise healing execution lifecycle states (SELF-HEALING R3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class SelfHealingExecutionLifecycleState(StrEnum):
    CREATED = "CREATED"
    APPROVAL_PENDING = "APPROVAL_PENDING"
    APPROVED = "APPROVED"
    EXECUTION_REQUESTED = "EXECUTION_REQUESTED"
    EXECUTING = "EXECUTING"
    OBSERVING = "OBSERVING"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"
    ROLLBACK_PENDING = "ROLLBACK_PENDING"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"


_ALLOWED: dict[
    SelfHealingExecutionLifecycleState,
    frozenset[SelfHealingExecutionLifecycleState],
] = {
    SelfHealingExecutionLifecycleState.CREATED: frozenset(
        {SelfHealingExecutionLifecycleState.APPROVAL_PENDING, SelfHealingExecutionLifecycleState.APPROVED},
    ),
    SelfHealingExecutionLifecycleState.APPROVAL_PENDING: frozenset(
        {
            SelfHealingExecutionLifecycleState.APPROVED,
            SelfHealingExecutionLifecycleState.FAILED,
        },
    ),
    SelfHealingExecutionLifecycleState.APPROVED: frozenset(
        {SelfHealingExecutionLifecycleState.EXECUTION_REQUESTED},
    ),
    SelfHealingExecutionLifecycleState.EXECUTION_REQUESTED: frozenset(
        {SelfHealingExecutionLifecycleState.EXECUTING},
    ),
    SelfHealingExecutionLifecycleState.EXECUTING: frozenset(
        {
            SelfHealingExecutionLifecycleState.OBSERVING,
            SelfHealingExecutionLifecycleState.FAILED,
        },
    ),
    SelfHealingExecutionLifecycleState.OBSERVING: frozenset(
        {SelfHealingExecutionLifecycleState.VALIDATING},
    ),
    SelfHealingExecutionLifecycleState.VALIDATING: frozenset(
        {
            SelfHealingExecutionLifecycleState.COMPLETED,
            SelfHealingExecutionLifecycleState.ROLLBACK_PENDING,
            SelfHealingExecutionLifecycleState.FAILED,
        },
    ),
    SelfHealingExecutionLifecycleState.ROLLBACK_PENDING: frozenset(
        {
            SelfHealingExecutionLifecycleState.ROLLED_BACK,
            SelfHealingExecutionLifecycleState.FAILED,
        },
    ),
    SelfHealingExecutionLifecycleState.COMPLETED: frozenset(),
    SelfHealingExecutionLifecycleState.ROLLED_BACK: frozenset(),
    SelfHealingExecutionLifecycleState.FAILED: frozenset(),
}


def assert_execution_lifecycle_transition(
    current: SelfHealingExecutionLifecycleState,
    target: SelfHealingExecutionLifecycleState,
) -> None:
    allowed = _ALLOWED.get(current, frozenset())
    if target not in allowed:
        raise ValueError(f"illegal execution lifecycle transition {current.value} -> {target.value}")


@dataclass(frozen=True, slots=True)
class SelfHealingExecutionLifecycleAuditEntry:
    workflow_id: str
    tenant_id: str
    from_state: SelfHealingExecutionLifecycleState
    to_state: SelfHealingExecutionLifecycleState
    actor: str
    reason: str
    recorded_at: datetime
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.workflow_id.startswith("sh_wf_"):
            raise ValueError("workflow_id must be sh_wf_*")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.actor.strip():
            raise ValueError("actor required")


__all__ = [
    "SelfHealingExecutionLifecycleAuditEntry",
    "SelfHealingExecutionLifecycleState",
    "assert_execution_lifecycle_transition",
]
