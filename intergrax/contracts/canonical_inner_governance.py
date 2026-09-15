# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""GR-3 — canonical inner governance identity binding (contracts only)."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.active_execution_task_scope import ActiveExecutionTaskScopePort
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest


class CanonicalInnerGovernanceViolation(RuntimeError):
    """Meaningful side effect is not bound to the active canonical Execution."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def assert_meaningful_side_effect_matches_active_execution(
    request: MeaningfulSideEffectRequest,
    *,
    active_task_id: TaskId,
    active_run_id: RunId,
    active_attempt_id: AttemptId,
    active_execution_id: ExecutionId,
) -> None:
    """Fail closed when any of the four IDs diverges from active canonical context."""
    if request.task_id != active_task_id:
        raise CanonicalInnerGovernanceViolation(
            reason="meaningful side effect task_id does not match active execution",
        )
    if request.run_id != active_run_id:
        raise CanonicalInnerGovernanceViolation(
            reason="meaningful side effect run_id does not match active execution",
        )
    if request.attempt_id != active_attempt_id:
        raise CanonicalInnerGovernanceViolation(
            reason="meaningful side effect attempt_id does not match active execution",
        )
    if request.execution_id != active_execution_id:
        raise CanonicalInnerGovernanceViolation(
            reason="meaningful side effect execution_id does not match active execution",
        )


class CanonicalInnerExecutionGuardPort(Protocol):
    """Replaceable guard — asserts inner operations stay inside active canonical Execution."""

    def assert_meaningful_side_effect_bound(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> None: ...


def require_active_execution_for_meaningful_side_effect(
    request: MeaningfulSideEffectRequest,
    *,
    task_scope: ActiveExecutionTaskScopePort,
) -> None:
    """Resolve active four-ID context and require exact match with ``request``."""
    active_run_id, active_attempt_id = require_active_execution_identity()
    active_execution_id = require_active_execution_id()
    active_task_id = task_scope.resolve_current_task_scope(
        run_id=active_run_id,
        attempt_id=active_attempt_id,
        execution_id=active_execution_id,
    )
    assert_meaningful_side_effect_matches_active_execution(
        request,
        active_task_id=active_task_id,
        active_run_id=active_run_id,
        active_attempt_id=active_attempt_id,
        active_execution_id=active_execution_id,
    )


__all__ = [
    "CanonicalInnerExecutionGuardPort",
    "CanonicalInnerGovernanceViolation",
    "assert_meaningful_side_effect_matches_active_execution",
    "require_active_execution_for_meaningful_side_effect",
]
