# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Shared execution scope identity for cross-domain read records (provider-neutral)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)


@runtime_checkable
class ExecutionScopeIdentity(Protocol):
    """Immutable provenance required to verify a read-model record belongs to one execution scope."""

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None
    execution_id: ExecutionId


__all__ = ["ExecutionScopeIdentity"]
