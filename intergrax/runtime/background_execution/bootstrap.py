# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical background execution identity bootstrap (BG-EXEC-1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    validate_task_id,
)


class BackgroundExecutionTenantMismatchError(ValueError):
    """Raised when transport and task tenant scopes disagree."""


@dataclass(frozen=True)
class BackgroundExecutionIdentity:
    """Platform-owned canonical identity for one background execution attempt."""

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId


def _resolve_tenant_scope(
    *,
    transport_tenant_id: str,
    task_tenant_id: str | None,
) -> str:
    transport = transport_tenant_id.strip()
    if not transport:
        raise ValueError("transport_tenant_id must be non-empty")
    if task_tenant_id is None:
        return transport
    task_scope = task_tenant_id.strip()
    if not task_scope:
        raise ValueError("task_tenant_id must be non-empty when provided")
    if transport != task_scope:
        raise BackgroundExecutionTenantMismatchError(
            f"tenant mismatch: transport={transport!r} task={task_scope!r}"
        )
    return transport


def bootstrap_background_execution(
    *,
    transport_tenant_id: str,
    task_tenant_id: str | None = None,
    canonical_task_id: TaskId | None = None,
) -> BackgroundExecutionIdentity:
    """
    Mint canonical runtime identity at the background worker boundary.

    ``TaskRequest.run_id`` and broker message ``run_id`` are transport queue
    correlation only and must not be passed here as canonical ``RunId``.
    """
    tenant_id = _resolve_tenant_scope(
        transport_tenant_id=transport_tenant_id,
        task_tenant_id=task_tenant_id,
    )
    task_id = (
        validate_task_id(canonical_task_id)
        if canonical_task_id is not None
        else mint_task_id()
    )
    return BackgroundExecutionIdentity(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
