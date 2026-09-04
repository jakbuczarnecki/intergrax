# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical background execution identity bootstrap (BG-EXEC-1 / BG-EXEC-2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
)
from intergrax.runtime.background_execution.identity_persistence import (
    BackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
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


def resolve_background_execution(
    *,
    transport_ref: BackgroundTransportExecutionRef,
    identity_persistence: BackgroundExecutionIdentityPersistence,
    task_tenant_id: str | None = None,
) -> BackgroundExecutionIdentity:
    """
    Resolve stable canonical TaskId/RunId/AttemptId for one transport execution.

    ``TaskRequest.run_id`` and broker message ``run_id`` are transport queue
    correlation only and must not be passed here as canonical ``RunId``.
    """
    tenant_id = _resolve_tenant_scope(
        transport_tenant_id=transport_ref.tenant_id,
        task_tenant_id=task_tenant_id,
    )
    scoped_ref = BackgroundTransportExecutionRef(
        tenant_id=tenant_id,
        provider=transport_ref.provider,
        transport_task_id=transport_ref.transport_task_id,
    )
    persisted = identity_persistence.resolve_or_create(scoped_ref)
    return BackgroundExecutionIdentity(
        tenant_id=tenant_id,
        task_id=persisted.task_id,
        run_id=persisted.run_id,
        attempt_id=persisted.attempt_id,
    )


def bootstrap_background_execution(
    *,
    transport_ref: BackgroundTransportExecutionRef,
    identity_persistence: BackgroundExecutionIdentityPersistence,
    task_tenant_id: str | None = None,
) -> BackgroundExecutionIdentity:
    """Worker-boundary alias for ``resolve_background_execution``."""
    return resolve_background_execution(
        transport_ref=transport_ref,
        identity_persistence=identity_persistence,
        task_tenant_id=task_tenant_id,
    )
