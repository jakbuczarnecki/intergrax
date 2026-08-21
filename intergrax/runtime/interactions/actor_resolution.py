# © Artur Czarnecki. All rights reserved.

"""Resolve actor identity from task intake (IDEAL-4.1/4.2)."""

from __future__ import annotations

from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.delegation import DelegationSpec
from intergrax.contracts.delegation_authority import (
    DelegationAuthorityError,
    ParentExecutionAuthority,
    mint_effective_delegation_authority,
)
from intergrax.contracts.request_identity_spine import request_identity_to_actor_identity
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.runtime.task.task import Task


def resolve_actor_from_task(task: Task) -> ActorIdentity:
    """Map task fields to a typed actor identity."""
    actor_kind_raw = task.metadata.get("actor_kind")
    if actor_kind_raw == ActorKind.SERVICE.value:
        kind = ActorKind.SERVICE
        actor_id = str(task.metadata.get("actor_id") or task.user_id or "anonymous")
    elif actor_kind_raw == ActorKind.AGENT.value:
        kind = ActorKind.AGENT
        actor_id = str(
            task.metadata.get("actor_id") or task.agent_id or task.user_id or "anonymous"
        )
    else:
        kind = ActorKind.USER
        actor_id = task.user_id or "anonymous"

    scopes_raw = task.metadata.get("permission_scopes", ())
    scopes = tuple(scopes_raw) if isinstance(scopes_raw, (list, tuple)) else ()

    return ActorIdentity(
        kind=kind,
        actor_id=actor_id,
        tenant_id=task.tenant_id or "default",
        delegated_from=task.metadata.get("delegated_from"),
        permission_scopes=scopes,
    )


def resolve_actor_from_envelope(envelope: TaskEnvelope) -> ActorIdentity:
    return resolve_actor_from_task(Task.from_envelope(envelope))


def narrow_delegation_scopes(
    parent: ActorIdentity,
    delegation: DelegationSpec,
) -> tuple[str, ...]:
    """Return effective delegated scopes; empty ActorIdentity scopes are unknown, not unlimited."""
    parent_authority = (
        ParentExecutionAuthority.scoped(parent.permission_scopes)
        if parent.permission_scopes
        else ParentExecutionAuthority.unknown()
    )
    try:
        effective = mint_effective_delegation_authority(
            parent=parent_authority,
            requested_permission_scopes=delegation.permission_scopes,
        )
    except DelegationAuthorityError:
        raise
    return effective.effective_permission_scopes
