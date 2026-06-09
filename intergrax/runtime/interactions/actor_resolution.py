# © Artur Czarnecki. All rights reserved.

"""Resolve actor identity from task intake (IDEAL-4.1/4.2)."""

from __future__ import annotations

from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.delegation import DelegationSpec
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.runtime.task.task import Task


def resolve_actor_from_task(task: Task) -> ActorIdentity:
    """Map task fields to a typed actor identity."""
    kind = ActorKind.USER
    actor_id = task.user_id or "anonymous"
    if task.metadata.get("actor_kind") == ActorKind.SERVICE.value:
        kind = ActorKind.SERVICE
        actor_id = str(task.metadata.get("service_id", actor_id))
    elif task.metadata.get("actor_kind") == ActorKind.AGENT.value:
        kind = ActorKind.AGENT
        actor_id = str(task.metadata.get("agent_actor_id", task.agent_id or actor_id))

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
    """Child scopes must be subset of parent scopes when parent declares scopes."""
    child_scopes = delegation.permission_scopes
    if not parent.permission_scopes:
        return child_scopes
    if not child_scopes:
        return parent.permission_scopes
    narrowed = tuple(s for s in child_scopes if s in parent.permission_scopes)
    return narrowed
