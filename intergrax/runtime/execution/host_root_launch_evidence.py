# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Host upstream authority evidence for root admission (not trusted authority)."""

from __future__ import annotations

from intergrax.contracts.autonomous_work.execution_authority import validate_authority_scopes
from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
from intergrax.contracts.delegation_authority import (
    ParentExecutionAuthority,
    resolve_root_parent_execution_authority,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.task.task import Task

_DEFAULT_HOST_UPSTREAM_SCOPES: tuple[str, ...] = ("workspace.execute",)


def host_upstream_collaborative_scopes(task: Task) -> tuple[str, ...]:
    authority = resolve_root_parent_execution_authority(task.execution_authority)
    if authority.unrestricted:
        return validate_authority_scopes(_DEFAULT_HOST_UPSTREAM_SCOPES)
    if authority.permission_scopes:
        return validate_authority_scopes(authority.permission_scopes)
    return validate_authority_scopes(_DEFAULT_HOST_UPSTREAM_SCOPES)


def host_upstream_effective_authority_decision() -> EffectiveAuthorityDecision:
    return EffectiveAuthorityDecision(
        decision=PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="host_task_upstream_evidence",
            policy_rule_id="host.root_execution.upstream_evidence",
        ),
    )


def host_workspace_id(task: Task) -> str:
    workspace = (task.metadata.get("workspace_id") or "").strip()
    if workspace:
        return workspace
    tenant = task.tenant_id.strip()
    return tenant or "default"


def host_principal_id(task: Task) -> str:
    principal = (task.user_id or "").strip()
    if principal:
        return principal
    return task.tenant_id.strip() or "anonymous"
