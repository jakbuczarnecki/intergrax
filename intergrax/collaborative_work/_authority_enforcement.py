# © Artur Czarnecki. All rights reserved.

"""Private Collaborative Work authority enforcement helpers shared by domain services."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
    CollaborativeWorkAuthorizationDenied,
    CollaborativeWorkEnforcementRequest,
    CollaborativeWorkEnforcementResult,
    MembershipResolutionMode,
    WorkspaceMembership,
)
from intergrax.contracts.runtime_policy import PolicyAction


@runtime_checkable
class _AuthorityContextRequest(Protocol):
    tenant_id: str
    workspace_id: str
    acting_principal_id: str
    delegator_principal_id: str | None
    membership: WorkspaceMembership | None
    membership_resolution_mode: MembershipResolutionMode
    delegation: AuthorityDelegation | None


def _require_collaborative_allow(
    *,
    enforcement_gate: CollaborativeWorkEnforcementGate,
    operation_id: str,
    request: _AuthorityContextRequest,
    resource_scope: str,
) -> CollaborativeWorkEnforcementResult:
    enforcement_request = CollaborativeWorkEnforcementRequest(
        tenant_id=request.tenant_id,
        workspace_id=request.workspace_id,
        operation_id=operation_id,
        acting_principal_id=request.acting_principal_id,
        delegator_principal_id=request.delegator_principal_id,
        resource_scope=resource_scope,
        membership=request.membership,
        membership_resolution_mode=request.membership_resolution_mode,
        delegation=request.delegation,
    )
    result = enforcement_gate.evaluate(enforcement_request)
    if result.composition.decision.action is not PolicyAction.ALLOW:
        raise CollaborativeWorkAuthorizationDenied(enforcement_result=result)
    return result
