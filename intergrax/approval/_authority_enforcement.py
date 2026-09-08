# © Artur Czarnecki. All rights reserved.

"""Private Approval authority enforcement helpers — reuse MP-1 enforcement gate."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.approval.errors import ApprovalAuthorizationDenied
from intergrax.collaborative_work.enforcement_gate import (
    CollaborativeWorkEnforcementGate,
)
from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
    CollaborativeWorkEnforcementRequest,
    CollaborativeWorkEnforcementResult,
    MembershipResolutionMode,
    WorkspaceMembership,
)
from intergrax.contracts.runtime_policy import PolicyAction


@runtime_checkable
class _ApprovalAuthorityContextRequest(Protocol):
    tenant_id: str
    workspace_id: str
    acting_principal_id: str
    delegator_principal_id: str | None
    membership: WorkspaceMembership | None
    membership_resolution_mode: MembershipResolutionMode
    delegation: AuthorityDelegation | None


def require_approval_allow(
    *,
    enforcement_gate: CollaborativeWorkEnforcementGate,
    operation_id: str,
    request: _ApprovalAuthorityContextRequest,
    resource_scope: str,
) -> CollaborativeWorkEnforcementResult:
    """Evaluate MP-1 authority for one Approval mutation — fail closed on deny."""
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
        raise ApprovalAuthorizationDenied(enforcement_result=result)
    return result
