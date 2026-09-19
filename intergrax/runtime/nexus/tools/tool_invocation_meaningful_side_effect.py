# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Project orchestration tool invocation into collaborative-work MSE enforcement requests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    MembershipResolutionMode,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    require_active_execution_governance_identity,
)
from intergrax.runtime.governance.governance_identity_projection import (
    validate_governance_identity_projection,
)
from intergrax.runtime.nexus.tools.tool_invocation_inner_governance import (
    TOOL_INVOCATION_INNER_ACTION_PREFIX,
    build_tool_invocation_inner_governance_request,
)
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest

if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

ORCHESTRATION_TOOL_MSE_OPERATION_ID = TOOL_INVOCATION_INNER_ACTION_PREFIX


def build_tool_invocation_meaningful_side_effect_enforcement_request(
    *,
    state: RuntimeState,
    agent_id: str,
    contract: ToolContract,
    request: ToolExecutionRequest,
) -> CollaborativeWorkEnforcementRequest:
    """Pure projection — no authorization semantics."""
    governance_identity = require_active_execution_governance_identity()
    validate_governance_identity_projection(
        governance_identity,
        tenant_id=state.tenant_id,
    )
    side_effect = build_tool_invocation_inner_governance_request(
        state=state,
        agent_id=agent_id,
        contract=contract,
        request=request,
    )
    resource_scope = contract.tool_id
    return CollaborativeWorkEnforcementRequest(
        tenant_id=governance_identity.tenant_id,
        workspace_id=governance_identity.workspace_id,
        operation_id=ORCHESTRATION_TOOL_MSE_OPERATION_ID,
        acting_principal_id=governance_identity.principal_id,
        resource_scope=resource_scope,
        membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        meaningful_side_effect_request=side_effect,
    )


__all__ = [
    "ORCHESTRATION_TOOL_MSE_OPERATION_ID",
    "build_tool_invocation_meaningful_side_effect_enforcement_request",
]
