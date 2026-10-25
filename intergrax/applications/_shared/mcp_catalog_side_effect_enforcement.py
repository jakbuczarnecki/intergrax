# © Artur Czarnecki. All rights reserved.

"""Shared MSE enforcement projection for MCP catalog invoke (parent + child contexts)."""

from __future__ import annotations

from intergrax.contracts.agent_run import RequestIdentity, canonical_principal_id_from_request_identity
from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    MembershipResolutionMode,
)
from intergrax.contracts.execution_identity import (
    peek_active_execution_id,
    peek_active_execution_identity,
    validate_task_id,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.orchestration_tool_invocation_mse_operation import (
    CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    peek_active_execution_governance_identity,
)
from intergrax.runtime.governance.governance_identity_projection import (
    validate_governance_identity_projection,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskBinding


def build_catalog_side_effect_enforcement_request(
    *,
    binding: ActiveTaskBinding,
    tool_id: str,
    step_id: str,
    caller: RequestIdentity,
) -> CollaborativeWorkEnforcementRequest | None:
    governance_identity = peek_active_execution_governance_identity()
    active_identity = peek_active_execution_identity()
    active_execution_id = peek_active_execution_id()
    if (
        governance_identity is None
        or active_identity is None
        or active_execution_id is None
    ):
        return None
    validate_governance_identity_projection(
        governance_identity,
        tenant_id=binding.task.tenant_id,
    )
    active_run_id, active_attempt_id = active_identity
    if str(active_run_id) != str(binding.run_id):
        return None
    if governance_identity.tenant_id != binding.task.tenant_id:
        return None
    owner_principal = canonical_principal_id_from_request_identity(caller)
    if governance_identity.principal_id != owner_principal:
        return None
    task_id = validate_task_id(binding.task.task_id)
    side_effect = MeaningfulSideEffectRequest(
        action=f"{CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID}:{tool_id}",
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id=f"{tool_id}:{step_id}",
        task_id=task_id,
        run_id=active_run_id,
        attempt_id=active_attempt_id,
        execution_id=active_execution_id,
        principal_id=governance_identity.principal_id,
        tenant_id=governance_identity.tenant_id,
        resource=tool_id,
    )
    return CollaborativeWorkEnforcementRequest(
        tenant_id=governance_identity.tenant_id,
        workspace_id=governance_identity.workspace_id,
        operation_id=CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID,
        acting_principal_id=governance_identity.principal_id,
        resource_scope=tool_id,
        membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        meaningful_side_effect_request=side_effect,
    )


__all__ = ["build_catalog_side_effect_enforcement_request"]
