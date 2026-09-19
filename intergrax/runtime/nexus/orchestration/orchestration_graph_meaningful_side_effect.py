# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Project orchestration graph / non-tool consequential effects into MSE enforcement requests."""

from __future__ import annotations

from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    MembershipResolutionMode,
)
from intergrax.contracts.execution_identity import (
    require_active_execution_id,
    require_active_execution_identity,
    validate_task_id,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.orchestration_topology import OrchestrationSlotId
from intergrax.runtime.governance.active_execution_governance_identity import (
    require_active_execution_governance_identity,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    peek_governed_execution_task,
)
from intergrax.runtime.governance.governance_identity_projection import (
    validate_governance_identity_projection,
)

ORCHESTRATION_GRAPH_SLOT_ACTION_PREFIX = "orchestration.graph_slot"


def build_orchestration_graph_slot_meaningful_side_effect_request(
    *,
    slot_id: OrchestrationSlotId,
    operation_id: str,
    resource_scope: str,
    side_effect_scope_id: str,
    kinds: tuple[MeaningfulSideEffectKind, ...],
    external_target: str | None = None,
) -> MeaningfulSideEffectRequest:
    """Pure projection from active execution + slot identity — no authorization semantics."""
    governance_identity = require_active_execution_governance_identity()
    host_task = peek_governed_execution_task()
    if host_task is None:
        raise RuntimeError(
            "orchestration graph consequential effect requires an active governed host task",
        )
    validate_governance_identity_projection(
        governance_identity,
        tenant_id=host_task.tenant_id,
    )
    tenant_id = host_task.tenant_id
    active_run_id, active_attempt_id = require_active_execution_identity()
    active_execution_id = require_active_execution_id()
    task_id = validate_task_id(host_task.task_id)
    normalized_operation = operation_id.strip()
    if not normalized_operation:
        raise ValueError("operation_id must be non-empty")
    normalized_resource = resource_scope.strip()
    if not normalized_resource:
        raise ValueError("resource_scope must be non-empty")
    return MeaningfulSideEffectRequest(
        action=f"{ORCHESTRATION_GRAPH_SLOT_ACTION_PREFIX}:{normalized_operation}",
        kinds=kinds,
        side_effect_scope_id=side_effect_scope_id,
        task_id=task_id,
        run_id=active_run_id,
        attempt_id=active_attempt_id,
        execution_id=active_execution_id,
        principal_id=governance_identity.principal_id,
        tenant_id=tenant_id,
        resource=normalized_resource,
        external_target=external_target,
        correlation={"orchestration_slot_id": str(slot_id)},
    )


def build_orchestration_graph_slot_enforcement_request(
    *,
    slot_id: OrchestrationSlotId,
    side_effect: MeaningfulSideEffectRequest,
    operation_id: str,
    resource_scope: str,
) -> CollaborativeWorkEnforcementRequest:
    """Wrap a typed side-effect request for canonical MSE authorization."""
    governance_identity = require_active_execution_governance_identity()
    host_task = peek_governed_execution_task()
    if host_task is None:
        raise RuntimeError(
            "orchestration graph consequential effect requires an active governed host task",
        )
    validate_governance_identity_projection(
        governance_identity,
        tenant_id=host_task.tenant_id,
    )
    tenant_id = host_task.tenant_id
    workspace_id = governance_identity.workspace_id
    normalized_operation = operation_id.strip()
    if not normalized_operation:
        raise ValueError("operation_id must be non-empty")
    normalized_resource = resource_scope.strip()
    if not normalized_resource:
        raise ValueError("resource_scope must be non-empty")
    return CollaborativeWorkEnforcementRequest(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        operation_id=normalized_operation,
        acting_principal_id=governance_identity.principal_id,
        resource_scope=normalized_resource,
        membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        meaningful_side_effect_request=side_effect,
    )


__all__ = [
    "ORCHESTRATION_GRAPH_SLOT_ACTION_PREFIX",
    "build_orchestration_graph_slot_enforcement_request",
    "build_orchestration_graph_slot_meaningful_side_effect_request",
]
