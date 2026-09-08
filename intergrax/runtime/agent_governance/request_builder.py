# © Artur Czarnecki. All rights reserved.

"""Build governance authorization requests from tool invocation context."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.contracts.agent_runtime_governance import (
    AgentIdentity,
    ToolAuthorizationRequest,
    ToolAuthorizationRiskLevel,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest

if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

_RISK_MAP: dict[ToolRiskLevel, ToolAuthorizationRiskLevel] = {
    ToolRiskLevel.LOW: ToolAuthorizationRiskLevel.LOW,
    ToolRiskLevel.MEDIUM: ToolAuthorizationRiskLevel.MEDIUM,
    ToolRiskLevel.HIGH: ToolAuthorizationRiskLevel.HIGH,
    ToolRiskLevel.CRITICAL: ToolAuthorizationRiskLevel.CRITICAL,
}


def governance_capability_for_contract(contract: ToolContract) -> str:
    """Derive governance capability from tool contract metadata."""
    if contract.category.strip():
        return contract.category.strip()
    return contract.tool_id


def build_tool_authorization_request(
    *,
    state: RuntimeState,
    agent_id: str,
    contract: ToolContract,
    request: ToolExecutionRequest,
) -> ToolAuthorizationRequest:
    """Construct a typed governance request using existing execution identity."""
    execution_id: ExecutionId | None = None
    try:
        from intergrax.contracts.execution_identity import require_active_execution_id

        execution_id = require_active_execution_id()
    except RuntimeError:
        execution_id = None

    approval_ref: str | None = None
    if state.declarative_hitl_grant is not None:
        approval_ref = state.declarative_hitl_grant.grant_id
    elif request.declarative_hitl_invocation_scope_id:
        approval_ref = request.declarative_hitl_invocation_scope_id

    return ToolAuthorizationRequest(
        agent=AgentIdentity(
            agent_id=agent_id,
            tenant_id=state.tenant_id,
        ),
        task_id=validate_task_id(state.task_id),
        run_id=validate_run_id(state.run_id),
        attempt_id=_resolve_attempt_id(state),
        execution_id=execution_id,
        capability=governance_capability_for_contract(contract),
        tool_id=request.tool_id,
        requested_action=f"execute:{request.tool_id}",
        risk_level=_RISK_MAP.get(contract.risk_level, ToolAuthorizationRiskLevel.LOW),
        approval_evidence_ref=approval_ref,
    )


def _resolve_attempt_id(state: RuntimeState) -> AttemptId:
    from intergrax.contracts.execution_identity import require_active_execution_identity

    _, attempt_id = require_active_execution_identity()
    return validate_attempt_id(str(attempt_id))
