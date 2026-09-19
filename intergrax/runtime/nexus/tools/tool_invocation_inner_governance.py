# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Nexus runtime adapter: project active execution context into MeaningfulSideEffectRequest.

Pure projection only — no policy evaluation, allow/deny, or governance semantics ownership.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.contracts.execution_identity import (
    require_active_execution_id,
    require_active_execution_identity,
    validate_task_id,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    require_active_execution_governance_identity,
)
from intergrax.runtime.governance.governance_identity_projection import (
    validate_governance_identity_projection,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest

if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

TOOL_INVOCATION_INNER_ACTION_PREFIX = "orchestration.tool_invocation_authorization"


def build_tool_invocation_inner_governance_request(
    *,
    state: RuntimeState,
    agent_id: str,
    contract: ToolContract,
    request: ToolExecutionRequest,
) -> MeaningfulSideEffectRequest:
    """Typed inner-boundary request — four-ID binding for tool invoke authorization GEP."""
    _ = agent_id  # roster identity only; principal from active governance identity (ADR-GR-10-001)
    governance_identity = require_active_execution_governance_identity()
    validate_governance_identity_projection(
        governance_identity,
        tenant_id=state.tenant_id,
    )
    active_run_id, active_attempt_id = require_active_execution_identity()
    active_execution_id = require_active_execution_id()
    task_id = validate_task_id(state.task_id)
    kinds = (
        (MeaningfulSideEffectKind.MUTATION,)
        if contract.side_effects
        else (MeaningfulSideEffectKind.ACCESS,)
    )
    return MeaningfulSideEffectRequest(
        action=f"{TOOL_INVOCATION_INNER_ACTION_PREFIX}:{request.tool_id}",
        kinds=kinds,
        side_effect_scope_id=f"{request.tool_id}:{request.step_id}",
        task_id=task_id,
        run_id=active_run_id,
        attempt_id=active_attempt_id,
        execution_id=active_execution_id,
        principal_id=governance_identity.principal_id,
        tenant_id=governance_identity.tenant_id,
        resource=contract.tool_id,
    )


__all__ = [
    "TOOL_INVOCATION_INNER_ACTION_PREFIX",
    "build_tool_invocation_inner_governance_request",
]
