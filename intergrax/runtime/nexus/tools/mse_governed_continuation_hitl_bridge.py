# © Artur Czarnecki. All rights reserved.

"""Bridge MSE REQUIRE_HUMAN / ESCALATE → canonical governed continuation HITL pause."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.runtime.agent_governance.errors import ToolGovernanceApprovalRequiredError
from intergrax.runtime.governance.active_governed_execution_task import (
    peek_governed_execution_task,
)
from intergrax.runtime.human.governed_continuation_bridge import (
    apply_governed_continuation_pause,
    bridge_governed_continuation_to_governance,
)
from intergrax.runtime.interrupts.handler import GovernanceResolution
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.runtime.nexus.tracing.tools.tool_invocation import ToolInvocationErrorDiagV1
from intergrax.runtime.task.task import TaskState
from intergrax.tools.execution_models import ToolExecutionRequest

if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


@dataclass(frozen=True)
class GovernedContinuationHitlPauseRequired(RuntimeError):
    """Typed pause control-flow for MSE governed continuation — not TOOL_ERROR."""

    governed_continuation_request: GovernedContinuationRequest
    governance: GovernanceResolution

    def __str__(self) -> str:
        req = self.governed_continuation_request
        return (
            f"Meaningful side-effect governance requires human judgment "
            f"(continuation_request_id={req.continuation_request_id}, "
            f"operation_id={req.operation_id})."
        )


def raise_mse_governed_continuation_hitl_pause(
    error: ToolGovernanceApprovalRequiredError,
    *,
    state: RuntimeState,
    request: ToolExecutionRequest[object],
    agent_id: str,
) -> None:
    """Materialize canonical HITL pause from MSE continuation request, then raise."""
    continuation = error.governed_continuation_request
    if continuation is None:
        raise error

    governance = bridge_governed_continuation_to_governance(continuation)
    task = peek_governed_execution_task()
    if task is not None:
        apply_governed_continuation_pause(task, continuation)
        task.state = TaskState.WAITING_FOR_HUMAN
        task.sync_metadata()

    state.trace_event(
        component=TraceComponent.TOOLS,
        step="mse_governed_continuation_hitl_required",
        message="MSE authorization requires governed continuation human judgment.",
        level=TraceLevel.INFO,
        payload=ToolInvocationErrorDiagV1(
            tool_id=request.tool_id,
            step_id=str(request.step_id),
            error_code=RuntimeErrorCode.PERMISSION_ERROR,
            error_message=str(error),
        ),
    )
    raise GovernedContinuationHitlPauseRequired(
        governed_continuation_request=continuation,
        governance=governance,
    )


__all__ = [
    "GovernedContinuationHitlPauseRequired",
    "raise_mse_governed_continuation_hitl_pause",
]
