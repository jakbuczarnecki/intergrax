# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""RuntimeToolInvoker-backed Tool execution port for Stage-14 reference loop."""

from __future__ import annotations

from pydantic import BaseModel

from intergrax.autonomous_work.work_stage_capability_loop import (
    WorkStageToolExecutionPort,
    WorkStageToolExecutionRequest,
    WorkStageToolExecutionResult,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.tools.execution_models import ToolExecutionRequest


class _EmptyToolInput(BaseModel):
    """Deterministic empty input for reference-loop Tool invocations."""


class RuntimeToolInvokerWorkStagePort(WorkStageToolExecutionPort):
    """Route governed Tool candidates through canonical RuntimeToolInvoker."""

    def __init__(self, invoker: RuntimeToolInvoker, state: RuntimeState) -> None:
        self._invoker = invoker
        self._state = state

    def execute(self, request: WorkStageToolExecutionRequest) -> WorkStageToolExecutionResult:
        tool_id = request.candidate.identity.logical.logical_id
        run_id = self._state.run_id
        self._state._observability_emitter = None
        tool_request = ToolExecutionRequest(
            run_id=run_id,
            tool_id=tool_id,
            step_id=request.step_id,
            input=_EmptyToolInput(),
        )
        result = self._invoker.invoke(
            state=self._state,
            agent_id="work_stage_loop_reference",
            request=tool_request,
        )
        output_summary = "success" if result.success else "failure"
        return WorkStageToolExecutionResult(
            tool_id=tool_id,
            success=result.success,
            output_summary=output_summary,
        )
