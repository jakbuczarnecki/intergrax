# © Artur Czarnecki. All rights reserved.

"""Stage-14 proof-only RuntimeToolInvoker adapter for WorkStageToolExecutionPort.

The Stage-14 proof uses zero-input deterministic Tools (``input_schema()`` with no
required fields). This is not a general production Tool-input generation strategy.
"""

from __future__ import annotations

from intergrax.autonomous_work.work_stage_capability_loop import (
    WorkStageToolExecutionPort,
    WorkStageToolExecutionRequest,
    WorkStageToolExecutionResult,
)
from intergrax.contracts.execution_identity import validate_run_id
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.tools.execution_models import ToolExecutionRequest


class RuntimeToolInvokerWorkStagePort(WorkStageToolExecutionPort):
    """Route governed Tool candidates through canonical RuntimeToolInvoker (proof fixture)."""

    def __init__(self, invoker: RuntimeToolInvoker, state: RuntimeState) -> None:
        self._invoker = invoker
        self._state = state

    def execute(self, request: WorkStageToolExecutionRequest) -> WorkStageToolExecutionResult:
        canonical_request_run_id = validate_run_id(request.run_id)
        state_run_id = self._state.run_id
        if canonical_request_run_id != state_run_id:
            raise ValueError(
                "WorkStageToolExecutionRequest.run_id "
                f"({canonical_request_run_id!r}) does not match "
                f"RuntimeState.run_id ({state_run_id!r})",
            )
        tool_id = request.candidate.identity.logical.logical_id
        registered = self._invoker.registry.get(tool_id)
        tool_request = ToolExecutionRequest(
            run_id=canonical_request_run_id,
            tool_id=tool_id,
            step_id=request.step_id,
            input=registered.contract.input_schema(),
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
            run_id=canonical_request_run_id,
            step_id=request.step_id,
        )
