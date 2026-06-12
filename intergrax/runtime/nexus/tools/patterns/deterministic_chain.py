# © Artur Czarnecki. All rights reserved.

"""Deterministic tool chain invocation pattern (TOOL-ENG-20)."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState, ToolCallTrace
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_chain_mapper import build_chain_step_input
from intergrax.runtime.nexus.tools.tool_invocation_aggregate import ToolInvocationAggregate
from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationResult
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.runtime.nexus.tools.tool_verify_hooks import emit_high_risk_tool_verify_signal
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan
from intergrax.tools.execution_models import ToolExecutionRequest


class DeterministicChainPattern:
    """Execute ``ToolChainSpec`` steps sequentially with explicit field mapping."""

    @property
    def pattern_id(self) -> str:
        return "deterministic_chain"

    def execute(
        self,
        *,
        state: RuntimeState,
        invoker: RuntimeToolInvoker,
        planner: ToolPlannerProtocol,
        plan: ToolCallPlan | None,
        allowed_tool_ids: Sequence[str] | None,
        max_iterations: int,
        planner_input: str | list[ChatMessage],
    ) -> ToolInvocationResult:
        _ = planner, plan, allowed_tool_ids, max_iterations, planner_input
        chain = state.context.config.tool_chain_spec
        if chain is None or not chain.steps:
            return ToolInvocationResult(stop_reason="empty_tool_calls")

        user_query = state.request.message or ""
        prior_outputs: list[BaseModel] = []
        traces: list[ToolCallTrace] = []

        for index, step in enumerate(chain.steps):
            registered = invoker.registry.get(step.tool_id)
            step_input = build_chain_step_input(
                step,
                contract=registered.contract,
                user_query=user_query,
                prior_outputs=prior_outputs,
            )
            call = PlannedToolCall(
                step_id=step.step_id or f"chain-{index}",
                tool_id=step.tool_id,
                input=step_input,
            )
            req = ToolExecutionRequest(
                run_id=state.run_id,
                step_id=call.step_id,
                tool_id=call.tool_id,
                input=call.input,
                idempotency_key=f"{state.run_id}:chain:{index}:{call.tool_id}",
            )
            result = invoker.invoke(state=state, request=req, agent_id=state.request.agent_id)
            if result.success:
                output_preview = result.output.model_dump_json()[:400]
                error_msg = None
                prior_outputs.append(result.output)
            else:
                output_preview = None
                error_msg = result.error.error_message
            trace = ToolCallTrace(
                tool_name=call.tool_id,
                arguments=call.input.model_dump(),
                output_preview=output_preview,
                success=result.success,
                error_message=error_msg,
                raw_trace={},
            )
            traces.append(trace)
            emit_high_risk_tool_verify_signal(state=state, invoker=invoker, trace=trace)
            if not result.success:
                break

        if not traces:
            return ToolInvocationResult(stop_reason="empty_tool_calls")

        aggregate = ToolInvocationAggregate.from_traces(traces)
        return ToolInvocationResult(
            tool_traces=traces,
            loop_iterations=1,
            stop_reason="legacy_single_pass" if all(t.success for t in traces) else "empty_tool_calls",
            aggregate=aggregate,
        )
