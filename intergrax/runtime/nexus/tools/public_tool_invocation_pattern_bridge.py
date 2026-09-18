# © Artur Czarnecki. All rights reserved.

"""Adapts public Tools-domain invocation patterns for Nexus execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_invocation_pattern import (
    NexusToolInvocationPattern,
    ToolInvocationResult,
)
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.tools.core.tool_plan import ToolCallPlan
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.invocation_pattern.contracts import (
    ToolInvocationInvokerPort,
    ToolInvocationPattern,
    ToolInvocationPatternContext,
    ToolInvocationPatternResult,
    ToolInvocationPlannerPort,
)


class _RuntimeToolInvocationInvokerPort:
    def __init__(
        self,
        *,
        state: RuntimeState,
        invoker: RuntimeToolInvoker,
    ) -> None:
        self._state = state
        self._invoker = invoker

    def invoke_tool(
        self,
        *,
        agent_id: str,
        request: ToolExecutionRequest,
    ) -> ToolExecutionResult:
        return self._invoker.invoke(
            state=self._state,
            agent_id=agent_id,
            request=request,
        )


class _RuntimeToolInvocationPlannerPort:
    def __init__(self, planner: ToolPlannerProtocol) -> None:
        self._planner = planner

    def plan_tools(
        self,
        input_data: str | list[ChatMessage],
        context: object | None = None,
        *,
        run_id: str,
        allowed_tool_ids: Sequence[str] | None = None,
        tool_choice: object | None = None,
    ):
        return self._planner.plan_tools(
            input_data,
            context,
            run_id=run_id,
            allowed_tool_ids=allowed_tool_ids,
            tool_choice=tool_choice,
        )


def _context_from_state(state: RuntimeState) -> ToolInvocationPatternContext:
    config = state.context.config
    return ToolInvocationPatternContext(
        run_id=state.run_id,
        agent_id=state.request.agent_id,
        user_message=state.request.message,
        tools_mode=str(config.tools_mode),
        max_parallel_tool_calls=int(config.max_parallel_tool_calls),
    )


def _to_nexus_result(result: ToolInvocationPatternResult) -> ToolInvocationResult:
    return ToolInvocationResult(
        loop_iterations=result.loop_iterations,
        stop_reason=result.stop_reason,
        pattern_id=result.pattern_id,
        appended_messages=list(result.appended_messages),
        used_native_tool_messages=result.used_native_tool_messages,
        used_ce_tool_feedback=result.used_ce_tool_feedback,
    )


class PublicToolInvocationPatternBridge:
    """Wraps a public pattern so Nexus tool loops can execute it."""

    def __init__(self, pattern: ToolInvocationPattern) -> None:
        self._pattern = pattern

    @property
    def pattern_id(self) -> str:
        return self._pattern.pattern_id

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
        public_result = self._pattern.execute(
            context=_context_from_state(state),
            invoker=_RuntimeToolInvocationInvokerPort(state=state, invoker=invoker),
            planner=_RuntimeToolInvocationPlannerPort(planner),
            plan=plan,
            allowed_tool_ids=allowed_tool_ids,
            max_iterations=max_iterations,
            planner_input=planner_input,
        )
        nexus_result = _to_nexus_result(public_result)
        if not nexus_result.pattern_id:
            return replace(nexus_result, pattern_id=self.pattern_id)
        return nexus_result


def bridge_public_tool_invocation_pattern(
    pattern: ToolInvocationPattern,
) -> NexusToolInvocationPattern:
    return PublicToolInvocationPatternBridge(pattern)
