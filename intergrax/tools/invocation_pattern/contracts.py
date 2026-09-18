# © Artur Czarnecki. All rights reserved.

"""Public ToolInvocationPattern extension contract (Tools domain · PLUG-02)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from intergrax.llm.messages import ChatMessage
from intergrax.tools.core.tool_plan import ToolCallPlan
from intergrax.tools.core.tool_plan_decision import ToolPlanDecision
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult

ToolInvocationStopReason = Literal[
    "empty_tool_calls",
    "max_iterations",
    "planner_final_answer",
    "legacy_single_pass",
]


@dataclass(frozen=True, slots=True)
class ToolInvocationPatternContext:
    """Neutral execution view for custom invocation patterns."""

    run_id: str
    agent_id: str | None = None
    user_message: str | None = None
    tools_mode: str = "auto"
    max_parallel_tool_calls: int = 1


@runtime_checkable
class ToolInvocationPlannerPort(Protocol):
    """Plans tool calls without Nexus planner types."""

    def plan_tools(
        self,
        input_data: str | list[ChatMessage],
        context: object | None = None,
        *,
        run_id: str,
        allowed_tool_ids: Sequence[str] | None = None,
        tool_choice: object | None = None,
    ) -> ToolPlanDecision:
        ...


@runtime_checkable
class ToolInvocationInvokerPort(Protocol):
    """Invokes a single prepared tool execution request."""

    def invoke_tool(
        self,
        *,
        agent_id: str,
        request: ToolExecutionRequest,
    ) -> ToolExecutionResult:
        ...


@dataclass(slots=True)
class ToolInvocationPatternResult:
    """Orchestration outcome visible to public pattern plugins."""

    loop_iterations: int = 0
    stop_reason: ToolInvocationStopReason = "legacy_single_pass"
    pattern_id: str = ""
    appended_messages: list[ChatMessage] = field(default_factory=list)
    used_native_tool_messages: bool = False
    used_ce_tool_feedback: bool = False


@runtime_checkable
class ToolInvocationPattern(Protocol):
    """Public entry-point contract for ``intergrax.tool_invocation_patterns``."""

    @property
    def pattern_id(self) -> str:
        ...

    def execute(
        self,
        *,
        context: ToolInvocationPatternContext,
        invoker: ToolInvocationInvokerPort,
        planner: ToolInvocationPlannerPort,
        plan: ToolCallPlan | None,
        allowed_tool_ids: Sequence[str] | None,
        max_iterations: int,
        planner_input: str | list[ChatMessage],
    ) -> ToolInvocationPatternResult:
        ...
