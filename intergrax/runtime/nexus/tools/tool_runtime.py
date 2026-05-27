# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


@dataclass(frozen=True, slots=True)
class ToolInvocationPlan:
    """Runtime-neutral plan for capability step invocation."""

    use_rag: bool = False
    use_websearch: bool = False
    use_tools: bool = False


@dataclass(frozen=True, slots=True)
class ToolRuntimeResult:
    used_rag: bool
    used_websearch: bool
    used_tools: bool
    tool_trace_count: int


@runtime_checkable
class ToolPlanLike(Protocol):
    use_rag: bool
    use_websearch: bool
    use_tools: bool


class ToolRuntime:
    """
    Tier-1 runtime primitive for invoking Nexus capability steps.

    Agents declare tool needs via contract; runtime executes RAG / websearch / tools steps.
    """

    @staticmethod
    def plan_from_like(source: ToolPlanLike) -> ToolInvocationPlan:
        return ToolInvocationPlan(
            use_rag=bool(source.use_rag),
            use_websearch=bool(source.use_websearch),
            use_tools=bool(source.use_tools),
        )

    @staticmethod
    async def invoke(
        *,
        state: "RuntimeState",
        plan: ToolInvocationPlan,
        trace_step: str = "ToolRuntime",
        allowed_tools: Optional[Sequence[str]] = None,
    ) -> ToolRuntimeResult:
        from intergrax.runtime.nexus.tools.tool_access_policy import ToolAccessPolicy
        from intergrax.runtime.nexus.runtime_steps.rag_step import RagStep
        from intergrax.runtime.nexus.runtime_steps.tools_step import ToolsStep
        from intergrax.runtime.nexus.runtime_steps.websearch_step import WebsearchStep
        from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel

        plan = ToolAccessPolicy.apply(plan, allowed_tools=allowed_tools, state=state)

        cfg = state.context.config

        if plan.use_rag:
            if cfg.enable_rag:
                await RagStep().run(state)
            else:
                state.trace_event(
                    component=TraceComponent.PIPELINE,
                    step=trace_step,
                    message="Plan requested RAG but enable_rag is false; skipping RagStep.",
                    level=TraceLevel.WARNING,
                )

        if plan.use_websearch:
            if cfg.enable_websearch:
                await WebsearchStep().run(state)
            else:
                state.trace_event(
                    component=TraceComponent.PIPELINE,
                    step=trace_step,
                    message="Plan requested websearch but enable_websearch is false; skipping.",
                    level=TraceLevel.WARNING,
                )

        if plan.use_tools:
            if cfg.tools_agent and cfg.tool_invoker and cfg.tools_mode != "off":
                await ToolsStep().run(state)
            else:
                state.trace_event(
                    component=TraceComponent.PIPELINE,
                    step=trace_step,
                    message="Plan requested tools but tools are off or not configured; skipping.",
                    level=TraceLevel.WARNING,
                )

        return ToolRuntimeResult(
            used_rag=state.used_rag,
            used_websearch=state.used_websearch,
            used_tools=state.used_tools,
            tool_trace_count=len(state.tool_traces or []),
        )
