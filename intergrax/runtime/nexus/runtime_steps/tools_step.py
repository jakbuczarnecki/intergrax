# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from intergrax.prompts.registry.prompt_registry_resolver import resolve_yaml_prompt_registry
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.runtime.nexus.policies.runtime_policies import ExecutionKind
from intergrax.runtime.nexus.tools.catalog_dispatch import resolve_tool_registry
from intergrax.runtime.nexus.tools.tool_planner_input import resolve_tool_planner_input
from intergrax.runtime.nexus.tools.tool_selection import (
    ToolSelectionContext,
    resolve_planner_allowed_tool_ids,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.nexus.runtime_steps.tool_loop_step import (
    inject_tool_traces_system_context,
    run_bounded_tool_loop,
)
from intergrax.runtime.nexus.tracing.tools.tools_summary import ToolsSummaryDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


class ToolsStep(RuntimeStep):
    def execution_kind(self) -> ExecutionKind | None:
        return ExecutionKind.TOOL

    @staticmethod
    def _resolve_prompt_registry(state: RuntimeState) -> YamlPromptRegistry:
        return resolve_yaml_prompt_registry(
            registry=state.context.prompt_registry,
            catalog_path=state.context.config.prompt_catalog_path,
        )

    def tools_runtime_context_prompt(self, state: RuntimeState) -> str:
        registry = self._resolve_prompt_registry(state)
        localized = registry.resolve_localized("tools_runtime_context")
        return localized.system

    async def run(self, state: RuntimeState) -> None:
        """
        Run tools planner + runtime tool execution if configured.

        Tool planner (``ToolPlannerProtocol``) plans calls only.
        All execution is performed by RuntimeToolInvoker (Layer-2 runtime).
        """

        state.used_tools = False
        state.tool_traces = []
        state.tool_planner_answer = None

        invoker = state.context.config.tool_invoker
        tool_planner = state.context.config.tool_planner
        tools_mode = state.context.config.tools_mode

        if invoker is None or tool_planner is None or tools_mode == "off":
            return

        warning: Optional[str] = None
        error_type: Optional[str] = None
        error_message: Optional[str] = None

        try:
            planner_input = resolve_tool_planner_input(state)
            registry = resolve_tool_registry(invoker)
            if registry is not None:
                allowed_tool_ids = resolve_planner_allowed_tool_ids(
                    state.context.config.tool_selection_mode,
                    ToolSelectionContext(
                        registry=registry,
                        query=state.request.message or "",
                        skill_profile=state.context.config.skill_profile,
                        plan_allowed_tool_ids=state.tool_planner_allowed_tool_ids,
                        top_k=state.context.config.tool_selection_top_k,
                    ),
                )
            else:
                allowed_tool_ids = state.tool_planner_allowed_tool_ids

            loop_result = run_bounded_tool_loop(
                state=state,
                invoker=invoker,
                tool_planner=tool_planner,
                planner_input=planner_input,
                allowed_tool_ids=allowed_tool_ids,
                max_iterations=state.context.config.max_tool_iterations,
            )

            if not loop_result.tool_traces:
                if tools_mode == "required":
                    warning = "tools_mode='required' but no tools were planned."
            else:
                state.used_tools = True
                state.tool_traces = list(loop_result.tool_traces)

            if loop_result.used_native_tool_messages and loop_result.appended_messages:
                state.messages_for_llm.extend(loop_result.appended_messages)
            elif state.tool_traces:
                inject_tool_traces_system_context(
                    state,
                    state.tool_traces,
                    runtime_context_prompt=self.tools_runtime_context_prompt(state),
                )

        except Exception as e:
            print(e)
            error_type = type(e).__name__
            error_message = str(e)

        tool_names = sorted({t.tool_name for t in state.tool_traces if t.tool_name})

        state.trace_event(
            component=TraceComponent.ENGINE,
            step="tools",
            message="Tools planner + runtime execution step executed.",
            level=TraceLevel.ERROR if error_type else TraceLevel.INFO,
            payload=ToolsSummaryDiagV1(
                tools_mode=tools_mode,
                used_tools=state.used_tools,
                tool_calls_count=len(state.tool_traces),
                tool_names=tool_names,
                warning=warning,
                error_type=error_type,
                error_message=error_message,
            ),
        )
