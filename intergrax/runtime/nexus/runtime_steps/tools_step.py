# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from intergrax.llm.messages import ChatMessage
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.runtime.nexus.policies.runtime_policies import ExecutionKind
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.tools.execution_models import ToolExecutionRequest

if TYPE_CHECKING:
    from intergrax.runtime.nexus.config import ToolsContextScope

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState, ToolCallTrace
from intergrax.runtime.nexus.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.nexus.tracing.tools.tools_summary import ToolsSummaryDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


class ToolsStep(RuntimeStep):
    def execution_kind(self) -> ExecutionKind | None:
        return ExecutionKind.TOOL
    
    def TOOLS_RUNTIME_CONTEXT_PROMPT(self) -> str:
        registry = YamlPromptRegistry.create_default(load=True)
        localized = registry.resolve_localized("tools_runtime_context")
        return localized.system

    async def run(self, state: RuntimeState) -> None:
        """
        Run tools planner + runtime tool execution if configured.

        ToolsAgent is used only for planning.
        All execution is performed by RuntimeToolInvoker (Layer-2 runtime).
        """

        state.used_tools = False
        state.tool_traces = []
        state.tools_agent_answer = None

        invoker = state.context.config.tool_invoker
        tools_agent = state.context.config.tools_agent
        tools_mode = state.context.config.tools_mode

        if invoker is None or tools_agent is None or tools_mode == "off":
            return

        warning: Optional[str] = None
        error_type: Optional[str] = None
        error_message: Optional[str] = None
        
        try:
            
            decision = tools_agent.plan_tools(
                input_data=state.request.message,
                context=None,
                run_id=state.run_id,
            )

            tool_plan = decision.tool_plan
            
            if tool_plan is None or not tool_plan.calls:
                if tools_mode == "required":
                    warning = "tools_mode='required' but no tools were planned."
            else:
                for call in tool_plan.calls:
                    req = ToolExecutionRequest(
                        run_id=state.run_id,
                        step_id=call.step_id,
                        tool_id=call.tool_id,
                        input=call.input,
                    )

                    result = invoker.invoke(state=state, request=req)

                    state.used_tools = True

                    if result.success:
                        output_preview = result.output.model_dump_json()[:400]
                        error_msg = None
                    else:
                        output_preview = None
                        error_msg = result.error.error_message

                    state.tool_traces.append(
                        ToolCallTrace(
                            tool_name=call.tool_id,
                            arguments=call.input.model_dump(),
                            output_preview=output_preview,
                            success=result.success,
                            error_message=error_msg,
                            raw_trace={},
                        )
                    )

            # Inject tool execution results into LLM context
            if state.tool_traces:
                tool_lines: List[str] = []

                for t in state.tool_traces:
                    tool_lines.append(f"Tool '{t.tool_name}' was called.")

                    if t.arguments:
                        try:
                            args_str = json.dumps(t.arguments, ensure_ascii=False)
                        except Exception:
                            args_str = str(t.arguments)
                        tool_lines.append(f"Arguments: {args_str}")

                    if t.output_preview:
                        tool_lines.append("Output:")
                        tool_lines.append(t.output_preview)

                    if t.error_message:
                        tool_lines.append("Error:")
                        tool_lines.append(t.error_message)

                    tool_lines.append("")

                tools_context_for_llm = "\n".join(tool_lines).strip()
                if tools_context_for_llm:
                    insert_at = len(state.messages_for_llm) - 1

                    runtime_prompt = self.TOOLS_RUNTIME_CONTEXT_PROMPT().format(
                        context=tools_context_for_llm
                    )

                    state.messages_for_llm.insert(
                        insert_at,
                        ChatMessage(
                            role="system",
                            content=runtime_prompt,
                        ),
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
