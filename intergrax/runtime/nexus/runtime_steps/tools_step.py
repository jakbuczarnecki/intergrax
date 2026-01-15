# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.policies.runtime_policies import ExecutionKind
if TYPE_CHECKING:
    from intergrax.runtime.nexus.config import ToolsContextScope
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState, ToolCallTrace
from intergrax.runtime.nexus.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.nexus.tracing.tools.tools_summary import ToolsSummaryDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


class ToolsStep(RuntimeStep):

    def execution_kind(self) -> ExecutionKind | None:
        return ExecutionKind.TOOL

    async def run(self, state: RuntimeState) -> None:
        """
        Run tools agent (planning + tool calls) if configured.

        The tools result is:
          - optionally used as the final answer (when tools_mode != "off"),
          - appended as system context for the core LLM.
        """
        state.used_tools = False
        state.tool_traces = []
        state.tools_agent_answer = None

        use_tools = (
            state.context.config.tools_agent is not None
            and state.context.config.tools_mode != "off"
        )
        if not use_tools:
            return

        tools_mode = state.context.config.tools_mode

        tools_context = (
            "\n\n".join(state.tools_context_parts).strip()
            if state.tools_context_parts
            else None
        )

        warning: Optional[str] = None
        error_type: Optional[str] = None
        error_message: Optional[str] = None

        def _make_output_preview(raw: Any, *, limit: int = 400) -> Optional[str]:
            if raw is None:
                return None
            if isinstance(raw, str):
                s = raw
            else:
                try:
                    s = json.dumps(raw, ensure_ascii=False)
                except Exception:
                    s = str(raw)
            s = s.strip()
            if not s:
                return None
            return s[:limit]

        def _normalize_tool_traces(raw_traces: Any) -> List[ToolCallTrace]:
            if not isinstance(raw_traces, list):
                return []

            out: List[ToolCallTrace] = []
            for item in raw_traces:
                if not isinstance(item, dict):
                    continue

                tool_name = item.get("tool") or ""
                if not isinstance(tool_name, str):
                    tool_name = str(tool_name)

                args = item.get("args")
                arguments: Dict[str, Any] = args if isinstance(args, dict) else {}

                err = item.get("error")
                error_msg: Optional[str]
                if err is None:
                    error_msg = None
                elif isinstance(err, str):
                    error_msg = err
                else:
                    error_msg = str(err)

                success = not bool(error_msg)

                # Prefer explicit output_preview if tools agent provides it; otherwise derive from output.
                op = item.get("output_preview")
                if isinstance(op, str) and op.strip():
                    output_preview = op.strip()[:400]
                else:
                    output_preview = _make_output_preview(item.get("output"))

                out.append(
                    ToolCallTrace(
                        tool_name=tool_name,
                        arguments=arguments,
                        output_preview=output_preview,
                        success=success,
                        error_message=error_msg,
                        raw_trace=item,
                    )
                )
            return out

        try:
            # Decide what to pass as input_data for the tools agent.
            if state.context.config.tools_context_scope == ToolsContextScope.CURRENT_MESSAGE_ONLY:
                agent_input = state.request.message

            elif state.context.config.tools_context_scope == ToolsContextScope.CONVERSATION:
                if state.built_history_messages:
                    agent_input = state.built_history_messages
                else:
                    agent_input = state.base_history

            else:
                agent_input = state.messages_for_llm

            tools_result = state.context.config.tools_agent.run(
                input_data=agent_input,
                context=tools_context,
                stream=False,
                tool_choice=None,
                output_model=None,
                run_id=state.run_id,
                llm_usage_tracker=state.llm_usage_tracker,
            )

            if not isinstance(tools_result, dict):
                tools_result = {}

            state.tools_agent_answer = tools_result.get("answer", "") or None

            raw_traces = tools_result.get("tool_traces")
            state.tool_traces = _normalize_tool_traces(raw_traces)
            state.used_tools = bool(state.tool_traces)

            if tools_mode == "required" and not state.used_tools:
                warning = "tools_mode='required' but no tools were invoked by the tools_agent."

            # Inject executed tool calls as system context for core LLM.
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

                    raw_output = t.raw_trace.get("output") if isinstance(t.raw_trace, dict) else None
                    if raw_output is not None:
                        if isinstance(raw_output, (dict, list)):
                            try:
                                out_str = json.dumps(raw_output, ensure_ascii=False)
                            except Exception:
                                out_str = str(raw_output)
                        else:
                            out_str = str(raw_output)
                        tool_lines.append("Output:")
                        tool_lines.append(out_str)

                    if t.error_message:
                        tool_lines.append("Error:")
                        tool_lines.append(t.error_message)

                    tool_lines.append("")

                tools_context_for_llm = "\n".join(tool_lines).strip()
                if tools_context_for_llm:
                    insert_at = len(state.messages_for_llm) - 1
                    state.messages_for_llm.insert(
                        insert_at,
                        ChatMessage(
                            role="system",
                            content=(
                                "The following tool calls have been executed. "
                                "Use their results when answering the user.\n\n"
                                + tools_context_for_llm
                            ),
                        ),
                    )

        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)

        tool_names = sorted({t.tool_name for t in state.tool_traces if t.tool_name})

        state.trace_event(
            component=TraceComponent.ENGINE,
            step="tools",
            message="Tools agent step executed.",
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
