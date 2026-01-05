# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from typing import Any, Dict, List

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.drop_in_knowledge_mode.config import ToolsContextScope
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.planning.runtime_step_handlers import RuntimeStep


class ToolsStep(RuntimeStep):
    """
    Run tools agent (planning + tool calls) if configured.

    Output is stored in RuntimeState:
      - state.used_tools
      - state.tool_traces
      - state.tools_agent_answer

    Core LLM step may decide to reuse tools_agent_answer.
    """

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

        tools_context = (
            "\n\n".join(state.tools_context_parts).strip()
            if state.tools_context_parts
            else None
        )

        debug_tools: Dict[str, Any] = {
            "mode": state.context.config.tools_mode,
        }

        try:
            # Decide what to pass as input_data for the tools agent.
            if state.context.config.tools_context_scope == ToolsContextScope.CURRENT_MESSAGE_ONLY:
                agent_input = state.request.message

            elif state.context.config.tools_context_scope == ToolsContextScope.CONVERSATION:
                # Use history built by ContextBuilder if available,
                # otherwise fall back to base_history.
                if state.built_history_messages:
                    agent_input = state.built_history_messages
                else:
                    agent_input = state.base_history

            else:
                # FULL_CONTEXT or any future scope:
                # pass entire message list built so far.
                agent_input = state.messages_for_llm

            tools_result = state.context.config.tools_agent.run(
                input_data=agent_input,
                context=tools_context,
                stream=False,
                tool_choice=None,
                output_model=None,
                run_id=state.run_id,
                llm_usage_tracker = state.llm_usage_tracker
            )

            state.tools_agent_answer = tools_result.get("answer", "") or None
            state.tool_traces = tools_result.get("tool_traces") or []
            state.used_tools = bool(state.tool_traces)

            debug_tools["used_tools"] = state.used_tools
            debug_tools["tool_traces"] = state.tool_traces
            if state.tools_agent_answer:
                debug_tools["agent_answer_preview"] = str(state.tools_agent_answer)[:200]
            if state.context.config.tools_mode == "required" and not state.used_tools:
                debug_tools["warning"] = (
                    "tools_mode='required' but no tools were invoked by the tools_agent."
                )

            # Inject executed tool calls as system context for core LLM.
            if state.tool_traces:
                tool_lines: List[str] = []
                for t in state.tool_traces:
                    name = t.get("tool")
                    args = t.get("args")
                    output = t.get("output")

                    tool_lines.append(f"Tool '{name}' was called.")
                    if args is not None:
                        try:
                            args_str = json.dumps(args, ensure_ascii=False)
                        except Exception:
                            args_str = str(args)
                        tool_lines.append(f"Arguments: {args_str}")

                    if output is not None:
                        if isinstance(output, (dict, list)):
                            try:
                                out_str = json.dumps(output, ensure_ascii=False)
                            except Exception:
                                out_str = str(output)
                        else:
                            out_str = str(output)
                        tool_lines.append("Output:")
                        tool_lines.append(out_str)

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
            debug_tools["tools_error"] = str(e)

        state.set_debug_section("tools",  debug_tools)

        # Trace tools step.
        state.trace_event(
            component="engine",
            step="tools",
            message="Tools agent step executed.",
            data={
                "tools_mode": state.context.config.tools_mode,
                "used_tools": state.used_tools,
                "tool_traces_count": len(state.tool_traces),
            },
        )
