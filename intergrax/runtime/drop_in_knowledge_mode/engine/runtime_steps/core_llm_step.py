# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any, Dict

from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.contract import RuntimeStep


class CoreLLMStep(RuntimeStep):
    """
    Call the core LLM adapter and decide on the final answer text,
    possibly falling back to tools_agent_answer when needed.
    """

    async def run(self, state: RuntimeState) -> None:
        # If tools were used and we have an explicit agent answer, prefer it.
        if state.used_tools and state.tools_agent_answer:
            # Trace the fact that we are reusing the tools agent answer
            # instead of calling the core LLM adapter.
            state.trace_event(
                component="engine",
                step="core_llm",
                message="Using tools_agent_answer as the final answer.",
                data={
                    "used_tools_answer": True,
                    "has_tools_agent_answer": True,
                },
            )
            state.raw_answer =  str(state.tools_agent_answer)
            return

        try:
            # Determine the per-request max output tokens, if any.
            max_output_tokens = state.request.max_output_tokens

            generate_kwargs: Dict[str, Any] = {}
            if max_output_tokens is not None:
                # Pass a max_tokens hint to the adapter. If the adapter ignores
                # it or uses a different keyword, that should be handled inside
                # the adapter implementation.
                generate_kwargs["max_tokens"] = max_output_tokens

            msgs = state.messages_for_llm
            if not msgs or msgs[-1].role != "user":
                raise Exception(
                    f"Last message must be 'user' (got: {msgs[-1].role if msgs else 'None'})."
                )

            raw_answer = state.context.config.llm_adapter.generate_messages(
                state.messages_for_llm,
                run_id=state.run_id,
                **generate_kwargs,
            )

            state.trace_event(
                component="engine",
                step="core_llm",
                message="Core LLM adapter returned answer.",
                data={
                    "used_tools_answer": False,
                    "adapter_return_type": "str",
                    "answer_len": len(raw_answer),
                    "answer_is_empty": not bool(raw_answer),
                },
            )
            
            state.raw_answer = raw_answer

        except Exception as e:
            state.set_debug_value("llm_error", str(e))

            # Trace the error and whether a tools_agent_answer fallback is available.
            state.trace_event(
                component="engine",
                step="core_llm_error",
                message="Core LLM adapter failed; falling back if possible.",
                data={
                    "error": str(e),
                    "has_tools_agent_answer": bool(state.tools_agent_answer),
                },
            )

            if state.tools_agent_answer:
                state.raw_answer = (
                    "[ERROR] LLM adapter failed, falling back to tools agent answer.\n"
                    f"Details: {e}\n\n"
                    f"{state.tools_agent_answer}"
                )
                return

            state.raw_answer = f"[ERROR] LLM adapter failed: {e}"