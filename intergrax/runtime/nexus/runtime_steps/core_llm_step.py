# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any, Dict

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.nexus.tracing.adapters.core_llm_adapter_failed import CoreLLMAdapterFailedDiagV1
from intergrax.runtime.nexus.tracing.adapters.core_llm_adapter_returned import CoreLLMAdapterReturnedDiagV1
from intergrax.runtime.nexus.tracing.adapters.core_llm_used_tools_agent_answer import CoreLLMUsedToolsAgentAnswerDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


class CoreLLMStep(RuntimeStep):
    """
    Call the core LLM adapter and decide on the final answer text,
    possibly falling back to tools_agent_answer when needed.
    """

    async def run(self, state: RuntimeState) -> None:
        # If tools were used and we have an explicit agent answer, prefer it.
        if state.used_tools and state.tools_agent_answer:
            state.trace_event(
                component=TraceComponent.ENGINE,
                step="core_llm",
                message="Using tools_agent_answer as the final answer.",
                level=TraceLevel.INFO,
                payload=CoreLLMUsedToolsAgentAnswerDiagV1(
                    used_tools_answer=True,
                    has_tools_agent_answer=True,
                ),
            )
            state.raw_answer = str(state.tools_agent_answer)
            return

        try:
            # Determine the per-request max output tokens, if any.
            max_output_tokens = state.request.max_output_tokens

            generate_kwargs: Dict[str, Any] = {}
            if max_output_tokens is not None:
                generate_kwargs["max_tokens"] = max_output_tokens

            msgs = state.messages_for_llm
            if not msgs or msgs[-1].role != "user":
                last_role = msgs[-1].role if msgs else None
                roles_tail = [m.role for m in msgs[-8:]] if msgs else []

                # Production-grade: emit trace event BEFORE raising, so incidents are diagnosable even if exception is swallowed upstream.
                state.trace_event(
                    level=TraceLevel.ERROR,
                    component=TraceComponent.RUNTIME,
                    step="CoreLLMStep",
                    message=(
                        "Runtime invariant violated: messages_for_llm must end with a 'user' message. "
                        f"got_last_role={last_role!r}, messages_count={len(msgs) if msgs else 0}, roles_tail={roles_tail!r}"
                    ),
                    payload=None,
                )

                raise ValueError(
                    "Runtime invariant violated: messages_for_llm must end with a 'user' message. "
                    f"got_last_role={last_role!r}, messages_count={len(msgs) if msgs else 0}, roles_tail={roles_tail!r}"
                )


            raw_answer = state.context.config.llm_adapter.generate_messages(
                msgs,
                run_id=state.run_id,
                **generate_kwargs,
            )

            state.trace_event(
                component=TraceComponent.ENGINE,
                step="core_llm",
                message="Core LLM adapter returned answer.",
                level=TraceLevel.INFO,
                payload=CoreLLMAdapterReturnedDiagV1(
                    used_tools_answer=False,
                    adapter_return_type="str",
                    answer_len=len(raw_answer),
                    answer_is_empty=not bool(raw_answer),
                ),
            )

            state.raw_answer = raw_answer

        except Exception as e:
            # Trace the error and whether a tools_agent_answer fallback is available.
            state.trace_event(
                component=TraceComponent.ENGINE,
                step="core_llm_error",
                message="Core LLM adapter failed; falling back if possible.",
                level=TraceLevel.ERROR,
                payload=CoreLLMAdapterFailedDiagV1(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    has_tools_agent_answer=bool(state.tools_agent_answer),
                ),
            )

            if state.tools_agent_answer:
                state.raw_answer = (
                    "[ERROR] LLM adapter failed, falling back to tools agent answer.\n"
                    f"Details: {e}\n\n"
                    f"{state.tools_agent_answer}"
                )
                return

            state.raw_answer = f"[ERROR] LLM adapter failed: {e}"