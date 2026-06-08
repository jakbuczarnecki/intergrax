# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any, Dict

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.nexus.policies.runtime_policies import ExecutionKind
from intergrax.runtime.nexus.context.context_preflight import verify_context_preflight
from intergrax.runtime.nexus.tracing.adapters.core_llm_adapter_failed import CoreLLMAdapterFailedDiagV1
from intergrax.runtime.nexus.tracing.adapters.core_llm_adapter_returned import CoreLLMAdapterReturnedDiagV1
from intergrax.runtime.nexus.tracing.adapters.core_llm_call_recorded import CoreLLMCallRecordedDiagV1
from intergrax.runtime.nexus.tracing.adapters.core_llm_used_tools_agent_answer import CoreLLMUsedToolsAgentAnswerDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


class CoreLLMStep(RuntimeStep):
    """
    Call the core LLM adapter and decide on the final answer text,
    possibly falling back to tool_planner_answer when needed.
    """

    def execution_kind(self) -> ExecutionKind | None:
        return ExecutionKind.LLM

    async def run(self, state: RuntimeState) -> None:
        # If tools were used and we have an explicit agent answer, prefer it.
        if state.used_tools and state.tool_planner_answer:
            state.trace_event(
                component=TraceComponent.ENGINE,
                step="core_llm",
                message="Using tool_planner_answer as the final answer.",
                level=TraceLevel.INFO,
                payload=CoreLLMUsedToolsAgentAnswerDiagV1(
                    used_tools_answer=True,
                    has_tool_planner_answer=True,
                ),
            )
            state.raw_answer = str(state.tool_planner_answer)
            return

        try:
            # Determine the per-request max output tokens, if any.
            max_output_tokens = state.request.max_output_tokens

            generate_kwargs: Dict[str, Any] = {}
            if max_output_tokens is not None:
                generate_kwargs["max_tokens"] = max_output_tokens

            msgs = state.messages_for_llm
            verify_context_preflight(
                msgs,
                state.context.config.llm_adapter,
                max_output_tokens=max_output_tokens,
            )
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


            completion = state.context.config.llm_adapter.generate_messages(
                msgs,
                run_id=state.run_id,
                **generate_kwargs,
            )
            state.last_llm_adapter_response = completion
            answer_text = completion.content

            usage = completion.usage
            input_tokens = int(usage.input_tokens) if usage else 0
            output_tokens = int(usage.output_tokens) if usage else 0
            state.trace_event(
                component=TraceComponent.ENGINE,
                step="core_llm",
                message="Core LLM adapter returned answer.",
                level=TraceLevel.INFO,
                payload=CoreLLMAdapterReturnedDiagV1(
                    used_tools_answer=False,
                    finish_reason=completion.finish_reason.value,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    answer_len=len(answer_text),
                    answer_is_empty=not bool(answer_text),
                ),
            )
            state.trace_event(
                component=TraceComponent.ENGINE,
                step="core_llm",
                message="Core LLM call recorded for replay.",
                level=TraceLevel.INFO,
                payload=CoreLLMCallRecordedDiagV1(
                    model=completion.model or "",
                    provider=completion.provider or "",
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    finish_reason=completion.finish_reason.value,
                    response_id=completion.response_id,
                    has_refusal=bool(completion.refusal),
                    has_tool_calls=completion.has_tool_calls,
                ),
            )

            state.raw_answer = answer_text

        except Exception as e:
            # Trace the error and whether a tool_planner_answer fallback is available.
            state.trace_event(
                component=TraceComponent.ENGINE,
                step="core_llm_error",
                message="Core LLM adapter failed; falling back if possible.",
                level=TraceLevel.ERROR,
                payload=CoreLLMAdapterFailedDiagV1(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    has_tool_planner_answer=bool(state.tool_planner_answer),
                ),
            )

            if state.tool_planner_answer:
                state.raw_answer = (
                    "[ERROR] LLM adapter failed, falling back to tool planner answer.\n"
                    f"Details: {e}\n\n"
                    f"{state.tool_planner_answer}"
                )
                return

            state.raw_answer = f"[ERROR] LLM adapter failed: {e}"