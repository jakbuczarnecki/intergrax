# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import (
    RuntimeAnswer,
    RouteInfo,
    RuntimeStats,
    ToolCallInfo,
)


class PersistAndBuildAnswerStep(RuntimeStep):
    """
    Persist assistant message into the session and build RuntimeAnswer
    including RouteInfo and RuntimeStats.
    """

    async def run(self, state: RuntimeState) -> None:
        answer_text = state.raw_answer

        # Fallback if answer is empty for any reason
        if not isinstance(answer_text, str) or not answer_text.strip():
            answer_text = (
                str(state.tools_agent_answer)
                if state.tools_agent_answer
                else "[ERROR] Empty answer from runtime."
            )

        session = state.session
        assert session is not None, "Session must be set before persistence."

        assistant_message = ChatMessage(
            role="assistant",
            content=answer_text,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        await state.context.session_manager.append_message(session.id, assistant_message)

        # Strategy label
        if state.used_rag and state.used_websearch and state.used_tools:
            strategy = "llm_with_rag_websearch_and_tools"
        elif state.used_rag and state.used_tools:
            strategy = "llm_with_rag_and_tools"
        elif state.used_websearch and state.used_tools:
            strategy = "llm_with_websearch_and_tools"
        elif state.used_tools:
            strategy = "llm_with_tools"
        elif state.used_rag and state.used_websearch:
            strategy = "llm_with_rag_and_websearch"
        elif state.used_rag:
            strategy = "llm_with_rag_context_builder"
        elif state.used_websearch:
            strategy = "llm_with_websearch"
        elif state.used_attachments_context:
            strategy = "llm_with_session_attachments"
        elif state.ingestion_results:
            strategy = "llm_only_with_ingestion"
        else:
            strategy = "llm_only"

        route_info = RouteInfo(
            used_rag=state.used_rag and state.context.config.enable_rag,
            used_websearch=state.used_websearch and state.context.config.enable_websearch,
            used_tools=state.used_tools and state.context.config.tools_mode != "off",
            used_user_profile=state.used_user_profile,
            used_user_longterm_memory=state.used_user_longterm_memory and state.context.config.enable_user_longterm_memory,
            strategy=strategy,
            extra={
                "used_attachments_context": bool(state.used_attachments_context),
                "attachments_chunks": int(state.debug_trace.get("attachments_chunks", 0) or 0),
            },
        )

        # Token stats are still placeholders – can be wired from LLM adapter later.
        stats = RuntimeStats(
            total_tokens=None,
            input_tokens=None,
            output_tokens=None,
            rag_tokens=None,
            websearch_tokens=None,
            tool_tokens=None,
            duration_ms=None,
            extra={},
        )

        tool_calls_for_answer: List[ToolCallInfo] = []
        for t in state.tool_traces:
            tool_calls_for_answer.append(
                ToolCallInfo(
                    tool_name=t.get("tool") or "",
                    arguments=t.get("args") or {},
                    result_summary=(
                        t.get("output_preview")
                        if isinstance(t.get("output_preview"), str)
                        else None
                    ),
                    success=not bool(t.get("error")),
                    error_message=t.get("error"),
                    extra={"raw_trace": t},
                )
            )

        # Trace persistence and answer building step.
        state.trace_event(
            component="engine",
            step="persist_and_build_answer",
            message="Assistant answer persisted and RuntimeAnswer built.",
            data={
                "session_id": session.id,
                "strategy": strategy,
                "used_rag": state.used_rag,
                "used_websearch": state.used_websearch,
                "used_tools": state.used_tools,
            },
        )

        state.runtime_answer = RuntimeAnswer(
            answer=answer_text,
            citations=[],
            route=route_info,
            tool_calls=tool_calls_for_answer,
            stats=stats,
            raw_model_output=None,
        )