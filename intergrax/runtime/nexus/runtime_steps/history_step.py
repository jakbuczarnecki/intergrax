# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.nexus.tracing.history.history_summary import HistorySummaryDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceLevel


class HistoryStep(RuntimeStep):
    """
    Build conversation history for the LLM.

    This step is responsible only for selecting and shaping the
    conversational context (previous user/assistant turns).

    Retrieval (RAG) is handled separately in `RagStep`.
    """

    async def run(self, state: RuntimeState) -> None:
        session = state.session
        assert session is not None, "Session must be set before history step."
        req = state.request
        base_history = state.base_history

        if state.context.context_builder is not None:
            built = await state.context.context_builder.build_context(
                session=session,
                request=req,
                base_history=base_history,
            )

            state.context_builder_result = built

            history_messages = built.history_messages or []
            state.messages_for_llm.extend(history_messages)
            state.built_history_messages = history_messages
            state.history_includes_current_user = True
        else:
            state.messages_for_llm.extend(base_history)
            state.built_history_messages = base_history
            state.history_includes_current_user = True

        # Trace history building step (summary).
        state.trace_event(
            component="engine",
            step="history",
            message="Conversation history built for LLM.",
            level=TraceLevel.INFO,
            payload=HistorySummaryDiagV1(
                base_history_length=len(state.base_history),
                history_length=len(state.built_history_messages),
                history_includes_current_user=state.history_includes_current_user,
            ),
        )
