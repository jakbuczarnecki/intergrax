# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations


from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.tools import format_rag_context, insert_context_before_last_user
from intergrax.runtime.drop_in_knowledge_mode.tracing.rag.rag_summary import RagSummaryDiagV1
from intergrax.runtime.drop_in_knowledge_mode.tracing.trace_models import TraceLevel


class RagStep(RuntimeStep):
    """
    Build RAG layer (if configured) on top of the already constructed conversation history.

    Responsibilities:
      - ensure ContextBuilder result exists (fallback build_context if missing),
      - use RagPromptBuilder to build context messages from retrieved chunks,
      - inject RAG context messages before the last user message,
      - prepare compact RAG text for tools agent (state.tools_context_parts),
      - set debug fields + trace event.
    """

    async def run(self, state: RuntimeState) -> None:
        # Defaults
        state.used_rag = False

        ctx = state.context

        # RAG disabled => no-op (still trace summary for baseline observability)
        if not ctx.config.enable_rag:
            state.trace_event(
                component="engine",
                step="rag",
                message="RAG disabled; skipping.",
                level=TraceLevel.INFO,
                payload=RagSummaryDiagV1(
                    rag_enabled=False,
                    used_rag=False,
                    chunks_count=0,
                    context_messages_count=0,
                    warning=None,
                ),
            )
            return

        if ctx.context_builder is None:
            raise RuntimeError("RAG enabled but ContextBuilder is not configured.")

        # Prefer result from HistoryStep (normal flow)
        built = state.context_builder_result

        # Fallback: build here (should be rare)
        if built is None:
            session = state.session
            assert session is not None, "Session must be set before RAG step."

            built = await ctx.context_builder.build_context(
                session=session,
                request=state.request,
                base_history=state.base_history,
            )
            state.context_builder_result = built

        rag_info = built.rag_debug_info or {}

        retrieved_chunks = built.retrieved_chunks or []
        state.used_rag = bool(rag_info.get("used", bool(retrieved_chunks)))

        if not state.used_rag:
            state.trace_event(
                component="engine",
                step="rag",
                message="RAG enabled but not used (no retrieved context).",
                level=TraceLevel.INFO,
                payload=RagSummaryDiagV1(
                    rag_enabled=True,
                    used_rag=False,
                    chunks_count=0,
                    context_messages_count=0,
                    warning=None,
                ),
            )
            return

        if ctx.rag_prompt_builder is None:
            raise RuntimeError("RAG enabled but rag_prompt_builder is not configured.")

        # Build RAG prompt bundle (context messages only)
        bundle = ctx.rag_prompt_builder.build_rag_prompt(built)
        context_messages = bundle.context_messages or []

        # Inject context before last user message
        if context_messages:
            insert_context_before_last_user(state, context_messages)

        # Compact textual form of RAG context for tools agent
        rag_context_text = format_rag_context(retrieved_chunks)
        if rag_context_text:
            state.tools_context_parts.append("RAG CONTEXT:\n" + rag_context_text)

        # Trace summary
        state.trace_event(
            component="engine",
            step="rag",
            message="RAG context built and injected.",
            level=TraceLevel.INFO,
            payload=RagSummaryDiagV1(
                rag_enabled=True,
                used_rag=True,
                chunks_count=len(retrieved_chunks),
                context_messages_count=len(context_messages),
                warning=None,
            ),
        )
