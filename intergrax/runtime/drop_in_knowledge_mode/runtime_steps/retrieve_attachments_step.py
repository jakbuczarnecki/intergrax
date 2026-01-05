# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.tools import format_rag_context, insert_context_before_last_user


class RetrieveAttachmentsStep(RuntimeStep):
    """
    Retrieve relevant chunks from session-ingested attachments (AttachmentIngestionService)
    and inject them into the LLM context.

    Key requirements:
        - Independent from enable_rag (must work even when enable_rag=False).
        - Reuse existing retrieval components (EmbeddingManager + VectorstoreManager.query).
        - Filter by session_id + user_id (+ tenant/workspace if available).
        - Inject as context messages using _insert_context_before_last_user.
    """

    async def run(self, state: RuntimeState) -> None:
        # Defaults
        state.used_attachments_context = False
        state.set_debug_value("attachments_chunks", 0)

        if state.context.ingestion_service is None:
            state.set_debug_section("attachments", {"used": False, "reason": "ingestion_service_not_configured"})
            return

        session = state.session
        if session is None:
            state.set_debug_section("attachments", {"used": False, "reason": "session_not_initialized"})
            return

        # Retrieval (no coupling to enable_rag)
        res = await state.context.ingestion_service.search_session_attachments(
            query=state.request.message,
            session_id=session.id,
            user_id=state.request.user_id,
            tenant_id=session.tenant_id,
            workspace_id=session.workspace_id,
            top_k=6,
            score_threshold=None,
        )

        dbg = (res or {}).get("debug") or {}
        used = bool((res or {}).get("used"))
        chunks = (res or {}).get("hits") or []

        state.used_attachments_context = used and bool(chunks)
        state.set_debug_section("attachments", {
            **dbg,
            "used": bool(used and chunks),
            "hits_count": len(chunks),
        })
        state.set_debug_value("attachments_chunks", len(chunks))

        if not state.used_attachments_context:
            return

        # Build a single system context message (same injection pattern as RAG).
        attachments_context_text = format_rag_context(chunks)
        if not attachments_context_text.strip():
            # If somehow chunks exist but formatting is empty, treat as unused.
            state.used_attachments_context = False
            state.set_debug_section("attachments", {
                **dbg,
                "used": False,
                "reason": "empty_formatted_context",
            })
            state.set_debug_value("attachments_chunks", 0)
            return

        content = "SESSION ATTACHMENTS (retrieved):\n" + attachments_context_text

        context_messages = [
            ChatMessage(
                role="system",
                content=content,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        ]

        insert_context_before_last_user(state, context_messages)

        # Provide compact textual form also for tools agent (same pattern as RAG).
        state.tools_context_parts.append("SESSION ATTACHMENTS:\n" + attachments_context_text)

        # Trace step
        state.trace_event(
            component="engine",
            step="attachments_context",
            message="Session attachments retrieval executed and context injected.",
            data={
                "used_attachments_context": state.used_attachments_context,
                "retrieved_chunks": len(chunks),
            },
        )