# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.nexus.runtime_steps.tools import format_rag_context, insert_context_before_last_user
from intergrax.runtime.nexus.tracing.attachments.attachments_context_summary import AttachmentsContextSummaryDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceLevel


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
        state.attachments_chunks_count = 0

        TOP_K = 6

        configured = state.context.ingestion_service is not None
        session = state.session
        has_session = session is not None

        if not configured:
            state.trace_event(
                component="engine",
                step="attachments_context",
                message="Session attachments retrieval skipped (ingestion_service not configured).",
                level=TraceLevel.INFO,
                payload=AttachmentsContextSummaryDiagV1(
                    configured=False,
                    has_session=has_session,
                    used_attachments_context=False,
                    hits_count=0,
                    top_k=TOP_K,
                    reason="ingestion_service_not_configured",
                    error_type=None,
                    error_message=None,
                ),
            )
            return

        if session is None:
            state.trace_event(
                component="engine",
                step="attachments_context",
                message="Session attachments retrieval skipped (session not initialized).",
                level=TraceLevel.INFO,
                payload=AttachmentsContextSummaryDiagV1(
                    configured=True,
                    has_session=False,
                    used_attachments_context=False,
                    hits_count=0,
                    top_k=TOP_K,
                    reason="session_not_initialized",
                    error_type=None,
                    error_message=None,
                ),
            )
            return

        try:
            # Retrieval (no coupling to enable_rag)
            res = await state.context.ingestion_service.search_session_attachments(
                query=state.request.message,
                session_id=session.id,
                user_id=state.request.user_id,
                tenant_id=session.tenant_id,
                workspace_id=session.workspace_id,
                top_k=TOP_K,
                score_threshold=None,
            )
        except Exception as e:
            state.trace_event(
                component="engine",
                step="attachments_context",
                message="Session attachments retrieval failed.",
                level=TraceLevel.ERROR,
                payload=AttachmentsContextSummaryDiagV1(
                    configured=True,
                    has_session=True,
                    used_attachments_context=False,
                    hits_count=0,
                    top_k=TOP_K,
                    reason="search_failed",
                    error_type=type(e).__name__,
                    error_message=str(e),
                ),
            )
            raise

        used_flag = bool((res or {}).get("used"))
        chunks = (res or {}).get("hits") or []

        state.used_attachments_context = bool(used_flag and chunks)
        state.attachments_chunks_count = int(len(chunks))

        if not state.used_attachments_context:
            state.trace_event(
                component="engine",
                step="attachments_context",
                message="Session attachments retrieval executed (no context used).",
                level=TraceLevel.INFO,
                payload=AttachmentsContextSummaryDiagV1(
                    configured=True,
                    has_session=True,
                    used_attachments_context=False,
                    hits_count=len(chunks),
                    top_k=TOP_K,
                    reason="no_hits_or_not_used",
                    error_type=None,
                    error_message=None,
                ),
            )
            return

        # Build a single system context message (same injection pattern as RAG).
        attachments_context_text = format_rag_context(chunks)
        if not attachments_context_text.strip():
            # If somehow chunks exist but formatting is empty, treat as unused.
            state.used_attachments_context = False
            state.attachments_chunks_count = 0

            state.trace_event(
                component="engine",
                step="attachments_context",
                message="Session attachments retrieved but formatted context is empty; treating as unused.",
                level=TraceLevel.INFO,
                payload=AttachmentsContextSummaryDiagV1(
                    configured=True,
                    has_session=True,
                    used_attachments_context=False,
                    hits_count=len(chunks),
                    top_k=TOP_K,
                    reason="empty_formatted_context",
                    error_type=None,
                    error_message=None,
                ),
            )
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

        # Trace step summary
        state.trace_event(
            component="engine",
            step="attachments_context",
            message="Session attachments retrieval executed and context injected.",
            level=TraceLevel.INFO,
            payload=AttachmentsContextSummaryDiagV1(
                configured=True,
                has_session=True,
                used_attachments_context=True,
                hits_count=len(chunks),
                top_k=TOP_K,
                reason=None,
                error_type=None,
                error_message=None,
            ),
        )