# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.ingestion.ingestion_service import IngestionResult
from intergrax.runtime.drop_in_knowledge_mode.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import RuntimeRequest


class SessionAndIngestStep(RuntimeStep):
    """
    Load or create a session, ingest attachments (RAG), append the user
    message and initialize debug_trace.

    IMPORTANT:
        - This step does NOT load conversation history.
        - History is loaded and preprocessed in `_step_build_base_history`.
    """

    async def run(self, state: RuntimeState) -> None:
        req = state.request

        # 1. Load or create session
        session = await state.context.session_manager.get_session(req.session_id)
        if session is None:
            session = await state.context.session_manager.create_session(
                session_id=req.session_id,
                user_id=req.user_id,
                tenant_id=req.tenant_id or state.context.config.tenant_id,
                workspace_id=req.workspace_id or state.context.config.workspace_id,
                metadata=req.metadata,
            )

        # 1a. Ingest attachments into vector store (if any)
        ingestion_results: List[IngestionResult] = []
        if req.attachments:
            if state.context.ingestion_service is None:
                raise ValueError(
                    "Attachments were provided but ingestion_service is not configured. "
                    "Pass ingestion_service explicitly to control where attachments are indexed."
                )
    
            ingestion_results = await state.context.ingestion_service.ingest_attachments_for_session(
                attachments=req.attachments,
                session_id=session.id,
                user_id=req.user_id,
                tenant_id=session.tenant_id,
                workspace_id=session.workspace_id,
            )

        # 2. Append user message to session history
        user_message = self._build_session_message_from_request(req)
        await state.context.session_manager.append_message(session.id, user_message)

        # Reload the session to ensure we have the latest metadata
        session = await state.context.session_manager.get_session(session.id) or session

        # Initialize debug trace – history will be attached later
        debug_trace: Dict[str, Any] = {
            "session_id": session.id,
            "user_id": session.user_id,
        }

        if ingestion_results:
            debug_trace["ingestion"] = [
                {
                    "attachment_id": r.attachment_id,
                    "attachment_type": r.attachment_type,
                    "num_chunks": r.num_chunks,
                    "vector_ids_count": len(r.vector_ids),
                    "metadata": r.metadata,
                }
                for r in ingestion_results
            ]

        # Trace session and ingestion step.
        state.trace_event(
            component="engine",
            step="session_and_ingest",
            message="Session loaded/created and user message appended; attachments ingested.",
            data={
                "session_id": session.id,
                "user_id": session.user_id,
                "tenant_id": session.tenant_id,
                "attachments_count": len(req.attachments or []),
                "ingestion_results_count": len(ingestion_results),
            },
        )

        state.session = session
        state.ingestion_results = ingestion_results
        state.set_debug_section("session_and_ingest", debug_trace)

        # NOTE: state.base_history is intentionally left empty here.


    def _build_session_message_from_request(
        self,
        request: RuntimeRequest,
    ) -> ChatMessage:
        """
        Construct a ChatMessage from a RuntimeRequest to be stored in the session.

        Attachments from `request.attachments` are represented at the request level
        and can be linked via metadata if needed.
        """
        return ChatMessage(
            role="user",
            content=request.message,
            created_at=datetime.now(timezone.utc).isoformat(),
        )