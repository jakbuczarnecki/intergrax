# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Core runtime engine for Drop-In Knowledge Mode.

This module defines the `DropInKnowledgeRuntime` class, which:
  - loads or creates chat sessions,
  - appends user messages,
  - builds a conversation history for the LLM,
  - (in the future) augments context with RAG, web search, tools and memory,
  - produces a `RuntimeAnswer` object as a high-level response.

At this stage, the engine is a skeleton:
  - session handling is implemented,
  - LLM integration is left as a TODO,
  - no RAG / web search / tools yet.

The goal is to provide a single, simple entrypoint that can be used from
FastAPI, Streamlit, MCP-like environments, CLI tools, etc.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import List, Optional

from intergrax.runtime.drop_in_knowledge_mode.config import RuntimeConfig
from intergrax.runtime.drop_in_knowledge_mode.response_schema import (
    RuntimeRequest,
    RuntimeAnswer,
    Citation,
    RouteInfo,
    RuntimeStats,
)
from intergrax.runtime.drop_in_knowledge_mode.session_store import (
    SessionStore,
    SessionMessage,
    ChatSession,
)
from intergrax.llm.conversational_memory import ChatMessage


class DropInKnowledgeRuntime:
    """
    High-level conversational runtime for the Intergrax framework.

    This class is designed to behave like a ChatGPT/Claude-style engine,
    but fully powered by Intergrax components (LLM adapters, RAG, web search,
    tools, memory, etc.).

    Responsibilities (current stage):
      - Accept a RuntimeRequest.
      - Load or create a ChatSession via SessionStore.
      - Append the user message to the session.
      - Build a list of ChatMessage objects representing the conversation
        history (limited by RuntimeConfig).
      - Call the configured LLM adapter (TODO).
      - Produce a RuntimeAnswer with the final answer text and metadata.

    In later stages this engine will:
      - ingest attachments and index them for RAG,
      - build rich context from memory, RAG, web search and tools,
      - support agentic flows via a supervisor,
      - expose observability hooks and cost tracking.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        session_store: SessionStore,
    ) -> None:
        self._config = config
        self._session_store = session_store


    async def ask(self, request: RuntimeRequest) -> RuntimeAnswer:
        """
        Main async entrypoint for the runtime.

        Steps:
          1. Load or create a session for (user_id, session_id).
          2. Append the user message (with attachments) to the session.
          3. Build an LLM-ready conversation history.
          4. Call the LLM adapter (integrated with Intergrax LLMAdapter).
          5. Append the assistant message to the session.
          6. Return a RuntimeAnswer object.

        At this stage the LLM call is a placeholder and will be replaced with
        a real integration once the engine wiring is complete.
        """
        # 1. Load or create session
        session = await self._session_store.get_session(request.session_id)
        if session is None:
            session = await self._session_store.create_session(      
                session_id=request.session_id,          
                user_id=request.user_id,
                tenant_id=request.tenant_id or self._config.tenant_id,
                workspace_id=request.workspace_id or self._config.workspace_id,
                metadata=request.metadata,
            )

        # 2. Append user message
        user_message = self._build_session_message_from_request(request)
        await self._session_store.append_message(session.id, user_message)

        # Reload the session to ensure we have the latest messages
        session = await self._session_store.get_session(session.id) or session

        # 3. Build conversation history for the LLM
        history = self._build_chat_history(session)

        # 4. Call the LLM adapter using the conversation history.
        #    We use the generic `generate_messages(...)` API provided by
        #    LangChainOllamaAdapter (and other Intergrax adapters with the
        #    same interface).
        try:
            answer_text = self._config.llm_adapter.generate_messages(
                history,
                max_tokens=self._config.max_output_tokens,
                # temperature can be passed here later once you add it to RuntimeConfig
            )
        except Exception as e:
            # In case of any adapter failure, we return a diagnostic message.
            answer_text = f"[ERROR] LLM adapter failed: {e}"

        # 5. Append assistant message (placeholder) to the session
        assistant_session_msg = SessionMessage(
            id=str(uuid.uuid4()),
            role="assistant",  # MessageRole alias; type-compatible with the rest of the framework
            content=answer_text,
            created_at=datetime.now(timezone.utc),
            attachments=[],
            metadata={"placeholder": True},
        )
        await self._session_store.append_message(session.id, assistant_session_msg)

        # 6. Build and return RuntimeAnswer
        route_info = RouteInfo(
            used_rag=False,
            used_websearch=False,
            used_tools=False,
            used_long_term_memory=False,
            used_user_profile=self._config.enable_user_profile_memory,
            strategy="simple_placeholder",
            extra={},
        )

        stats = RuntimeStats(
            # Token information will be filled once LLM integration is in place.
            total_tokens=None,
            input_tokens=None,
            output_tokens=None,
            rag_tokens=None,
            websearch_tokens=None,
            tool_tokens=None,
            duration_ms=None,
            extra={},
        )

        return RuntimeAnswer(
            answer=answer_text,
            citations=[],
            route=route_info,
            tool_calls=[],
            stats=stats,
            raw_model_output=None,
            debug_trace={
                "history_length": len(history),
                "session_id": session.id,
                "user_id": session.user_id,
                "config": {
                    "llm_label": self._config.llm_label,
                    "embedding_label": self._config.embedding_label,
                    "vectorstore_label": self._config.vectorstore_label,
                },
            },
        )

    def ask_sync(self, request: RuntimeRequest) -> RuntimeAnswer:
        """
        Synchronous wrapper around `ask()`.

        Useful for environments where `await` is not easily available,
        such as simple scripts or some notebook setups.
        """
        return asyncio.run(self.ask(request))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_session_message_from_request(
        self,
        request: RuntimeRequest,
    ) -> SessionMessage:
        """
        Construct a SessionMessage from a RuntimeRequest.

        For now, attachments from `request.attachments` are not transformed
        into AttachmentRef objects yet. This will be handled in a dedicated
        ingestion pipeline in a later step.
        """
        return SessionMessage(
            id=str(uuid.uuid4()),
            role="user",  # MessageRole; kept as a literal for compatibility
            content=request.message,
            created_at=datetime.now(timezone.utc),
            attachments=[],
            metadata={
                "runtime_request_metadata": request.metadata,
                "attachment_placeholders": list(request.attachments),
            },
        )

    def _build_chat_history(self, session: ChatSession) -> List[ChatMessage]:
        """
        Build a list of ChatMessage objects from the stored session messages.

        Only the last `max_history_messages` messages are included, in order
        to keep the context size under control. Token-level truncation will be
        handled later using `max_history_tokens`.
        """
        messages = session.messages[-self._config.max_history_messages :]
        history: List[ChatMessage] = []

        for msg in messages:
            # We convert stored SessionMessage into ChatMessage understood
            # by the LLM adapters. At this stage we only copy role/content.
            chat_msg = ChatMessage(
                role=msg.role,  # MessageRole
                content=msg.content,
                created_at=msg.created_at.isoformat(),
                # tool_call_id, name, tool_calls will be filled later if needed
            )
            history.append(chat_msg)

        return history
