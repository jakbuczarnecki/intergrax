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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Protocol

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
from intergrax.llm.conversational_memory import ChatMessage, MessageRole
from intergrax.runtime.drop_in_knowledge_mode.ingestion import (
    AttachmentIngestionService,
    IngestionResult,
)
from intergrax.runtime.drop_in_knowledge_mode.attachments import (
    FileSystemAttachmentResolver,
)
from intergrax.runtime.drop_in_knowledge_mode.context_builder import ContextBuilder

from .context_builder import RetrievedChunk, BuiltContext




@dataclass
class RagPromptBundle:
    """
    Container for prompt elements related to RAG:

    - system_prompt: final system prompt string to be sent to the model
      (may be equal to BuiltContext.system_prompt or modified).
    - context_messages: extra messages (usually system-level) injecting
      retrieved document context.
    """
    system_prompt: str
    context_messages: List[ChatMessage]


class RagPromptBuilder(Protocol):
    """
    Strategy interface for building the RAG-related part of the prompt.

    You can provide a custom implementation and pass it to
    DropInKnowledgeRuntime to fully control:

    - the exact system prompt text,
    - how retrieved chunks are formatted and injected as messages.
    """

    def build_rag_prompt(self, built: BuiltContext) -> RagPromptBundle:
        ...


class DefaultRagPromptBuilder(RagPromptBuilder):
    """
    Default prompt builder for Drop-In Knowledge Mode.

    Responsibilities:
    - Use the system_prompt from BuiltContext as-is.
    - If retrieved_chunks are present, format them into a single
      additional system-level message with natural, model-friendly text.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config

    def build_rag_prompt(self, built: BuiltContext) -> RagPromptBundle:
        # Start from the system prompt computed by ContextBuilder
        system_prompt = built.system_prompt

        context_messages: List[ChatMessage] = []

        if built.retrieved_chunks:
            rag_context_text = self._format_rag_context(built.retrieved_chunks)
            context_messages.append(
                ChatMessage(
                    role="system",
                    content=(
                        "The following excerpts were retrieved from the user's "
                        "documents. Use them as factual context when answering "
                        "the user's question.\n\n"
                        f"{rag_context_text}"
                    ),
                )
            )

        return RagPromptBundle(
            system_prompt=system_prompt,
            context_messages=context_messages,
        )


    def _format_rag_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Build a compact, model-friendly text block from retrieved chunks.

        Design goals:
        - Provide enough semantic context.
        - Avoid internal markers ([CTX ...], scores, ids) that the model
          could copy into the final answer.
        - Keep format simple and natural.
        """
        if not chunks:
            return ""

        lines: List[str] = []

        for ch in chunks:
            source_name = (
                ch.metadata.get("source_name")
                or ch.metadata.get("attachment_id")
                or "document"
            )
            lines.append(f"Source: {source_name}")
            lines.append("Excerpt:")
            lines.append(ch.text)
            lines.append("")  # blank line separator

        # Optional: add truncation based on config (e.g. max chars)
        # For now we keep full text and rely on upstream chunking.
        return "\n".join(lines)
    

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
      - Call the configured LLM adapter.
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
        ingestion_service: Optional[AttachmentIngestionService] = None,
        context_builder: Optional[ContextBuilder] = None,
        rag_prompt_builder: Optional[RagPromptBuilder] = None,
    ) -> None:
        self._config = config
        self._session_store = session_store
        self._ingestion_service = ingestion_service or AttachmentIngestionService(
            resolver=FileSystemAttachmentResolver(),
            embedding_manager=config.embedding_manager,
            vectorstore_manager=config.vectorstore_manager,
        )
        self._context_builder = context_builder or ContextBuilder(
            config=config,
            vectorstore_manager=config.vectorstore_manager,
        )
        self._rag_prompt_builder: RagPromptBuilder = (
            rag_prompt_builder or DefaultRagPromptBuilder(config)
        )


    async def ask(self, request: RuntimeRequest) -> RuntimeAnswer:
        """
        Main async entrypoint for the runtime.

        Steps:
        1. Load or create a session for (user_id, session_id).
        2. Ingest attachments (if any) into the vector store (RAG backend).
        3. Append the user message (with attachments) to the session.
        4. Build an LLM-ready context:
            - system prompt,
            - reduced chat history,
            - optional retrieved chunks from documents (RAG).
        5. Call the LLM adapter with the constructed messages.
        6. Append the assistant message to the session.
        7. Return a RuntimeAnswer object.

        This method is designed to be "ChatGPT-like":
        - if the user attaches files to a message, they are ingested automatically
        and used as context for the current and subsequent turns in the session,
        without any extra configuration or flags.
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

        # 1a. If there are attachments in the request, ingest them into RAG
        ingestion_results: List[IngestionResult] = []
        if request.attachments:
            ingestion_results = await self._ingestion_service.ingest_attachments_for_session(
                attachments=request.attachments,
                session_id=session.id,
                user_id=request.user_id,
                tenant_id=session.tenant_id,
                workspace_id=session.workspace_id,
            )

        # 2. Append user message to the session
        user_message = self._build_session_message_from_request(request)
        await self._session_store.append_message(session.id, user_message)

        # Reload the session to ensure we have the latest messages
        session = await self._session_store.get_session(session.id) or session

        debug_trace: Dict[str, Any] = {
            "session_id": session.id,
            "user_id": session.user_id,
            "config": {
                "llm_label": self._config.llm_label,
                "embedding_label": self._config.embedding_label,
                "vectorstore_label": self._config.vectorstore_label,
            },
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

        # 3. Build LLM-ready context
        messages_for_llm: List[ChatMessage] = []
        used_rag_flag: bool = False
        
        if (self._config.enable_rag
            and hasattr(self, "_context_builder")
            and self._context_builder is not None
        ):
            # 3a. Use ContextBuilder to construct:
            #     - system prompt
            #     - reduced history
            #     - retrieved chunks from vector store
            built = await self._context_builder.build_context(
                session=session,
                request=request,
            )

            debug_trace["rag"] = built.rag_debug_info
            used_rag_flag = bool(built.retrieved_chunks)

            # 3a.1 Delegated RAG prompt building:
            #      system prompt + context messages from RagPromptBuilder.
            bundle = self._rag_prompt_builder.build_rag_prompt(built)

            # Final system prompt (possibly customized)
            messages_for_llm.append(
                ChatMessage(role="system", content=bundle.system_prompt)
            )

            # Any additional context messages (usually system-level)
            messages_for_llm.extend(bundle.context_messages)

            # 3a.2 Reduced chat history
            messages_for_llm.extend(built.history_messages or [])

            # 3a.3 Current user question
            messages_for_llm.append(
                ChatMessage(role="user", content=request.message)
            )

            debug_trace["history_length"] = len(built.history_messages or [])
            debug_trace["rag_chunks"] = len(built.retrieved_chunks or [])

        else:
            # 3b. Fallback: legacy behavior without ContextBuilder / RAG
            history = self._build_chat_history(session)
            messages_for_llm = history
            debug_trace["history_length"] = len(history)
            used_rag_flag = False

        # 4. Call the LLM adapter using the constructed messages.
        #    We use the generic `generate_messages(...)` API provided by
        #    LangChainOllamaAdapter (and other Intergrax adapters with the
        #    same interface).
        try:
            answer_text = self._config.llm_adapter.generate_messages(
                messages_for_llm,
                max_tokens=self._config.max_output_tokens,
                # temperature can be passed here later once you add it to RuntimeConfig
            )
        except Exception as e:
            # In case of any adapter failure, we return a diagnostic message.
            answer_text = f"[ERROR] LLM adapter failed: {e}"

        # 5. Append assistant message to the session
        assistant_session_msg = SessionMessage(
            id=str(uuid.uuid4()),
            role="assistant",  # MessageRole alias; type-compatible with the rest of the framework
            content=answer_text,
            created_at=datetime.now(timezone.utc),
            attachments=[],
            metadata={"placeholder": False},
        )
        await self._session_store.append_message(session.id, assistant_session_msg)

        # 6. Build RouteInfo and RuntimeStats
        strategy: str
        if used_rag_flag and self._config.enable_rag:
            strategy = "llm_with_rag_context_builder"
        elif ingestion_results:
            strategy = "llm_only_with_ingestion"
        else:
            strategy = "llm_only"

        route_info = RouteInfo(
            used_rag=used_rag_flag and self._config.enable_rag,
            used_websearch=False,
            used_tools=False,
            used_long_term_memory=False,
            used_user_profile=self._config.enable_user_profile_memory,
            strategy=strategy,
            extra={},
        )

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

        # 7. Return RuntimeAnswer
        return RuntimeAnswer(
            answer=answer_text,
            citations=[],
            route=route_info,
            tool_calls=[],
            stats=stats,
            raw_model_output=None,
            debug_trace=debug_trace,
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

    def _format_rag_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Build a compact, model-friendly text block from retrieved chunks.

        Design goals:
        - Provide enough semantic context.
        - Avoid internal markers ([CTX ...], scores, ids) that the model could
        copy into the final answer.
        - Keep format simple and natural.
        """
        if not chunks:
            return ""

        lines: List[str] = []
        for ch in chunks:
            source_name = ch.metadata.get("source_name") or ch.metadata.get("attachment_id") or "document"
            lines.append(f"Source: {source_name}")
            lines.append("Excerpt:")
            lines.append(ch.text)
            lines.append("")  # blank line separator

        # You can add truncation logic here if needed, e.g. based on character count.
        return "\n".join(lines)
    

    def _build_session_message_from_request(
        self,
        request: RuntimeRequest,
    ) -> SessionMessage:
        """
        Construct a SessionMessage from a RuntimeRequest.

        Attachments from `request.attachments` are stored directly in the
        `attachments` field, so that the session fully represents what
        the user has provided (text + files).
        """
        return SessionMessage(
            id=str(uuid.uuid4()),
            role="user",  # MessageRole; kept as a literal for compatibility
            content=request.message,
            created_at=datetime.now(timezone.utc),
            attachments=list(request.attachments),
            metadata={
                "runtime_request_metadata": request.metadata,
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
            # Normalize created_at to a string (ISO-like) regardless of how it is stored.
            # In some SessionStore implementations it may already be a string.
            created_at_value = getattr(msg, "created_at", None)

            if hasattr(created_at_value, "isoformat"):
                # datetime or similar object
                created_at_str = created_at_value.isoformat()
            else:
                # already a string or None; we pass it through
                created_at_str = created_at_value

            chat_msg = ChatMessage(
                role=msg.role,  # MessageRole
                content=msg.content,
                created_at=created_at_str,
                # tool_call_id, name, tool_calls will be filled later if needed
            )
            history.append(chat_msg)

        return history
