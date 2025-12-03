# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Core runtime engine for Drop-In Knowledge Mode.

This module defines the `DropInKnowledgeRuntime` class, which:
  - loads or creates chat sessions,
  - appends user messages,
  - builds a conversation history for the LLM,
  - augments context with RAG, web search and tools,
  - produces a `RuntimeAnswer` object as a high-level response.

The goal is to provide a single, simple entrypoint that can be used from
FastAPI, Streamlit, MCP-like environments, CLI tools, etc.

Refactored as a stateful pipeline:

  - RuntimeState holds all intermediate data (session, history, flags, debug).
  - Each step mutates the state and can be inspected in isolation.
  - ask() just wires the steps together in a readable order.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from intergrax.runtime.drop_in_knowledge_mode.config import RuntimeConfig, ToolsContextScope
from intergrax.runtime.drop_in_knowledge_mode.rag_prompt_builder import (
    DefaultRagPromptBuilder,
    RagPromptBuilder,
)
from intergrax.runtime.drop_in_knowledge_mode.response_schema import (
    RuntimeRequest,
    RuntimeAnswer,
    RouteInfo,
    RuntimeStats,
    ToolCallInfo,
)
from intergrax.runtime.drop_in_knowledge_mode.session_store import (
    SessionStore,
    ChatSession,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.drop_in_knowledge_mode.ingestion import (
    AttachmentIngestionService,
    IngestionResult,
)
from intergrax.runtime.drop_in_knowledge_mode.attachments import (
    FileSystemAttachmentResolver,
)
from intergrax.runtime.drop_in_knowledge_mode.context_builder import (
    ContextBuilder,
    RetrievedChunk,
)
from intergrax.runtime.drop_in_knowledge_mode.websearch_prompt_builder import (
    DefaultWebSearchPromptBuilder,
    WebSearchPromptBuilder,
)
from intergrax.websearch.service.websearch_executor import WebSearchExecutor


# ----------------------------------------------------------------------
# Stateful pipeline model
# ----------------------------------------------------------------------


@dataclass
class RuntimeState:
    """
    Mutable state object passed through the runtime pipeline.

    It aggregates:
      - request and session metadata,
      - ingestion results,
      - conversation history and model-ready messages,
      - flags indicating which subsystems were used (RAG, websearch, tools, memory),
      - tools traces and agent answer,
      - full debug_trace for observability & diagnostics.
    """

    # Input
    request: RuntimeRequest

    # Session and ingestion
    session: Optional[ChatSession] = None
    ingestion_results: List[IngestionResult] = field(default_factory=list)

    # Conversation / context
    base_history: List[ChatMessage] = field(default_factory=list)
    messages_for_llm: List[ChatMessage] = field(default_factory=list)
    tools_context_parts: List[str] = field(default_factory=list)
    built_history_messages: List[ChatMessage] = field(default_factory=list)
    history_includes_current_user: bool = False

    # ContextBuilder intermediate result (history + retrieved chunks)
    context_builder_result: Optional[Any] = None

    # Usage flags
    used_rag: bool = False
    used_websearch: bool = False
    used_tools: bool = False
    used_long_term_memory: bool = False  # reserved for future integration
    used_user_profile: bool = False      # reserved for future integration

    # Tools
    tools_agent_answer: Optional[str] = None
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)

    # Debug / diagnostics
    debug_trace: Dict[str, Any] = field(default_factory=dict)
    websearch_debug: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# DropInKnowledgeRuntime
# ----------------------------------------------------------------------


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
      - Build an LLM-ready context:
          * system prompt(s),
          * chat history from SessionStore,
          * optional retrieved chunks from documents (RAG),
          * optional web search context (if enabled),
          * optional tools results.
      - Call the main LLM adapter once with the fully enriched context
        to produce the final answer.
      - Append the assistant message to the session.
      - Return a RuntimeAnswer with the final answer text and metadata.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        session_store: SessionStore,
        ingestion_service: Optional[AttachmentIngestionService] = None,
        context_builder: Optional[ContextBuilder] = None,
        rag_prompt_builder: Optional[RagPromptBuilder] = None,
        websearch_prompt_builder: Optional[WebSearchPromptBuilder] = None,
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

        self._websearch_executor: Optional[WebSearchExecutor] = None
        if self._config.enable_websearch and self._config.websearch_executor:
            # Use user-supplied executor instance
            self._websearch_executor = self._config.websearch_executor

        self._websearch_prompt_builder: Optional[WebSearchPromptBuilder] = (
            websearch_prompt_builder or DefaultWebSearchPromptBuilder(config)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ask(self, request: RuntimeRequest) -> RuntimeAnswer:
        """
        Main async entrypoint for the runtime.

        Pipeline:
          1. Session + ingestion + user message appended.
          2. Base history builder (load & preprocess conversation history).
          3. History layer (conversation history for the LLM).
          4. RAG layer (retrieval + RAG system/context messages).
          5. Web search layer (optional).
          6. Ensure current user message is present at the end of context.
          7. Tools layer (planning + tool calls).
          8. Core LLM call.
          9. Persist assistant answer and build RuntimeAnswer with route info.
        """
        state = RuntimeState(request=request)

        # Step 1: session lifecycle + ingestion & debug init (no history)
        await self._step_session_and_ingest(state)

        # Step 2: build base history (load & preprocess conversation history)
        await self._step_build_base_history(state)

        # Step 3: history layer
        await self._step_history(state)

        # Step 4: RAG layer
        await self._step_rag(state)

        # Step 5: Web search layer
        await self._step_websearch(state)

        # Step 6: Ensure current user message is in the final prompt
        self._ensure_current_user_message(state)

        # Step 7: Tools layer
        await self._step_tools(state)

        # Step 8: Core LLM
        answer_text = self._step_core_llm(state)

        # Step 9: Persist assistant message and build RuntimeAnswer
        runtime_answer = await self._step_persist_and_build_answer(state, answer_text)
        return runtime_answer



    def ask_sync(self, request: RuntimeRequest) -> RuntimeAnswer:
        """
        Synchronous wrapper around `ask()`.

        Useful for environments where `await` is not easily available,
        such as simple scripts or some notebook setups.
        """
        return asyncio.run(self.ask(request))

    # ------------------------------------------------------------------
    # Step 1: session + ingestion (no history)
    # ------------------------------------------------------------------

    async def _step_session_and_ingest(self, state: RuntimeState) -> None:
        """
        Load or create a session, ingest attachments (RAG), append the user
        message and initialize debug_trace.

        IMPORTANT:
          - This step does NOT load conversation history.
          - History is loaded and preprocessed in `_step_build_base_history`.
        """
        req = state.request

        # 1. Load or create session
        session = await self._session_store.get_session(req.session_id)
        if session is None:
            session = await self._session_store.create_session(
                session_id=req.session_id,
                user_id=req.user_id,
                tenant_id=req.tenant_id or self._config.tenant_id,
                workspace_id=req.workspace_id or self._config.workspace_id,
                metadata=req.metadata,
            )

        # 1a. Ingest attachments into vector store (if any)
        ingestion_results: List[IngestionResult] = []
        if req.attachments:
            ingestion_results = await self._ingestion_service.ingest_attachments_for_session(
                attachments=req.attachments,
                session_id=session.id,
                user_id=req.user_id,
                tenant_id=session.tenant_id,
                workspace_id=session.workspace_id,
            )

        # 2. Append user message to session history
        user_message = self._build_session_message_from_request(req)
        await self._session_store.append_message(session.id, user_message)

        # Reload the session to ensure we have the latest metadata
        session = await self._session_store.get_session(session.id) or session

        # Initialize debug trace – history will be attached later
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

        state.session = session
        state.ingestion_results = ingestion_results
        state.debug_trace = debug_trace
        # NOTE: state.base_history is intentionally left empty here.


    # ------------------------------------------------------------------
    # Step 2: build base history (load & preprocess)
    # ------------------------------------------------------------------

    async def _step_build_base_history(self, state: RuntimeState) -> None:
        """
        Load and preprocess the conversation history for the current session.

        This step is the single place where we:
          - fetch the full session history from SessionStore,
          - optionally truncate it to the last N messages,
          - optionally compute token usage,
          - optionally summarize older parts of the conversation.

        The resulting `state.base_history` is treated as the canonical,
        preprocessed conversation history for all subsequent steps.
        """
        session = state.session
        assert session is not None, "Session must be set before building history."

        # 1. Load raw history from SessionStore
        raw_history: List[ChatMessage] = self._build_chat_history(session)

        # 2. Preprocess history for the current model/context limits.
        #    For now this is a no-op; in the future you can:
        #      - limit to last N messages,
        #      - compute token counts,
        #      - summarize older messages on the fly.
        base_history = raw_history

        state.base_history = base_history

        # Update debug trace with history-related info
        state.debug_trace["base_history_length"] = len(base_history)


    
    # ------------------------------------------------------------------
    # Step 3: history
    # ------------------------------------------------------------------

    async def _step_history(self, state: RuntimeState) -> None:
        """
        Build conversation history for the LLM.

        This step is responsible only for selecting and shaping the
        conversational context (previous user/assistant turns).

        Retrieval (RAG) is handled separately in `_step_rag`.
        """
        session = state.session
        assert session is not None, "Session must be set before history step."
        req = state.request
        base_history = state.base_history

        if self._context_builder is not None:
            # Delegate history shaping (truncation, system message stitching, etc.)
            # to ContextBuilder, but do NOT inject RAG here.
            built = await self._context_builder.build_context(
                session=session,
                request=req,
                base_history=base_history,
            )

            # Keep the full result for the RAG step.
            state.context_builder_result = built

            history_messages = built.history_messages or []
            state.messages_for_llm.extend(history_messages)
            state.built_history_messages = history_messages
            state.history_includes_current_user = True
            state.debug_trace["history_length"] = len(history_messages)
        else:
            # No ContextBuilder configured – use raw history from SessionStore.
            state.messages_for_llm.extend(base_history)
            state.built_history_messages = base_history
            state.history_includes_current_user = True
            state.debug_trace["history_length"] = len(base_history)


    # ------------------------------------------------------------------
    # Step 4: RAG
    # ------------------------------------------------------------------

    async def _step_rag(self, state: RuntimeState) -> None:
        """
        Build RAG layer (if configured) on top of the already constructed
        conversation history.

        This step:
          - uses the result from ContextBuilder (if available),
          - injects RAG system and context messages,
          - prepares a compact text summary of retrieved chunks for tools.
        """
        # Default values in case RAG is disabled or no chunks are retrieved.
        state.used_rag = False
        state.debug_trace.setdefault("rag_chunks", 0)

        # If RAG is globally disabled, do nothing.
        if not getattr(self._config, "enable_rag", True):
            return

        # We need ContextBuilder to produce retrieved chunks.
        if self._context_builder is None:
            return

        built = state.context_builder_result

        # In normal flow _step_history should have already called build_context
        # and stored the result. As a fallback, we can call it here.
        if built is None:
            session = state.session
            assert session is not None, "Session must be set before RAG step."
            built = await self._context_builder.build_context(
                session=session,
                request=state.request,
                base_history=state.base_history,
            )
            state.context_builder_result = built

        rag_info = getattr(built, "rag_debug_info", None) or {}
        state.debug_trace["rag"] = rag_info

        retrieved_chunks = getattr(built, "retrieved_chunks", None) or []
        state.used_rag = bool(rag_info.get("used", bool(retrieved_chunks)))

        if not state.used_rag:
            state.debug_trace["rag_chunks"] = 0
            return

        # RAG-specific prompt construction
        bundle = self._rag_prompt_builder.build_rag_prompt(built)

        # System prompt from RAG layer
        state.messages_for_llm.append(
            ChatMessage(role="system", content=bundle.system_prompt)
        )

        # Additional context messages built from retrieved chunks
        context_messages = bundle.context_messages or []
        state.messages_for_llm.extend(context_messages)

        # Compact textual form of RAG context for tools agent
        rag_context_text = self._format_rag_context(retrieved_chunks)
        if rag_context_text:
            state.tools_context_parts.append("RAG CONTEXT:\n" + rag_context_text)

        state.debug_trace["rag_chunks"] = len(retrieved_chunks)


    # ------------------------------------------------------------------
    # Step 5: Web search
    # ------------------------------------------------------------------

    async def _step_websearch(self, state: RuntimeState) -> None:
        """
        Run optional web search and inject results as context messages.
        """
        state.websearch_debug = {}
        state.used_websearch = False

        if (
            not self._config.enable_websearch
            or self._websearch_executor is None
            or self._websearch_prompt_builder is None
        ):
            return

        try:
            web_docs = await self._websearch_executor.search_async(
                query=state.request.message,
                top_k=self._config.max_docs_per_query,
                language=None,
                top_n_fetch=None,
                serialize=True,
            )

            if not web_docs:
                return

            state.used_websearch = True

            bundle = self._websearch_prompt_builder.build_websearch_prompt(web_docs)
            context_messages = bundle.context_messages or []
            state.messages_for_llm.extend(context_messages)
            state.websearch_debug.update(bundle.debug_info or {})

            # Compact textual context for tools
            web_context_texts: List[str] = []
            for msg in context_messages:
                if getattr(msg, "content", None):
                    web_context_texts.append(msg.content)
            if web_context_texts:
                state.tools_context_parts.append(
                    "WEBSEARCH CONTEXT:\n" + "\n\n".join(web_context_texts)
                )

        except Exception as exc:
            state.websearch_debug["error"] = str(exc)

        if state.websearch_debug:
            state.debug_trace["websearch"] = state.websearch_debug

    # ------------------------------------------------------------------
    # Step 6: Ensure current user message
    # ------------------------------------------------------------------

    def _ensure_current_user_message(self, state: RuntimeState) -> None:
        """
        Ensure that the latest user message is present in the final prompt.

        ContextBuilder may decide to already include it in history; if not,
        we append it explicitly as the last user message.
        """
        if not state.history_includes_current_user:
            state.messages_for_llm.append(
                ChatMessage(
                    role="user",
                    content=state.request.message,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            )

    # ------------------------------------------------------------------
    # Step 7: Tools
    # ------------------------------------------------------------------

    async def _step_tools(self, state: RuntimeState) -> None:
        """
        Run tools agent (planning + tool calls) if configured.

        The tools result is:
          - optionally used as the final answer (when tools_mode != "off"),
          - appended as system context for the core LLM.
        """
        state.used_tools = False
        state.tool_traces = []
        state.tools_agent_answer = None

        use_tools = (
            self._config.tools_agent is not None
            and self._config.tools_mode != "off"
        )
        if not use_tools:
            return

        tools_context = (
            "\n\n".join(state.tools_context_parts).strip()
            if state.tools_context_parts
            else None
        )

        debug_tools: Dict[str, Any] = {
            "mode": self._config.tools_mode,
        }

        try:
            # Decide what to pass as input_data for the tools agent.
            if self._config.tools_context_scope == ToolsContextScope.CURRENT_MESSAGE_ONLY:
                agent_input = state.request.message

            elif self._config.tools_context_scope == ToolsContextScope.CONVERSATION:
                # Use history built by ContextBuilder if available,
                # otherwise fall back to base_history.
                if state.built_history_messages:
                    agent_input = state.built_history_messages
                else:
                    agent_input = state.base_history

            else:
                # FULL_CONTEXT or any future scope:
                # pass entire message list built so far.
                agent_input = state.messages_for_llm

            tools_result = self._config.tools_agent.run(
                input_data=agent_input,
                context=tools_context,
                stream=False,
                tool_choice=None,
                output_model=None,
            )

            state.tools_agent_answer = tools_result.get("answer", "") or None
            state.tool_traces = tools_result.get("tool_traces") or []
            state.used_tools = bool(state.tool_traces)

            debug_tools["used_tools"] = state.used_tools
            debug_tools["tool_traces"] = state.tool_traces
            if state.tools_agent_answer:
                debug_tools["agent_answer_preview"] = str(state.tools_agent_answer)[:200]
            if self._config.tools_mode == "required" and not state.used_tools:
                debug_tools["warning"] = (
                    "tools_mode='required' but no tools were invoked by the tools_agent."
                )

            # Inject executed tool calls as system context for core LLM.
            if state.tool_traces:
                tool_lines: List[str] = []
                for t in state.tool_traces:
                    name = t.get("tool")
                    args = t.get("args")
                    output = t.get("output")

                    tool_lines.append(f"Tool '{name}' was called.")
                    if args is not None:
                        try:
                            args_str = json.dumps(args, ensure_ascii=False)
                        except Exception:
                            args_str = str(args)
                        tool_lines.append(f"Arguments: {args_str}")

                    if output is not None:
                        if isinstance(output, (dict, list)):
                            try:
                                out_str = json.dumps(output, ensure_ascii=False)
                            except Exception:
                                out_str = str(output)
                        else:
                            out_str = str(output)
                        tool_lines.append("Output:")
                        tool_lines.append(out_str)

                    tool_lines.append("")

                tools_context_for_llm = "\n".join(tool_lines).strip()
                if tools_context_for_llm:
                    state.messages_for_llm.append(
                        ChatMessage(
                            role="system",
                            content=(
                                "The following tool calls have been executed. "
                                "Use their results when answering the user.\n\n"
                                + tools_context_for_llm
                            ),
                        )
                    )

        except Exception as e:
            debug_tools["tools_error"] = str(e)

        state.debug_trace["tools"] = debug_tools

    # ------------------------------------------------------------------
    # Step 8: Core LLM
    # ------------------------------------------------------------------

    def _step_core_llm(self, state: RuntimeState) -> str:
        """
        Call the core LLM adapter and decide on the final answer text,
        possibly falling back to tools_agent_answer when needed.
        """
        # If tools were used and we have an explicit agent answer, prefer it.
        if state.used_tools and state.tools_agent_answer:
            return str(state.tools_agent_answer)

        try:
            raw_answer = self._config.llm_adapter.generate_messages(
                state.messages_for_llm,
                max_tokens=self._config.max_output_tokens,
            )

            if isinstance(raw_answer, str):
                return raw_answer

            content = getattr(raw_answer, "content", None)
            if isinstance(content, str) and content.strip():
                return content

            return str(raw_answer)

        except Exception as e:
            state.debug_trace["llm_error"] = str(e)

            if state.tools_agent_answer:
                return (
                    "[ERROR] LLM adapter failed, falling back to tools agent answer.\n"
                    f"Details: {e}\n\n"
                    f"{state.tools_agent_answer}"
                )

            return f"[ERROR] LLM adapter failed: {e}"

    # ------------------------------------------------------------------
    # Step 9: Persist answer & build RuntimeAnswer
    # ------------------------------------------------------------------

    async def _step_persist_and_build_answer(
        self,
        state: RuntimeState,
        answer_text: str,
    ) -> RuntimeAnswer:
        """
        Append assistant message to the session and build a RuntimeAnswer,
        including RouteInfo and RuntimeStats.
        """
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
        await self._session_store.append_message(session.id, assistant_message)

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
        elif state.ingestion_results:
            strategy = "llm_only_with_ingestion"
        else:
            strategy = "llm_only"

        route_info = RouteInfo(
            used_rag=state.used_rag and self._config.enable_rag,
            used_websearch=state.used_websearch and self._config.enable_websearch,
            used_tools=state.used_tools and self._config.tools_mode != "off",
            used_long_term_memory=state.used_long_term_memory,
            used_user_profile=state.used_user_profile,
            strategy=strategy,
            extra={},
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

        return RuntimeAnswer(
            answer=answer_text,
            citations=[],
            route=route_info,
            tool_calls=tool_calls_for_answer,
            stats=stats,
            raw_model_output=None,
            debug_trace=state.debug_trace,
        )

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
            source_name = (
                ch.metadata.get("source_name")
                or ch.metadata.get("attachment_id")
                or "document"
            )
            lines.append(f"Source: {source_name}")
            lines.append("Excerpt:")
            lines.append(ch.text)
            lines.append("")  # blank line separator

        return "\n".join(lines)

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

    def _build_chat_history(self, session: ChatSession) -> List[ChatMessage]:
        """
        Load raw conversation history for the given session.

        This method is responsible only for fetching history from SessionStore.
        Any model-specific preprocessing (truncation, summarization, token
        accounting) should happen in `_step_build_base_history`, not here.
        """
        return self._session_store.get_conversation_history(session=session)

