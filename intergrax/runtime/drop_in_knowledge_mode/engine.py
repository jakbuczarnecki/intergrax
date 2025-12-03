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

The goal is to provide a single, simple entrypoint that can be used from
FastAPI, Streamlit, MCP-like environments, CLI tools, etc.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple

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
from intergrax.runtime.drop_in_knowledge_mode.context_builder import ContextBuilder, RetrievedChunk
from intergrax.runtime.drop_in_knowledge_mode.websearch_prompt_builder import (
    DefaultWebSearchPromptBuilder,
    WebSearchPromptBuilder,
)
from intergrax.websearch.service.websearch_executor import WebSearchExecutor


class DropInKnowledgeRuntime:
    """
    High-level conversational runtime for the Intergrax framework.

    This class is designed to behave like a ChatGPT/Claude-style engine,
    but fully powered by Intergrax components (LLM adapters, RAG, web search,
    tools, memory, etc.).
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

        High-level pipeline:
          1. Load or create a session, ingest attachments, append user message.
          2. Build base conversational history from SessionStore.
          3. Build RAG + history layer (ContextBuilder).
          4. Optionally enrich with web search.
          5. Optionally run tools agent (planning + tool calls).
          6. Call core LLM adapter with fully built context.
          7. Append assistant answer to history and build RuntimeAnswer.
        """

        # 1) Session + ingestion + user message
        session, ingestion_results, base_history = await self._load_or_create_session_and_ingest(request)

        # 2) Initialize debug trace
        debug_trace: Dict[str, Any] = self._init_debug_trace(
            session=session,
            ingestion_results=ingestion_results,
            base_history=base_history,
        )

        # 3) Build RAG + history layer
        messages_for_llm: List[ChatMessage] = []
        tools_context_parts: List[str] = []

        (
            messages_for_llm,
            tools_context_parts,
            used_rag_flag,
            rag_debug,
            built_history_messages,
            history_includes_current_user,
        ) = await self._build_rag_and_history_layer(
            session=session,
            request=request,
            base_history=base_history,
            messages_for_llm=messages_for_llm,
            tools_context_parts=tools_context_parts,
        )

        if rag_debug:
            debug_trace["rag"] = rag_debug
        debug_trace["history_length"] = len(built_history_messages or base_history)
        debug_trace["rag_chunks"] = (
            rag_debug.get("rag_chunks", 0) if rag_debug else 0
        )

        # 4) Web search layer
        (
            messages_for_llm,
            tools_context_parts,
            used_websearch_flag,
            websearch_debug,
        ) = await self._run_websearch_layer(
            request=request,
            messages_for_llm=messages_for_llm,
            tools_context_parts=tools_context_parts,
        )

        if websearch_debug:
            debug_trace["websearch"] = websearch_debug

        # Ensure current user message is present as the last user message,
        # if the history builder did not already include it.
        if not history_includes_current_user:
            messages_for_llm.append(ChatMessage(role="user", content=request.message))

        # 5) Tools layer
        used_tools_flag: bool = False
        tools_agent_answer: Optional[str] = None
        tool_traces: List[Dict[str, Any]] = []

        (
            messages_for_llm,
            used_tools_flag,
            tools_agent_answer,
            tool_traces,
            tools_debug,
        ) = await self._run_tools_layer(
            request=request,
            messages_for_llm=messages_for_llm,
            tools_context_parts=tools_context_parts,
            base_history=base_history,
            built_history_messages=built_history_messages,
        )

        if tools_debug:
            debug_trace["tools"] = tools_debug
        if "tools_error" in tools_debug:
            debug_trace["tools_error"] = tools_debug["tools_error"]

        # 6) Core LLM call
        answer_text = self._call_core_llm(
            messages_for_llm=messages_for_llm,
            tools_agent_answer=tools_agent_answer,
            used_tools_flag=used_tools_flag,
            debug_trace=debug_trace,
        )

        if not isinstance(answer_text, str) or not answer_text.strip():
            answer_text = (
                str(tools_agent_answer)
                if tools_agent_answer
                else "[ERROR] Empty answer from runtime."
            )

        # 7) Append assistant message to the session
        assistant_message = ChatMessage(
            role="assistant",
            content=answer_text,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        await self._session_store.append_message(session.id, assistant_message)

        # 8) Build RouteInfo and RuntimeStats
        if used_rag_flag and used_websearch_flag and used_tools_flag:
            strategy = "llm_with_rag_websearch_and_tools"
        elif used_rag_flag and used_tools_flag:
            strategy = "llm_with_rag_and_tools"
        elif used_websearch_flag and used_tools_flag:
            strategy = "llm_with_websearch_and_tools"
        elif used_tools_flag:
            strategy = "llm_with_tools"
        elif used_rag_flag and used_websearch_flag:
            strategy = "llm_with_rag_and_websearch"
        elif used_rag_flag:
            strategy = "llm_with_rag_context_builder"
        elif used_websearch_flag:
            strategy = "llm_with_websearch"
        elif ingestion_results:
            strategy = "llm_only_with_ingestion"
        else:
            strategy = "llm_only"

        route_info = RouteInfo(
            used_rag=used_rag_flag and self._config.enable_rag,
            used_websearch=used_websearch_flag and self._config.enable_websearch,
            used_tools=used_tools_flag and self._config.tools_mode != "off",
            # These flags will be wired when profile / LTM memory are integrated.
            used_long_term_memory=False,
            used_user_profile=False,
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

        tool_calls_for_answer: List[ToolCallInfo] = []
        for t in tool_traces:
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
    # Step 1: Session + ingestion + user message
    # ------------------------------------------------------------------

    async def _load_or_create_session_and_ingest(
        self,
        request: RuntimeRequest,
    ) -> Tuple[ChatSession, List[IngestionResult], List[ChatMessage]]:
        """
        Load or create a session, ingest attachments if present,
        append the user message and return:

          - ChatSession,
          - ingestion results,
          - base conversational history.
        """
        session = await self._session_store.get_session(request.session_id)
        if session is None:
            session = await self._session_store.create_session(
                session_id=request.session_id,
                user_id=request.user_id,
                tenant_id=request.tenant_id or self._config.tenant_id,
                workspace_id=request.workspace_id or self._config.workspace_id,
                metadata=request.metadata,
            )

        ingestion_results: List[IngestionResult] = []
        if request.attachments:
            ingestion_results = await self._ingestion_service.ingest_attachments_for_session(
                attachments=request.attachments,
                session_id=session.id,
                user_id=request.user_id,
                tenant_id=session.tenant_id,
                workspace_id=session.workspace_id,
            )

        user_message = self._build_session_message_from_request(request)
        await self._session_store.append_message(session.id, user_message)

        # Reload the session to ensure we have the latest metadata
        session = await self._session_store.get_session(session.id) or session

        base_history: List[ChatMessage] = self._build_chat_history(session)
        return session, ingestion_results, base_history

    def _init_debug_trace(
        self,
        *,
        session: ChatSession,
        ingestion_results: List[IngestionResult],
        base_history: List[ChatMessage],
    ) -> Dict[str, Any]:
        """
        Initialize a debug trace structure with core metadata and
        ingestion information.
        """
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

        debug_trace["base_history_length"] = len(base_history)
        return debug_trace

    # ------------------------------------------------------------------
    # Step 3: RAG + history layer
    # ------------------------------------------------------------------

    async def _build_rag_and_history_layer(
        self,
        *,
        session: ChatSession,
        request: RuntimeRequest,
        base_history: List[ChatMessage],
        messages_for_llm: List[ChatMessage],
        tools_context_parts: List[str],
    ) -> Tuple[
        List[ChatMessage],
        List[str],
        bool,
        Dict[str, Any],
        List[ChatMessage],
        bool,
    ]:
        """
        Build the RAG + history layer using ContextBuilder (if configured).

        Returns:
          - updated messages_for_llm,
          - updated tools_context_parts,
          - used_rag_flag,
          - rag_debug_info dict,
          - history_messages used for tools scope "CONVERSATION",
          - history_includes_current_user flag.
        """
        used_rag_flag: bool = False
        rag_debug: Dict[str, Any] = {}
        history_includes_current_user: bool = False
        history_messages: List[ChatMessage] = []

        if self._context_builder is not None:
            built = await self._context_builder.build_context(
                session=session,
                request=request,
                base_history=base_history,
            )

            rag_info = built.rag_debug_info or {}
            rag_debug = rag_info

            used_rag_flag = bool(rag_info.get("used", bool(built.retrieved_chunks)))

            if used_rag_flag:
                # RAG-specific prompt construction
                bundle = self._rag_prompt_builder.build_rag_prompt(built)

                # System prompt from RAG layer
                messages_for_llm.append(
                    ChatMessage(role="system", content=bundle.system_prompt)
                )

                # Additional context messages built from retrieved chunks
                messages_for_llm.extend(bundle.context_messages or [])

                # Compact textual form of RAG context for tools agent
                rag_context_text = self._format_rag_context(built.retrieved_chunks or [])
                if rag_context_text:
                    tools_context_parts.append("RAG CONTEXT:\n" + rag_context_text)

                rag_debug["rag_chunks"] = len(built.retrieved_chunks or [])
            else:
                rag_debug["rag_chunks"] = 0

            # Conversation history – always appended, regardless of RAG
            history_messages = built.history_messages or []
            messages_for_llm.extend(history_messages)
            history_includes_current_user = True
        else:
            # No ContextBuilder configured – fall back to a simple history-only prompt.
            messages_for_llm.extend(base_history)
            history_messages = base_history
            history_includes_current_user = True
            used_rag_flag = False
            rag_debug["rag_chunks"] = 0

        return (
            messages_for_llm,
            tools_context_parts,
            used_rag_flag,
            rag_debug,
            history_messages,
            history_includes_current_user,
        )

    # ------------------------------------------------------------------
    # Step 4: Web search layer
    # ------------------------------------------------------------------

    async def _run_websearch_layer(
        self,
        *,
        request: RuntimeRequest,
        messages_for_llm: List[ChatMessage],
        tools_context_parts: List[str],
    ) -> Tuple[List[ChatMessage], List[str], bool, Dict[str, Any]]:
        """
        Run optional web search and inject results as context messages.
        """
        used_websearch_flag: bool = False
        websearch_debug: Dict[str, Any] = {}

        if (
            self._config.enable_websearch
            and self._websearch_executor is not None
            and self._websearch_prompt_builder is not None
        ):
            try:
                web_docs = await self._websearch_executor.search_async(
                    query=request.message,
                    top_k=self._config.max_docs_per_query,
                    language=None,
                    top_n_fetch=None,
                    serialize=True,
                )

                if web_docs:
                    used_websearch_flag = True

                    bundle = self._websearch_prompt_builder.build_websearch_prompt(
                        web_docs
                    )

                    # Inject web search context messages AFTER RAG/system/history,
                    # but BEFORE the final user message.
                    messages_for_llm.extend(bundle.context_messages or [])

                    if bundle.debug_info:
                        websearch_debug.update(bundle.debug_info)

                    # Tools context from websearch (compact textual form)
                    web_context_texts: List[str] = []
                    for msg in bundle.context_messages or []:
                        if getattr(msg, "content", None):
                            web_context_texts.append(msg.content)
                    if web_context_texts:
                        tools_context_parts.append(
                            "WEBSEARCH CONTEXT:\n" + "\n\n".join(web_context_texts)
                        )

            except Exception as exc:
                websearch_debug["error"] = str(exc)

        return messages_for_llm, tools_context_parts, used_websearch_flag, websearch_debug

    # ------------------------------------------------------------------
    # Step 5: Tools layer
    # ------------------------------------------------------------------

    async def _run_tools_layer(
        self,
        *,
        request: RuntimeRequest,
        messages_for_llm: List[ChatMessage],
        tools_context_parts: List[str],
        base_history: List[ChatMessage],
        built_history_messages: List[ChatMessage],
    ) -> Tuple[
        List[ChatMessage],
        bool,
        Optional[str],
        List[Dict[str, Any]],
        Dict[str, Any],
    ]:
        """
        Run tools agent layer if configured.

        Returns:
          - updated messages_for_llm,
          - used_tools_flag,
          - tools_agent_answer (if any),
          - tool_traces,
          - tools_debug_info dict.
        """
        tools_debug: Dict[str, Any] = {}
        tools_agent_answer: Optional[str] = None
        tool_traces: List[Dict[str, Any]] = []
        used_tools_flag: bool = False

        use_tools = (
            self._config.tools_agent is not None
            and self._config.tools_mode != "off"
        )

        if not use_tools:
            return messages_for_llm, used_tools_flag, tools_agent_answer, tool_traces, tools_debug

        tools_context = (
            "\n\n".join(tools_context_parts).strip()
            if tools_context_parts
            else None
        )

        try:
            # Decide what to pass as input_data for the tools agent,
            # depending on configured scope.
            if self._config.tools_context_scope == ToolsContextScope.CURRENT_MESSAGE_ONLY:
                agent_input = request.message

            elif self._config.tools_context_scope == ToolsContextScope.CONVERSATION:
                # If ContextBuilder was used, we have built_history_messages.
                # Otherwise fall back to base_history.
                if built_history_messages:
                    agent_input = built_history_messages
                else:
                    agent_input = base_history

            else:
                # FULL_CONTEXT (or future additional scopes):
                # pass the entire message list built so far.
                agent_input = messages_for_llm

            tools_result = self._config.tools_agent.run(
                input_data=agent_input,
                context=tools_context,
                stream=False,
                tool_choice=None,
                output_model=None,
            )
            tools_agent_answer = tools_result.get("answer", "") or None
            tool_traces = tools_result.get("tool_traces") or []
            used_tools_flag = bool(tool_traces)

            tools_debug.update(
                {
                    "mode": self._config.tools_mode,
                    "used_tools": used_tools_flag,
                    "tool_traces": tool_traces,
                }
            )
            if tools_agent_answer:
                tools_debug["agent_answer_preview"] = str(tools_agent_answer)[:200]
            if self._config.tools_mode == "required" and not used_tools_flag:
                tools_debug[
                    "warning"
                ] = "tools_mode='required' but no tools were invoked by the tools_agent."

            # Inject executed tool calls as system context for the core LLM
            if tool_traces:
                tool_lines: List[str] = []
                for t in tool_traces:
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
                    messages_for_llm.append(
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
            tools_debug["tools_error"] = str(e)

        return messages_for_llm, used_tools_flag, tools_agent_answer, tool_traces, tools_debug

    # ------------------------------------------------------------------
    # Step 6: Core LLM call
    # ------------------------------------------------------------------

    def _call_core_llm(
        self,
        *,
        messages_for_llm: List[ChatMessage],
        tools_agent_answer: Optional[str],
        used_tools_flag: bool,
        debug_trace: Dict[str, Any],
    ) -> str:
        """
        Call the core LLM adapter and decide on the final answer text,
        possibly falling back to tools_agent_answer when needed.
        """
        if used_tools_flag and tools_agent_answer:
            return str(tools_agent_answer)

        try:
            raw_answer = self._config.llm_adapter.generate_messages(
                messages_for_llm,
                max_tokens=self._config.max_output_tokens,
            )

            if isinstance(raw_answer, str):
                return raw_answer

            content = getattr(raw_answer, "content", None)
            if isinstance(content, str) and content.strip():
                return content

            return str(raw_answer)

        except Exception as e:
            debug_trace["llm_error"] = str(e)

            if tools_agent_answer:
                return (
                    "[ERROR] LLM adapter failed, falling back to tools agent answer.\n"
                    f"Details: {e}\n\n"
                    f"{tools_agent_answer}"
                )

            return f"[ERROR] LLM adapter failed: {e}"

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
        Build a conversation history for the LLM using the SessionStore.

        The engine does not know how the history is constructed internally:
        SessionStore is free to use ConversationalMemory, additional
        filters, tagging, or other memory mechanisms.
        """
        return self._session_store.get_conversation_history(session=session)
