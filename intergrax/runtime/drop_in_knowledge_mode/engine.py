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
import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from intergrax.runtime.drop_in_knowledge_mode.config import RuntimeConfig
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
    SessionMessage,
    ChatSession,
)
from intergrax.llm.conversational_memory import ChatMessage
from intergrax.runtime.drop_in_knowledge_mode.ingestion import (
    AttachmentIngestionService,
    IngestionResult,
)
from intergrax.runtime.drop_in_knowledge_mode.attachments import (
    FileSystemAttachmentResolver,
)
from intergrax.runtime.drop_in_knowledge_mode.context_builder import ContextBuilder
from intergrax.runtime.drop_in_knowledge_mode.websearch_prompt_builder import (
    DefaultWebSearchPromptBuilder,
    WebSearchPromptBuilder,
)

from intergrax.runtime.drop_in_knowledge_mode.context_builder import RetrievedChunk

from intergrax.websearch.service.websearch_executor import WebSearchExecutor


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
        if self._config.enable_websearch:
            if self._config.websearch_executor:
                # Use user-supplied executor instance
                self._websearch_executor = self._config.websearch_executor

        self._websearch_prompt_builder: Optional[WebSearchPromptBuilder] = (
            websearch_prompt_builder or DefaultWebSearchPromptBuilder(config)
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
            - optional retrieved chunks from documents (RAG),
            - optional web search context (if enabled).
        5. Optionally execute tools (via IntergraxToolsAgent) and inject their
           results into the context.
        6. Call the main LLM adapter once with the fully enriched context
           (RAG + websearch + tools results) to produce the final answer.
        7. Append the assistant message to the session.
        8. Return a RuntimeAnswer object.

        This method is designed to be "ChatGPT-like":
        - if the user attaches files to a message, they are ingested automatically
          and used as context for the current and subsequent turns in the session,
          without any extra configuration or flags.
        - when tools are enabled, they are used as data sources (via IntergraxToolsAgent)
          and their results are injected into the context, but the final answer is
          always generated by the main LLM over the unified context.
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

        # 3. Build LLM-ready context (RAG + history + websearch)
        messages_for_llm: List[ChatMessage] = []
        history_includes_current_user: bool = False
        used_rag_flag: bool = False
        used_websearch_flag: bool = False
        used_tools_flag: bool = False
        tool_traces: List[Dict[str, Any]] = []

        # Textual context for tools agent (RAG + websearch layers).
        # This is a compact, human-readable summary that will be injected
        # as system context inside IntergraxToolsAgent.run(...).
        tools_context_parts: List[str] = []

        if (
            self._config.enable_rag
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

            # System prompt from RAG builder
            messages_for_llm.append(
                ChatMessage(role="system", content=bundle.system_prompt)
            )

            # Additional system-level/context messages (e.g. formatted RAG chunks)
            messages_for_llm.extend(bundle.context_messages)

            # 3a.2 Reduced chat history
            messages_for_llm.extend(built.history_messages or [])
            history_includes_current_user = True

            debug_trace["history_length"] = len(built.history_messages or [])
            debug_trace["rag_chunks"] = len(built.retrieved_chunks or [])

            # 3a.3 Tools context from RAG (compact textual form)
            rag_context_text = self._format_rag_context(built.retrieved_chunks or [])
            if rag_context_text:
                tools_context_parts.append("RAG CONTEXT:\n" + rag_context_text)

        else:
            # 3b. Fallback: legacy behavior without ContextBuilder / RAG
            history = self._build_chat_history(session)
            messages_for_llm = history
            history_includes_current_user = True
            debug_trace["history_length"] = len(history)
            used_rag_flag = False

        # 3c. Optional: Web search enrichment layer
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
                    language=None,  # (Optional) No language binding at runtime level for now.
                    top_n_fetch=None,
                    serialize=True,  # Ensure results are returned as Python dictionaries.
                )

                if web_docs:
                    used_websearch_flag = True

                    # Delegate formatting of web search results to WebSearchPromptBuilder
                    bundle = self._websearch_prompt_builder.build_websearch_prompt(
                        web_docs
                    )

                    # Inject web search context messages AFTER RAG/system/history,
                    # but BEFORE the final user message.
                    messages_for_llm.extend(bundle.context_messages)

                    # Merge debug info from builder (num_docs, top_urls, etc.)
                    if bundle.debug_info:
                        websearch_debug.update(bundle.debug_info)

                    # Tools context from websearch (compact textual form)
                    web_context_texts: List[str] = []
                    for msg in bundle.context_messages:
                        if getattr(msg, "content", None):
                            web_context_texts.append(msg.content)
                    if web_context_texts:
                        tools_context_parts.append(
                            "WEBSEARCH CONTEXT:\n" + "\n\n".join(web_context_texts)
                        )

            except Exception as exc:
                # Capture errors so the runtime doesn't break if the web search provider fails.
                websearch_debug["error"] = str(exc)

        # Store debugging info in the trace object only if websearch was attempted or produced metadata.
        if websearch_debug:
            debug_trace["websearch"] = websearch_debug

        # 3d. Current user message (always last in the base context)
        if not history_includes_current_user:
            messages_for_llm.append(ChatMessage(role="user", content=request.message))

        # 4. Optional tools execution layer (planning + tool calls).
        # The tools agent decides whether to call tools and which ones, but the final
        # answer is always produced by the main LLM over the unified context.
        use_tools = (
            self._config.tools_agent is not None
            and self._config.tools_mode != "off"
        )

        tools_agent_answer: Optional[str] = None

        if use_tools:
            # Construct textual context for tools agent from RAG + websearch layers.
            tools_context = (
                "\n\n".join(tools_context_parts).strip()
                if tools_context_parts
                else None
            )
            try:
                tools_result = self._config.tools_agent.run(
                    input_data=request.message,
                    context=tools_context,
                    stream=False,
                    tool_choice=None,  # Let the tools agent / underlying LLM decide.
                    output_model=None,
                )
                tools_agent_answer = tools_result.get("answer", "") or None
                tool_traces = tools_result.get("tool_traces") or []
                used_tools_flag = bool(tool_traces)

                debug_tools: Dict[str, Any] = {
                    "mode": self._config.tools_mode,
                    "used_tools": used_tools_flag,
                    "tool_traces": tool_traces,
                }
                if tools_agent_answer:
                    debug_tools["agent_answer_preview"] = str(tools_agent_answer)[:200]
                if self._config.tools_mode == "required" and not used_tools_flag:
                    debug_tools[
                        "warning"
                    ] = "tools_mode='required' but no tools were invoked by the tools_agent."
                debug_trace["tools"] = debug_tools

                # Inject tool call results into the LLM context as a structured system message,
                # so the main LLM can use them when composing the final answer.
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

                        tool_lines.append("")  # separator between tool calls

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
                debug_trace["tools_error"] = str(e)

        # 5. Final call to the main LLM with all enriched context
        # try:
        #     answer_text = self._config.llm_adapter.generate_messages(
        #         messages_for_llm,
        #         max_tokens=self._config.max_output_tokens,
        #         # temperature can be passed here later once you add it to RuntimeConfig
        #     )
        # except Exception as e:
        #     # In case of any adapter failure, optionally fall back to tools_agent answer.
        #     if tools_agent_answer:
        #         answer_text = (
        #             "[ERROR] LLM adapter failed, falling back to tools agent answer.\n"
        #             f"Details: {e}\n\n"
        #             f"{tools_agent_answer}"
        #         )
        #     else:
        #         answer_text = f"[ERROR] LLM adapter failed: {e}"
                # 5. Final call to the main LLM with all enriched context
                # 5. Decide how to produce the final answer text.
        #    If tools were used and the tools agent already produced a
        #    natural-language answer, we treat it as the final answer.
        #    Otherwise we call the main LLM with the full enriched context.
        answer_text: str

        if used_tools_flag and tools_agent_answer:
            # Tools agent (planner + executor + answer composer) already
            # produced a final answer. We do NOT call the main LLM again
            # to avoid overriding it with an empty or redundant response.
            answer_text = str(tools_agent_answer)
        else:
            # Fallback: no tools were used or tools agent did not produce
            # an answer. Use the main LLM with the enriched context.
            try:
                raw_answer = self._config.llm_adapter.generate_messages(
                    messages_for_llm,
                    max_tokens=self._config.max_output_tokens,
                )

                # Normalize various adapter return types to a plain string.
                if isinstance(raw_answer, str):
                    answer_text = raw_answer
                else:
                    content = getattr(raw_answer, "content", None)
                    if isinstance(content, str) and content.strip():
                        answer_text = content
                    else:
                        # Last resort: string representation of the object.
                        answer_text = str(raw_answer)

            except Exception as e:
                if tools_agent_answer:
                    answer_text = (
                        "[ERROR] LLM adapter failed, falling back to tools agent answer.\n"
                        f"Details: {e}\n\n"
                        f"{tools_agent_answer}"
                    )
                else:
                    answer_text = f"[ERROR] LLM adapter failed: {e}"

        # Final safety: make sure we store a non-empty string.
        if not isinstance(answer_text, str) or not answer_text.strip():
            answer_text = (
                str(tools_agent_answer)
                if tools_agent_answer
                else "[ERROR] Empty answer from runtime."
            )



        # 6. Append assistant message to the session
        assistant_session_msg = SessionMessage(
            id=str(uuid.uuid4()),
            role="assistant",  # MessageRole alias; type-compatible with the rest of the framework
            content=answer_text,
            created_at=datetime.now(timezone.utc),
            attachments=[],
            metadata={"placeholder": False},
        )
        await self._session_store.append_message(session.id, assistant_session_msg)

        # 7. Build RouteInfo and RuntimeStats
        # Strategy depending on RAG / websearch / tools usage
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

        # Prepare lightweight tool call summaries for the answer (for UI / logging).
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

        # 8. Return RuntimeAnswer
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
