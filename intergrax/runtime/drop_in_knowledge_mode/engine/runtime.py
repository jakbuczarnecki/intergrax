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
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from intergrax.runtime.drop_in_knowledge_mode.config import RuntimeConfig, ToolsContextScope
from intergrax.runtime.drop_in_knowledge_mode.context.engine_history_layer import HistoryLayer
from intergrax.runtime.drop_in_knowledge_mode.ingestion.ingestion_service import AttachmentIngestionService, IngestionResult
from intergrax.runtime.drop_in_knowledge_mode.prompts.history_prompt_builder import (
    DefaultHistorySummaryPromptBuilder,
    HistorySummaryPromptBuilder,
)
from intergrax.runtime.drop_in_knowledge_mode.prompts.rag_prompt_builder import (
    DefaultRagPromptBuilder,
    RagPromptBuilder,
)
from intergrax.runtime.drop_in_knowledge_mode.reasoning.reasoning_layer import ReasoningLayer
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import (
    RuntimeRequest,
    RuntimeAnswer,
    RouteInfo,
    RuntimeStats,
    ToolCallInfo,
)
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState

from intergrax.llm.messages import ChatMessage

from intergrax.runtime.drop_in_knowledge_mode.ingestion.attachments import (
    FileSystemAttachmentResolver,
)
from intergrax.runtime.drop_in_knowledge_mode.context.context_builder import (
    ContextBuilder,
    RetrievedChunk,
)
from intergrax.runtime.drop_in_knowledge_mode.prompts.websearch_prompt_builder import (
    DefaultWebSearchPromptBuilder,
    WebSearchPromptBuilder,
)
from intergrax.runtime.drop_in_knowledge_mode.session.session_manager import SessionManager
from intergrax.websearch.service.websearch_executor import WebSearchExecutor


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
      - Load or create a ChatSession via SessionManager.
      - Append the user message to the session.
      - Build an LLM-ready context:
          * system prompt(s),
          * chat history from SessionManager,
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
        session_manager: SessionManager,
        ingestion_service: Optional[AttachmentIngestionService] = None,
        context_builder: Optional[ContextBuilder] = None,
        rag_prompt_builder: Optional[RagPromptBuilder] = None,
        websearch_prompt_builder: Optional[WebSearchPromptBuilder] = None,
        history_prompt_builder: Optional[HistorySummaryPromptBuilder] = None,
    ) -> None:

        self._config = config
        self._session_manager = session_manager

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

        self._history_prompt_builder: HistorySummaryPromptBuilder = (
            history_prompt_builder or DefaultHistorySummaryPromptBuilder(config)
        )

        self._history_layer = HistoryLayer(
            config=config,
            session_manager=session_manager,
            history_prompt_builder=self._history_prompt_builder,
        )

        self._reasoning_layer = ReasoningLayer(
            config=config
        )        

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, request: RuntimeRequest) -> RuntimeAnswer:
        """
        Main async entrypoint for the runtime.

        Pipeline:
          1. Session + ingestion + user message appended.
          2. Memory layer (user/org profile memory, long-term memory facts).
          3. Base history builder (load & preprocess conversation history).
          4. History layer (conversation history for the LLM).
          5. Instructions layer (final system prompt).
          6. RAG layer (retrieval + RAG system/context messages).
          7. Web search layer (optional).
          8. Ensure current user message is present at the end of context.
          9. Tools layer (planning + tool calls).
         10. Core LLM call.
         11. Persist assistant answer and build RuntimeAnswer with route info.
        """
        state = RuntimeState(request=request)

        # Initial trace entry for this request.
        self._trace(
            state,
            component="engine",
            step="run_start",
            message="DropInKnowledgeRuntime.run() called.",
            data={
                "session_id": request.session_id,
                "user_id": request.user_id,
                "tenant_id": request.tenant_id or self._config.tenant_id,
            },
        )

        # 1. Session + ingestion
        await self._step_session_and_ingest(state)

        # 2. Memory layer (user/org)
        await self._step_memory_layer(state)

        # 3. Build base history (load & preprocess)
        await self._step_build_base_history(state)

        # 4. History layer (ContextBuilder / raw)
        await self._step_history(state)

        # 5. Instructions layer (final system prompt)
        await self._step_instructions(state)

        # 6. RAG
        await self._step_rag(state)

        # 7. Web search
        await self._step_websearch(state)

        # 8. Ensure current user message
        self._ensure_current_user_message(state)

        # 9. Tools
        await self._step_tools(state)

        # 10. Core LLM
        answer_text = self._step_core_llm(state)

        # 11. Persist + RuntimeAnswer
        runtime_answer = await self._step_persist_and_build_answer(state, answer_text)

        # Final trace entry for this request.
        self._trace(
            state,
            component="engine",
            step="run_end",
            message="DropInKnowledgeRuntime.run() finished.",
            data={
                "strategy": runtime_answer.route.strategy,
                "used_rag": runtime_answer.route.used_rag,
                "used_websearch": runtime_answer.route.used_websearch,
                "used_tools": runtime_answer.route.used_tools,
            },
        )

        return runtime_answer

    def run_sync(self, request: RuntimeRequest) -> RuntimeAnswer:
        """
        Synchronous wrapper around `run()`.

        Useful for environments where `await` is not easily available,
        such as simple scripts or some notebook setups.
        """
        return asyncio.run(self.run(request))

    def _trace(
        self,
        state: RuntimeState,
        *,
        component: str,
        step: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Append a single, structured diagnostic entry to the state's debug_trace.

        This helper:
          - ensures a consistent schema for all pipeline steps,
          - does not introspect objects (no getattr / reflection),
          - relies only on data explicitly provided by the caller.
        """
        if data is None:
            data = {}

        trace = state.debug_trace
        steps = trace.setdefault("steps", [])

        steps.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "component": component,
                "step": step,
                "message": message,
                "data": data,
            }
        )

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
        session = await self._session_manager.get_session(req.session_id)
        if session is None:
            session = await self._session_manager.create_session(
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
        await self._session_manager.append_message(session.id, user_message)

        # Reload the session to ensure we have the latest metadata
        session = await self._session_manager.get_session(session.id) or session

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

        # Trace session and ingestion step.
        self._trace(
            state,
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
        state.debug_trace = debug_trace
        # NOTE: state.base_history is intentionally left empty here.

    # ------------------------------------------------------------------
    # Step 2: memory layer (user/org context + profile instructions)
    # ------------------------------------------------------------------

    async def _step_memory_layer(self, state: RuntimeState) -> None:
        """
        Load profile-based instruction fragments for this request.

        Rules:
          - Use profile memory only if enabled in RuntimeConfig.
          - Do NOT rebuild or cache anything here yet (this is step 1 only).
          - Extract prebuilt 'system_prompt' strings from profile bundles.
          - Store the resulting fragments in RuntimeState so the engine
            can merge them into a system message later.
        """
        session = state.session
        assert session is not None, "Session must exist before memory layer."

        cfg = self._config

        user_instr: Optional[str] = None
        org_instr: Optional[str] = None

        # 1) User profile memory (optional)
        if cfg.enable_user_profile_memory:
            user_instr_candidate = await self._session_manager.get_user_profile_instructions_for_session(
                session=session
            )
            if isinstance(user_instr_candidate, str):
                stripped = user_instr_candidate.strip()
                if stripped:
                    user_instr = stripped
                    state.used_user_profile = True

        # 2) Organization profile memory (optional)
        if cfg.enable_org_profile_memory:
            org_instr_candidate = await self._session_manager.get_org_profile_instructions_for_session(
                session=session
            )
            if isinstance(org_instr_candidate, str):
                stripped = org_instr_candidate.strip()
                if stripped:
                    org_instr = stripped
                    # For now we reuse the same flag to indicate that some profile
                    # (user or organization) has been used.
                    state.used_user_profile = True

        # 3) Store extracted profile instruction fragments in state
        state.profile_user_instructions = user_instr
        state.profile_org_instructions = org_instr

        # 4) Long-term memory hook (implemented in next steps)
        #    This step is a placeholder and intentionally does nothing now.
        if cfg.enable_long_term_memory:
            pass

        # 5) Debug info
        state.debug_trace["memory_layer"] = {
            "implemented": True,
            "has_user_profile_instructions": bool(user_instr),
            "has_org_profile_instructions": bool(org_instr),
            "enable_user_profile_memory": cfg.enable_user_profile_memory,
            "enable_org_profile_memory": cfg.enable_org_profile_memory,
            "enable_long_term_memory": cfg.enable_long_term_memory,
        }

        # Trace memory layer step.
        self._trace(
            state,
            component="engine",
            step="memory_layer",
            message="Profile-based instructions loaded for session.",
            data={
                "has_user_profile_instructions": bool(user_instr),
                "has_org_profile_instructions": bool(org_instr),
                "enable_user_profile_memory": cfg.enable_user_profile_memory,
                "enable_org_profile_memory": cfg.enable_org_profile_memory,
                "enable_long_term_memory": cfg.enable_long_term_memory,
            },
        )

    # ------------------------------------------------------------------
    # Step 3: build base history (load & preprocess)
    # ------------------------------------------------------------------

    async def _step_build_base_history(self, state: RuntimeState) -> None:
        await self._history_layer.build_base_history(state)

    # ------------------------------------------------------------------
    # Step 4: history
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
            # Fall back to using the base_history as-is (no additional
            # history layer beyond what ContextBuilder already produced).
            state.messages_for_llm.extend(base_history)
            state.built_history_messages = base_history
            state.history_includes_current_user = True
            state.debug_trace["history_length"] = len(base_history)

        # Trace history building step.
        self._trace(
            state,
            component="engine",
            step="history",
            message="Conversation history built for LLM.",
            data={
                "history_length": len(state.built_history_messages),
                "base_history_length": len(state.base_history),
                "history_includes_current_user": state.history_includes_current_user,
            },
        )

    # ------------------------------------------------------------------
    # Step 5: RAG
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
        if not self._config.enable_rag:
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

        rag_info = built.rag_debug_info or {}
        state.debug_trace["rag"] = rag_info

        retrieved_chunks = built.retrieved_chunks or []
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

        # Trace RAG step.
        self._trace(
            state,
            component="engine",
            step="rag",
            message="RAG step executed.",
            data={
                "rag_enabled": self._config.enable_rag,
                "used_rag": state.used_rag,
                "retrieved_chunks": len(retrieved_chunks),
            },
        )

    # ------------------------------------------------------------------
    # Step 6: Web search
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
                if msg.content:
                    web_context_texts.append(msg.content)
            if web_context_texts:
                state.tools_context_parts.append(
                    "WEBSEARCH CONTEXT:\n" + "\n\n".join(web_context_texts)
                )

        except Exception as exc:
            state.websearch_debug["error"] = str(exc)

        if state.websearch_debug:
            state.debug_trace["websearch"] = state.websearch_debug

        # Trace web search step.
        self._trace(
            state,
            component="engine",
            step="websearch",
            message="Web search step executed.",
            data={
                "websearch_enabled": self._config.enable_websearch,
                "used_websearch": state.used_websearch,
                "has_error": "error" in (state.websearch_debug or {}),
            },
        )

    # ------------------------------------------------------------------
    # Step 7: Ensure current user message
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
    # Step 8: Tools
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

        # Trace tools step.
        self._trace(
            state,
            component="engine",
            step="tools",
            message="Tools agent step executed.",
            data={
                "tools_mode": self._config.tools_mode,
                "used_tools": state.used_tools,
                "tool_traces_count": len(state.tool_traces),
            },
        )

    # ------------------------------------------------------------------
    # Step 9: Core LLM
    # ------------------------------------------------------------------

    def _step_core_llm(self, state: RuntimeState) -> str:
        """
        Call the core LLM adapter and decide on the final answer text,
        possibly falling back to tools_agent_answer when needed.
        """
        # If tools were used and we have an explicit agent answer, prefer it.
        if state.used_tools and state.tools_agent_answer:
            # Trace the fact that we are reusing the tools agent answer
            # instead of calling the core LLM adapter.
            self._trace(
                state,
                component="engine",
                step="core_llm",
                message="Using tools_agent_answer as the final answer.",
                data={
                    "used_tools_answer": True,
                    "has_tools_agent_answer": True,
                },
            )
            return str(state.tools_agent_answer)

        try:
            # Determine the per-request max output tokens, if any.
            max_output_tokens = state.request.max_output_tokens

            generate_kwargs: Dict[str, Any] = {}
            if max_output_tokens is not None:
                # Pass a max_tokens hint to the adapter. If the adapter ignores
                # it or uses a different keyword, that should be handled inside
                # the adapter implementation.
                generate_kwargs["max_tokens"] = max_output_tokens

            raw_answer = self._config.llm_adapter.generate_messages(
                state.messages_for_llm,
                **generate_kwargs,
            )

            # LLM adapter may return a simple string.
            if isinstance(raw_answer, str):
                self._trace(
                    state,
                    component="engine",
                    step="core_llm",
                    message="Core LLM adapter returned a plain string.",
                    data={
                        "used_tools_answer": False,
                        "adapter_return_type": "str",
                    },
                )
                return raw_answer

            # Or an object with a .content attribute (dataclass / pydantic model).
            if hasattr(raw_answer, "content"):
                content_value = raw_answer.content  # type: ignore[attr-defined]
                if isinstance(content_value, str) and content_value.strip():
                    self._trace(
                        state,
                        component="engine",
                        step="core_llm",
                        message="Core LLM adapter returned an object with non-empty content.",
                        data={
                            "used_tools_answer": False,
                            "adapter_return_type": type(raw_answer).__name__,
                            "has_content_attribute": True,
                        },
                    )
                    return content_value

            # Fallback: stringify whatever the adapter returned.
            self._trace(
                state,
                component="engine",
                step="core_llm",
                message="Core LLM adapter returned a non-string object without usable content; using stringified value.",
                data={
                    "used_tools_answer": False,
                    "adapter_return_type": type(raw_answer).__name__,
                    "has_content_attribute": hasattr(raw_answer, "content"),
                },
            )
            return str(raw_answer)

        except Exception as e:
            state.debug_trace["llm_error"] = str(e)

            # Trace the error and whether a tools_agent_answer fallback is available.
            self._trace(
                state,
                component="engine",
                step="core_llm_error",
                message="Core LLM adapter failed; falling back if possible.",
                data={
                    "error": str(e),
                    "has_tools_agent_answer": bool(state.tools_agent_answer),
                },
            )

            if state.tools_agent_answer:
                return (
                    "[ERROR] LLM adapter failed, falling back to tools agent answer.\n"
                    f"Details: {e}\n\n"
                    f"{state.tools_agent_answer}"
                )

            return f"[ERROR] LLM adapter failed: {e}"

    # ------------------------------------------------------------------
    # Step 10: Persist answer & build RuntimeAnswer
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
        await self._session_manager.append_message(session.id, assistant_message)

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

        # Trace persistence and answer building step.
        self._trace(
            state,
            component="engine",
            step="persist_and_build_answer",
            message="Assistant answer persisted and RuntimeAnswer built.",
            data={
                "session_id": session.id,
                "strategy": strategy,
                "used_rag": state.used_rag,
                "used_websearch": state.used_websearch,
                "used_tools": state.used_tools,
            },
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
    # Step 11: instructions (final system prompt)
    # ------------------------------------------------------------------

    async def _step_instructions(self, state: RuntimeState) -> None:
        """
        Inject the final instructions as the first `system` message in the
        LLM prompt, if any instructions exist.

        This uses `_build_final_instructions` to combine:
          - per-request instructions,
          - user profile instructions,
          - organization profile instructions.

        It MUST be called AFTER the history step, so that:
          - history can be freely trimmed/summarized,
          - instructions are always the first system message,
          - instructions are never persisted in SessionStore.
        """
        instructions_text = await self._build_final_instructions(state)
        if not instructions_text:
            return
        
        # Apply reasoning / CoT policy (no-op in DIRECT mode)
        reasoning_result = self._reasoning_layer.apply_reasoning_to_instructions(
            state=state,
            base_system_instructions=instructions_text,
        )

        # Trace reasoning application
        self._trace(
            state,
            component="reasoning",
            step="apply_reasoning_to_instructions",
            message="Reasoning policy applied to system instructions.",
            data={
                "mode": reasoning_result.mode.value,
                "applied": reasoning_result.applied,
            },
        )

        system_message = ChatMessage(role="system", content=instructions_text)

        # `messages_for_llm` at this point should contain only history
        # (built by `_step_history`). We now prepend the system message.
        state.messages_for_llm = [system_message] + state.messages_for_llm

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

    async def _build_final_instructions(self, state: RuntimeState) -> Optional[str]:
        """
        Build the final instructions string for the current request/session.

        Sources:
          1) User-provided instructions from RuntimeRequest (if any).
          2) Profile-based user instructions prepared by _step_memory_layer.
          3) Profile-based organization instructions prepared by _step_memory_layer.

        The result is a single, short, LLM-ready text that can be used
        as a `system` message at the top of the prompt.

        This method:
          - does NOT touch SessionStore,
          - does NOT modify history,
          - only consolidates instruction fragments already present in the state.
        """
        parts: List[str] = []
        sources = {
            "request": False,
            "user_profile": False,
            "organization_profile": False,
        }

        # 1) User-provided instructions (per-request, ChatGPT/Gemini-style)
        if isinstance(state.request.instructions, str):
            user_instr = state.request.instructions.strip()
            if user_instr:
                parts.append(user_instr)
                sources["request"] = True

        # 2) User profile instructions prepared by the memory layer
        if isinstance(state.profile_user_instructions, str):
            profile_user = state.profile_user_instructions.strip()
            if profile_user:
                parts.append(profile_user)
                sources["user_profile"] = True

        # 3) Organization profile instructions prepared by the memory layer
        if isinstance(state.profile_org_instructions, str):
            profile_org = state.profile_org_instructions.strip()
            if profile_org:
                parts.append(profile_org)
                sources["organization_profile"] = True

        if not parts:
            state.debug_trace["instructions"] = {
                "has_instructions": False,
                "sources": sources,
            }
            return None

        # Simple concatenation for now; can be replaced with more structured
        # formatting (sections, headings) in the future.
        final_text = "\n\n".join(parts)

        state.debug_trace["instructions"] = {
            "has_instructions": True,
            "sources": sources,
        }

        return final_text
