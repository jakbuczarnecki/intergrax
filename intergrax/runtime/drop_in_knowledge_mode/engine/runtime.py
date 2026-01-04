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
import uuid

from intergrax.llm_adapters.llm_usage_track import LLMUsageTracker
from intergrax.runtime.drop_in_knowledge_mode.config import StepPlanningStrategy, ToolsContextScope
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_context import LLMUsageRunRecord, RuntimeContext
from intergrax.runtime.drop_in_knowledge_mode.ingestion.ingestion_service import IngestionResult
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import (
    RuntimeRequest,
    RuntimeAnswer,
    RouteInfo,
    RuntimeStats,
    ToolCallInfo,
)
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState

from intergrax.llm.messages import ChatMessage

from intergrax.runtime.drop_in_knowledge_mode.context.context_builder import (
    RetrievedChunk,
)


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
        context: RuntimeContext
    ) -> None:
        self.context = context


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, request: RuntimeRequest) -> RuntimeAnswer:
        """
        Main async entrypoint for the runtime.
        """

        run_id = f"run_{uuid.uuid4().hex}"

        state = RuntimeState(
            context=self.context,
            request=request,
            run_id=run_id,
            llm_usage_tracker=LLMUsageTracker(run_id=run_id)
        )        
        
        state.configure_llm_tracker()

        # Initial trace entry for this request.
        state.trace_event(
            component="engine",
            step="run_start",
            message="DropInKnowledgeRuntime.run() called.",
            data={
                "session_id": request.session_id,
                "user_id": request.user_id,
                "tenant_id": request.tenant_id or self.context.config.tenant_id,
                "run_id": state.run_id,
                "step_planning_strategy": str(self.context.config.step_planning_strategy),
            },
        )

        if self.context.config.step_planning_strategy == StepPlanningStrategy.OFF:
            runtime_answer = await self._run_pipeline_no_planner(state=state)
        
        elif self.context.config.step_planning_strategy == StepPlanningStrategy.STATIC_PLAN:
            runtime_answer = await self._run_pipeline_static_plan(state)
        
        elif self.context.config.step_planning_strategy == StepPlanningStrategy.DYNAMIC_LOOP:
            runtime_answer = await self._run_pipeline_dynamic_loop(state)

        else:
            raise ValueError(f"Unknown step_planning_strategy: {self.context.config.step_planning_strategy}")


        # Final trace entry for this request.
        state.trace_event(
            component="engine",
            step="run_end",
            message="DropInKnowledgeRuntime.run() finished.",
            data={
                "strategy": runtime_answer.route.strategy,
                "used_rag": runtime_answer.route.used_rag,
                "used_websearch": runtime_answer.route.used_websearch,
                "used_tools": runtime_answer.route.used_tools,
                "used_user_longterm_memory": runtime_answer.route.used_user_longterm_memory,
                "run_id":state.run_id
            },
        )
        
        await state.finalize_llm_tracker(request, runtime_answer)

        return runtime_answer


    async def _run_pipeline_no_planner(self, state: RuntimeState) -> RuntimeAnswer:
        # Session + ingestion
        await self._step_session_and_ingest(state)

        # Memory layer (user/org)
        await self._step_memory_layer(state)

        # Build base history (load & preprocess)
        await self._step_build_base_history(state)

        # History layer (ContextBuilder / raw)
        await self._step_history(state)

        # Instructions layer (final system prompt)
        await self._step_instructions(state)

        # RAG
        await self._step_rag(state)

        # User long-term memory
        await self._step_user_longterm_memory(state)

        # Session attachments context (retrieval from ingestion vectorstore)
        await self._step_ingested_attachments_context(state)

        # Web search
        await self._step_websearch(state)

        # Ensure current user message
        await self._ensure_current_user_message(state)

        # Tools
        await self._step_tools(state)

        # Core LLM
        await self._step_core_llm(state)

        # Persist + RuntimeAnswer
        await self._step_persist_and_build_answer(state)

        runtime_answer = state.runtime_answer
        if runtime_answer is None:
            raise RuntimeError("Persist step did not set state.runtime_answer.")
        return runtime_answer


    async def _run_pipeline_static_plan(self, state: RuntimeState) -> RuntimeAnswer:
        raise NotImplementedError(
            "StepPlanningStrategy.STATIC_PLAN is configured, but step planner is not implemented in this session."
        )

    async def _run_pipeline_dynamic_loop(self, state: RuntimeState) -> RuntimeAnswer:
        raise NotImplementedError(
            "StepPlanningStrategy.DYNAMIC_LOOP is configured, but step planner is not implemented in this session."
        )



    def run_sync(self, request: RuntimeRequest) -> RuntimeAnswer:
        """
        Synchronous wrapper around `run()`.

        Useful for environments where `await` is not easily available,
        such as simple scripts or some notebook setups.
        """
        return asyncio.run(self.run(request))
    

    # ------------------------------------------------------------------
    # Step: session + ingestion (no history)
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
        session = await self.context.session_manager.get_session(req.session_id)
        if session is None:
            session = await self.context.session_manager.create_session(
                session_id=req.session_id,
                user_id=req.user_id,
                tenant_id=req.tenant_id or self.context.config.tenant_id,
                workspace_id=req.workspace_id or self.context.config.workspace_id,
                metadata=req.metadata,
            )

        # 1a. Ingest attachments into vector store (if any)
        ingestion_results: List[IngestionResult] = []
        if req.attachments:
            if self.context.ingestion_service is None:
                raise ValueError(
                    "Attachments were provided but ingestion_service is not configured. "
                    "Pass ingestion_service explicitly to control where attachments are indexed."
                )
    
            ingestion_results = await self.context.ingestion_service.ingest_attachments_for_session(
                attachments=req.attachments,
                session_id=session.id,
                user_id=req.user_id,
                tenant_id=session.tenant_id,
                workspace_id=session.workspace_id,
            )

        # 2. Append user message to session history
        user_message = self._build_session_message_from_request(req)
        await self.context.session_manager.append_message(session.id, user_message)

        # Reload the session to ensure we have the latest metadata
        session = await self.context.session_manager.get_session(session.id) or session

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
        state.debug_trace = debug_trace
        # NOTE: state.base_history is intentionally left empty here.

    # ------------------------------------------------------------------
    # Step: memory layer (user/org context + profile instructions)
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

        cfg = self.context.config

        user_instr: Optional[str] = None
        org_instr: Optional[str] = None

        # 1) User profile memory (optional)
        if cfg.enable_user_profile_memory:
            user_instr_candidate = await self.context.session_manager.get_user_profile_instructions_for_session(
                session=session
            )
            if isinstance(user_instr_candidate, str):
                stripped = user_instr_candidate.strip()
                if stripped:
                    user_instr = stripped
                    state.used_user_profile = True

        # 2) Organization profile memory (optional)
        if cfg.enable_org_profile_memory:
            org_instr_candidate = await self.context.session_manager.get_org_profile_instructions_for_session(
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


        # 4) Debug info
        state.set_debug_section("memory_layer", {
            "implemented": True,
            "has_user_profile_instructions": bool(user_instr),
            "has_org_profile_instructions": bool(org_instr),
            "enable_user_profile_memory": cfg.enable_user_profile_memory,
            "enable_org_profile_memory": cfg.enable_org_profile_memory,
        })

        # Trace memory layer step.
        state.trace_event(
            component="engine",
            step="memory_layer",
            message="Profile-based instructions loaded for session.",
            data={
                "has_user_profile_instructions": bool(user_instr),
                "has_org_profile_instructions": bool(org_instr),
                "enable_user_profile_memory": cfg.enable_user_profile_memory,
                "enable_org_profile_memory": cfg.enable_org_profile_memory,
            },
        )

    # ------------------------------------------------------------------
    # Step: build base history (load & preprocess)
    # ------------------------------------------------------------------

    async def _step_build_base_history(self, state: RuntimeState) -> None:
        await self.context.history_layer.build_base_history(state)

    # ------------------------------------------------------------------
    # Step: history
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

        if self.context.context_builder is not None:
            # Delegate history shaping (truncation, system message stitching, etc.)
            # to ContextBuilder, but do NOT inject RAG here.
            built = await self.context.context_builder.build_context(
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
            state.set_debug_value("history_length", len(history_messages))
        else:
            # Fall back to using the base_history as-is (no additional
            # history layer beyond what ContextBuilder already produced).
            state.messages_for_llm.extend(base_history)
            state.built_history_messages = base_history
            state.history_includes_current_user = True
            state.set_debug_value("history_length", len(base_history))

        # Trace history building step.
        state.trace_event(
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
    # Step: RAG
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
        state.set_debug_value("rag_chunks", 0)

        # If RAG is globally disabled, do nothing.
        if not self.context.config.enable_rag:
            return
        
    
        if self.context.context_builder is None:
            raise RuntimeError("RAG enabled but ContextBuilder is not configured.")

        built = state.context_builder_result

        # In normal flow _step_history should have already called build_context
        # and stored the result. As a fallback, we can call it here.
        if built is None:
            session = state.session
            assert session is not None, "Session must be set before RAG step."
            built = await self.context.context_builder.build_context(
                session=session,
                request=state.request,
                base_history=state.base_history,
            )
            state.context_builder_result = built

        rag_info = built.rag_debug_info or {}
        state.set_debug_value("rag", rag_info)

        retrieved_chunks = built.retrieved_chunks or []
        state.used_rag = bool(rag_info.get("used", bool(retrieved_chunks)))

        if not state.used_rag:
            state.set_debug_value("rag_chunks", 0)
            return

        # RAG-specific prompt construction (context messages only)
        bundle = self.context.rag_prompt_builder.build_rag_prompt(built)

        # Additional context messages built from retrieved chunks
        context_messages = bundle.context_messages or []
        self._insert_context_before_last_user(state, context_messages)

        # Compact textual form of RAG context for tools agent
        rag_context_text = self._format_rag_context(retrieved_chunks)
        if rag_context_text:
            state.tools_context_parts.append("RAG CONTEXT:\n" + rag_context_text)

        state.set_debug_value("rag_chunks", len(retrieved_chunks))

        # Trace RAG step.
        state.trace_event(
            component="engine",
            step="rag",
            message="RAG step executed.",
            data={
                "rag_enabled": self.context.config.enable_rag,
                "used_rag": state.used_rag,
                "retrieved_chunks": len(retrieved_chunks),
            },
        )

    # ------------------------------------------------------------------
    # Step: Web search
    # ------------------------------------------------------------------

    async def _step_websearch(self, state: RuntimeState) -> None:
        """
        Run optional web search and inject results as context messages.
        """
        state.websearch_debug = {}
        state.used_websearch = False

        if (
            not self.context.config.enable_websearch
            or self.context.websearch_executor is None
            or self.context.websearch_prompt_builder is None
        ):
            return

        try:
            web_results = await self.context.websearch_executor.search_async(
                query=state.request.message,
                top_k=self.context.config.max_docs_per_query,
                language=None,
                top_n_fetch=None,
            )

            state.set_debug_section("websearch", {})
            dbg = state.debug_trace["websearch"]

            raw_preview = []
            for d in (web_results or [])[:5]:
                raw_preview.append(
                    {
                        "type": type(d).__name__,
                        "title": d.title,
                        "url": d.url,
                        "snippet_len": len(d.snippet or ""),
                        "text_len": len(d.text or ""),
                    }
                )

            dbg["raw_results_preview"] = raw_preview

            if not web_results:
                return

            state.used_websearch = True

            bundle = await self.context.websearch_prompt_builder.build_websearch_prompt(
                web_results=web_results,
                user_query=state.request.message,
                run_id=state.run_id,
            )

            context_messages = bundle.context_messages or []
            self._insert_context_before_last_user(state, context_messages)
            state.websearch_debug.update(bundle.debug_info or {})

            # Debug trace (no tools coupling)
            web_context_texts: List[str] = []
            for msg in context_messages:
                if msg.content:
                    web_context_texts.append(msg.content)

            dbg["context_blocks_count"] = len(web_context_texts)

            # Preview only to avoid bloating trace
            preview = "\n\n".join(web_context_texts[:1])
            dbg["context_preview"] = preview
            dbg["context_preview_chars"] = len(preview)

            # Guardrail signal: did websearch produce any grounded evidence?
            dbg["no_evidence"] = "No answer-relevant evidence extracted" in (preview or "")
            state.websearch_debug["no_evidence"] = dbg["no_evidence"]

            # Optional: doc-level preview (titles/urls only)
            dbg["docs_preview"] = [
                {
                    "title": d.title,
                    "url": d.url,
                }
                for d in (web_results or [])[:5]
            ]

        except Exception as exc:
            state.websearch_debug["error"] = str(exc)

        # Trace web search step.
        state.trace_event(
            component="engine",
            step="websearch",
            message="Web search step executed.",
            data={
                "websearch_enabled": self.context.config.enable_websearch,
                "used_websearch": state.used_websearch,
                "has_error": "error" in (state.websearch_debug or {}),
                "no_evidence": state.websearch_debug.get("no_evidence", False),
            },
        )

        

    # ------------------------------------------------------------------
    # Step: Ensure current user message
    # ------------------------------------------------------------------

    async def _ensure_current_user_message(self, state: RuntimeState) -> None:
        msg = (state.request.message or "").strip()
        if not msg:
            return

        if not state.messages_for_llm:
            state.messages_for_llm.append(ChatMessage(role="user", content=msg))
            return

        last = state.messages_for_llm[-1]
        last_content = (last.content or "").strip()

        # If the last message already equals the current user prompt, do nothing.
        if last.role == "user" and last_content == msg:
            return

        # Otherwise append current user prompt to enforce user-last semantics.
        state.messages_for_llm.append(ChatMessage(role="user", content=msg))


    # ------------------------------------------------------------------
    # Step: Tools
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
            self.context.config.tools_agent is not None
            and self.context.config.tools_mode != "off"
        )
        if not use_tools:
            return

        tools_context = (
            "\n\n".join(state.tools_context_parts).strip()
            if state.tools_context_parts
            else None
        )

        debug_tools: Dict[str, Any] = {
            "mode": self.context.config.tools_mode,
        }

        try:
            # Decide what to pass as input_data for the tools agent.
            if self.context.config.tools_context_scope == ToolsContextScope.CURRENT_MESSAGE_ONLY:
                agent_input = state.request.message

            elif self.context.config.tools_context_scope == ToolsContextScope.CONVERSATION:
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

            tools_result = self.context.config.tools_agent.run(
                input_data=agent_input,
                context=tools_context,
                stream=False,
                tool_choice=None,
                output_model=None,
                run_id=state.run_id,
                llm_usage_tracker = state.llm_usage_tracker
            )

            state.tools_agent_answer = tools_result.get("answer", "") or None
            state.tool_traces = tools_result.get("tool_traces") or []
            state.used_tools = bool(state.tool_traces)

            debug_tools["used_tools"] = state.used_tools
            debug_tools["tool_traces"] = state.tool_traces
            if state.tools_agent_answer:
                debug_tools["agent_answer_preview"] = str(state.tools_agent_answer)[:200]
            if self.context.config.tools_mode == "required" and not state.used_tools:
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
                    insert_at = len(state.messages_for_llm) - 1
                    state.messages_for_llm.insert(
                        insert_at,
                        ChatMessage(
                            role="system",
                            content=(
                                "The following tool calls have been executed. "
                                "Use their results when answering the user.\n\n"
                                + tools_context_for_llm
                            ),
                        ),
                    )

        except Exception as e:
            debug_tools["tools_error"] = str(e)

        state.set_debug_section("tools",  debug_tools)

        # Trace tools step.
        state.trace_event(
            component="engine",
            step="tools",
            message="Tools agent step executed.",
            data={
                "tools_mode": self.context.config.tools_mode,
                "used_tools": state.used_tools,
                "tool_traces_count": len(state.tool_traces),
            },
        )

    # ------------------------------------------------------------------
    # Step: Core LLM
    # ------------------------------------------------------------------

    async def _step_core_llm(self, state: RuntimeState) -> None:
        """
        Call the core LLM adapter and decide on the final answer text,
        possibly falling back to tools_agent_answer when needed.
        """
        # If tools were used and we have an explicit agent answer, prefer it.
        if state.used_tools and state.tools_agent_answer:
            # Trace the fact that we are reusing the tools agent answer
            # instead of calling the core LLM adapter.
            state.trace_event(
                component="engine",
                step="core_llm",
                message="Using tools_agent_answer as the final answer.",
                data={
                    "used_tools_answer": True,
                    "has_tools_agent_answer": True,
                },
            )
            state.raw_answer =  str(state.tools_agent_answer)
            return

        try:
            # Determine the per-request max output tokens, if any.
            max_output_tokens = state.request.max_output_tokens

            generate_kwargs: Dict[str, Any] = {}
            if max_output_tokens is not None:
                # Pass a max_tokens hint to the adapter. If the adapter ignores
                # it or uses a different keyword, that should be handled inside
                # the adapter implementation.
                generate_kwargs["max_tokens"] = max_output_tokens

            msgs = state.messages_for_llm
            if not msgs or msgs[-1].role != "user":
                raise Exception(
                    f"Last message must be 'user' (got: {msgs[-1].role if msgs else 'None'})."
                )

            raw_answer = self.context.config.llm_adapter.generate_messages(
                state.messages_for_llm,
                run_id=state.run_id,
                **generate_kwargs,
            )

            state.trace_event(
                component="engine",
                step="core_llm",
                message="Core LLM adapter returned answer.",
                data={
                    "used_tools_answer": False,
                    "adapter_return_type": "str",
                    "answer_len": len(raw_answer),
                    "answer_is_empty": not bool(raw_answer),
                },
            )
            
            state.raw_answer = raw_answer

        except Exception as e:
            state.set_debug_value("llm_error", str(e))

            # Trace the error and whether a tools_agent_answer fallback is available.
            state.trace_event(
                component="engine",
                step="core_llm_error",
                message="Core LLM adapter failed; falling back if possible.",
                data={
                    "error": str(e),
                    "has_tools_agent_answer": bool(state.tools_agent_answer),
                },
            )

            if state.tools_agent_answer:
                state.raw_answer = (
                    "[ERROR] LLM adapter failed, falling back to tools agent answer.\n"
                    f"Details: {e}\n\n"
                    f"{state.tools_agent_answer}"
                )
                return

            state.raw_answer = f"[ERROR] LLM adapter failed: {e}"

    # ------------------------------------------------------------------
    # Step: Persist answer & build RuntimeAnswer
    # ------------------------------------------------------------------

    async def _step_persist_and_build_answer(
        self,
        state: RuntimeState,
    ) -> None:
        """
        Append assistant message to the session and build a RuntimeAnswer,
        including RouteInfo and RuntimeStats.
        """

        answer_text = state.raw_answer

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
        await self.context.session_manager.append_message(session.id, assistant_message)

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
        elif state.used_attachments_context:
            strategy = "llm_with_session_attachments"
        elif state.ingestion_results:
            strategy = "llm_only_with_ingestion"
        else:
            strategy = "llm_only"

        route_info = RouteInfo(
            used_rag=state.used_rag and self.context.config.enable_rag,
            used_websearch=state.used_websearch and self.context.config.enable_websearch,
            used_tools=state.used_tools and self.context.config.tools_mode != "off",
            used_user_profile=state.used_user_profile,
            used_user_longterm_memory=state.used_user_longterm_memory and self.context.config.enable_user_longterm_memory,
            strategy=strategy,
            extra={
                "used_attachments_context": bool(state.used_attachments_context),
                "attachments_chunks": int(state.debug_trace.get("attachments_chunks", 0) or 0),
            },
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
        state.trace_event(
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

        state.runtime_answer = RuntimeAnswer(
            answer=answer_text,
            citations=[],
            route=route_info,
            tool_calls=tool_calls_for_answer,
            stats=stats,
            raw_model_output=None,
            debug_trace=state.debug_trace,
        )

    # ------------------------------------------------------------------
    # Step: instructions (final system prompt)
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
        
        system_message = ChatMessage(role="system", content=instructions_text)

        # `messages_for_llm` at this point should contain only history
        # (built by `_step_history`). We now prepend the system message.
        state.messages_for_llm = [system_message] + state.messages_for_llm


    # ------------------------------------------------------------------
    # Step: LongTerm memory RAG
    # ------------------------------------------------------------------
    async def _step_user_longterm_memory(self, state: RuntimeState) -> None:
        state.used_user_longterm_memory = False
        state.set_debug_value("user_longterm_memory_hits", 0)

        # Create a single debug section early and only mutate it later.
        state.set_debug_section("user_longterm_memory", {
            "enabled": bool(self.context.config.enable_user_longterm_memory),
            "used": False,
            "reason": None,
            "hits": 0,
            "retrieval_debug": {},
            "context_blocks_count": 0,
            "context_preview": "",
            "context_preview_chars": 0,
            "hits_preview": [],
        })
        dbg = state.debug_trace["user_longterm_memory"]

        if not self.context.config.enable_user_longterm_memory:
            dbg["reason"] = "disabled"
            return

        built = state.user_longterm_memory_result

        if built is None:
            session = state.session
            assert session is not None, "Session must be set before user long-term memory step."

            query = (state.request.message or "").strip()
            if not query:
                dbg["reason"] = "empty_query"
                dbg["used"] = False
                dbg["hits"] = 0
                state.set_debug_value("user_longterm_memory_hits", 0)
                return

            built = await self.context.session_manager.search_user_longterm_memory(
                user_id=session.user_id,
                query=query,
                top_k=self.context.config.max_longterm_entries_per_query,
                score_threshold=self.context.config.longterm_score_threshold,
            )
            state.user_longterm_memory_result = built

        built = built or {}
        ltm_info = built.get("debug") or {}
        hits = built.get("hits") or []

        # Keep retrieval debug inside the same section
        dbg["retrieval_debug"] = ltm_info

        state.used_user_longterm_memory = bool(ltm_info.get("used", bool(hits)))
        dbg["used"] = state.used_user_longterm_memory
        dbg["hits"] = len(hits)

        if not state.used_user_longterm_memory:
            dbg["reason"] = ltm_info.get("reason") or "no_hits_or_not_used"
            state.set_debug_value("user_longterm_memory_hits", 0)
            return

        bundle = self.context.user_longterm_memory_prompt_builder.build_user_longterm_memory_prompt(hits)
        context_messages = bundle.context_messages or []
        self._insert_context_before_last_user(state, context_messages)

        # Debug trace (no tools coupling)
        ltm_context_texts: List[str] = []
        for msg in context_messages:
            if msg.content:
                ltm_context_texts.append(msg.content)

        dbg["context_blocks_count"] = len(ltm_context_texts)

        preview = "\n\n".join(ltm_context_texts[:2])  # first 1-2 blocks is enough
        dbg["context_preview"] = preview
        dbg["context_preview_chars"] = len(preview)

        dbg["hits_preview"] = [
            {
                "entry_id": h.entry_id,
                "title": getattr(h, "title", None),
                "kind": getattr(getattr(h, "kind", None), "value", getattr(h, "kind", None)),
                "deleted": bool(getattr(h, "deleted", False)),
            }
            for h in hits[:5]
        ]

        state.set_debug_value("user_longterm_memory_hits", len(hits))

        state.trace_event(
            component="engine",
            step="user_longterm_memory",
            message="User long-term memory step executed.",
            data={
                "ltm_enabled": self.context.config.enable_user_longterm_memory,
                "used_user_longterm_memory": state.used_user_longterm_memory,
                "hits": len(hits),
            },
        )



    # ------------------------------------------------------------------
    # Step: session attachments context (retrieval from ingestion store)
    # ------------------------------------------------------------------

    async def _step_ingested_attachments_context(self, state: RuntimeState) -> None:
        """
        Retrieve relevant chunks from session-ingested attachments (AttachmentIngestionService)
        and inject them into the LLM context.

        Key requirements:
          - Independent from enable_rag (must work even when enable_rag=False).
          - Reuse existing retrieval components (EmbeddingManager + VectorstoreManager.query).
          - Filter by session_id + user_id (+ tenant/workspace if available).
          - Inject as context messages using _insert_context_before_last_user.
        """
        # Defaults
        state.used_attachments_context = False
        state.set_debug_value("attachments_chunks", 0)

        if self.context.ingestion_service is None:
            state.set_debug_section("attachments", {"used": False, "reason": "ingestion_service_not_configured"})
            return

        session = state.session
        if session is None:
            state.set_debug_section("attachments", {"used": False, "reason": "session_not_initialized"})
            return

        # Retrieval (no coupling to enable_rag)
        res = await self.context.ingestion_service.search_session_attachments(
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
        attachments_context_text = self._format_rag_context(chunks)
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

        self._insert_context_before_last_user(state, context_messages)

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
            state.set_debug_section("instructions", {
                "has_instructions": False,
                "sources": sources,
            })
            return None

        # Simple concatenation for now; can be replaced with more structured
        # formatting (sections, headings) in the future.
        final_text = "\n\n".join(parts)

        state.set_debug_section("instructions", {
            "has_instructions": True,
            "sources": sources,
        })

        return final_text
    

    def _format_longterm_memory_context(self, entries: List[Any]) -> str:
        """
        Format user long-term memory entries into a compact, prompt-ready text block.
        Keeps engine independent from retrieval details (no embeddings/vectorstore here).
        """
        lines: List[str] = []
        for e in entries:
            entry_id = getattr(e, "entry_id", None)
            content = (getattr(e, "content", "") or "").strip()
            if not content:
                continue
            prefix = f"[LTM:{entry_id}] " if entry_id is not None else "[LTM] "
            lines.append(prefix + content)

        return "\n".join(lines).strip()
    

    def _insert_context_before_last_user(self, state: RuntimeState, msgs: List[ChatMessage]) -> None:
        if not msgs:
            return

        # find last user message in current assembled prompt
        last_user_idx = None
        for i in range(len(state.messages_for_llm) - 1, -1, -1):
            if state.messages_for_llm[i].role == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            # no user in prompt yet -> append (rare, but safe)
            state.messages_for_llm.extend(msgs)
            return

        for m in msgs:
            state.messages_for_llm.insert(last_user_idx, m)
            last_user_idx += 1   


    

