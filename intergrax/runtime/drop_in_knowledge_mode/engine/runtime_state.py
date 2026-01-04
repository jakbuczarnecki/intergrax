# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.llm_usage_track import LLMUsageTracker
from intergrax.runtime.drop_in_knowledge_mode.context.context_builder import BuiltContext
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_context import LLMUsageRunRecord, RuntimeContext
from intergrax.runtime.drop_in_knowledge_mode.ingestion.ingestion_service import IngestionResult
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.drop_in_knowledge_mode.session.chat_session import ChatSession


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

    # Context
    context: RuntimeContext

    # Input
    request: RuntimeRequest

    run_id: str

    llm_usage_tracker: LLMUsageTracker

    # Session and ingestion
    session: Optional[ChatSession] = None
    ingestion_results: List[IngestionResult] = field(default_factory=list)
    used_attachments_context: bool = False

    # Conversation / context
    base_history: List[ChatMessage] = field(default_factory=list)
    messages_for_llm: List[ChatMessage] = field(default_factory=list)
    tools_context_parts: List[str] = field(default_factory=list)
    built_history_messages: List[ChatMessage] = field(default_factory=list)
    history_includes_current_user: bool = False

    # ContextBuilder intermediate result (history + retrieved chunks)
    context_builder_result: Optional[BuiltContext] = None

    # Long-term memory retrieval intermediate result (retrieved entries + context messages)
    user_longterm_memory_result: Optional[Any] = None

    # Profile-based instruction fragments prepared by the memory layer
    profile_user_instructions: Optional[str] = None
    profile_org_instructions: Optional[str] = None

    # Usage flags
    used_rag: bool = False
    used_websearch: bool = False
    used_tools: bool = False
    used_user_profile: bool = False
    used_user_longterm_memory: bool = False

    # Tools
    tools_agent_answer: Optional[str] = None
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)

    # Debug / diagnostics
    debug_trace: Dict[str, Any] = field(default_factory=dict)
    websearch_debug: Dict[str, Any] = field(default_factory=dict)

    # Token accounting (filled in _step_build_base_history)
    history_token_count: Optional[int] = None

    # Reasoning flags
    cap_rag_available: bool = False
    cap_user_ltm_available: bool = False
    cap_attachments_available: bool = False
    cap_websearch_available: bool = False
    cap_tools_available: bool = False


    # --- Core output (pipeline contract) ---
    # Filled by CoreLLM step
    raw_answer: Optional[str] = None

    # Filled by Persist step (final runtime output)
    runtime_answer: Optional[RuntimeAnswer] = None


    def trace_event(
        self,
        *,
        component: str,
        step: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        level: str = "info",
    ) -> None:
        """
        Append-only event log, stored under debug_trace["events"].
        """
        evt = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "component": component,
            "step": step,
            "message": message,
            "data": data or {},
        }
        self.debug_trace.setdefault("events", []).append(evt)


    def set_debug_section(self, key: str, value: Dict[str, Any]) -> None:
        """
        Structured debug snapshot (non-event) for a given step.
        """
        self.debug_trace[key] = value


    def set_debug_value(self, key: str, value: Any) -> None:
        self.debug_trace[key] = value


    def configure_llm_tracker(self) -> None:     

        if self.llm_usage_tracker is None:
            return
           
        self.llm_usage_tracker.register_adapter(self.context.config.llm_adapter, label="core_adapter")

        ta = self.context.config.tools_agent
        if ta is not None and ta.llm is not None:
            self.llm_usage_tracker.register_adapter(ta.llm, label="tools_agent")

        ws = self.context.config.websearch_config
        if ws is not None and ws.llm is not None:
            if ws.llm.map_adapter is not None:
                self.llm_usage_tracker.register_adapter(ws.llm.map_adapter, label="web_map_adapter")
            if ws.llm.reduce_adapter is not None:
                self.llm_usage_tracker.register_adapter(ws.llm.reduce_adapter, label="web_reduce_adapter")
            if ws.llm.rerank_adapter is not None:
                self.llm_usage_tracker.register_adapter(ws.llm.rerank_adapter, label="web_rerank_adapter")

    
    async def finalize_llm_tracker(self, request: RuntimeRequest, runtime_answer: RuntimeAnswer) -> None:
        if self.llm_usage_tracker is not None:
            runtime_answer.llm_usage_report = self.llm_usage_tracker.build_report()
            llm_usage_snapshot = runtime_answer.llm_usage_report.to_dict()
            self.set_debug_value("llm_usage", llm_usage_snapshot)

            if self.context.config.enable_llm_usage_collection and runtime_answer.llm_usage_report is not None:
                async with self.context.llm_usage_lock:
                    self.context.llm_usage_run_seq += 1
                    rec = LLMUsageRunRecord(
                        seq=self.context.llm_usage_run_seq,
                        ts_utc=datetime.now(timezone.utc),
                        run_id=self.run_id,
                        session_id=request.session_id,
                        user_id=request.user_id,
                        report=runtime_answer.llm_usage_report,
                    )
                    self.context.llm_usage_runs.append(rec)