# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.llm_usage_track import LLMUsageTracker
from intergrax.runtime.nexus.context.context_builder import BuiltContext
from intergrax.runtime.nexus.engine.runtime_context import LLMUsageRunRecord, RuntimeContext
from intergrax.runtime.nexus.ingestion.ingestion_service import IngestionResult
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.tracing.adapters.llm_usage_finalize import LLMUsageFinalizeDiag
from intergrax.runtime.nexus.tracing.adapters.llm_usage_snapshot import LLMUsageSnapshotDiag
from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload, ToolCallTrace, TraceComponent, TraceEvent, TraceLevel, utc_now_iso
from intergrax.utils.time_provider import SystemTimeProvider


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
      - full trace for observability & diagnostics.
    """

    # Context
    context: RuntimeContext

    # Input
    request: RuntimeRequest

    run_id: str

    started_at_utc: str = SystemTimeProvider.utc_now().isoformat()

    llm_usage_tracker: Optional[LLMUsageTracker] = None

    # Session and ingestion
    session: Optional[ChatSession] = None
    ingestion_results: List[IngestionResult] = field(default_factory=list)
    used_attachments_context: bool = False
    attachments_chunks_count: int = 0

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

    # Typed tool call traces (production runtime artifact).
    tool_traces: List[ToolCallTrace] = field(default_factory=list)


    # Production trace (append-only structured events)
    trace_events: List[TraceEvent] = field(default_factory=list)
    _trace_seq: int = field(default=0, init=False, repr=False)

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
        component: TraceComponent,
        step: str,
        message: str,
        level: TraceLevel = TraceLevel.INFO,
        payload: Optional[DiagnosticPayload] = None,
    ) -> None:
        if not self.run_id:
            raise RuntimeError("RuntimeState.run_id must be provided (got empty).")

        if payload is not None and not isinstance(payload, DiagnosticPayload):
            raise TypeError(f"payload must be DiagnosticPayload (got {type(payload).__name__}).")

        self._trace_seq += 1

        evt = TraceEvent(
            event_id=TraceEvent.new_id(),
            run_id=self.run_id,
            seq=self._trace_seq,
            ts_utc=utc_now_iso(),
            level=level,
            component=component,
            step=step,
            message=message,
            payload=payload,
            tags={},
        )
        self.trace_events.append(evt)

        writer = self.context.trace_writer
        if writer is not None:
            writer.append_event(evt)


    def configure_llm_tracker(self) -> None:     

        if self.llm_usage_tracker is None:
           self.llm_usage_tracker = LLMUsageTracker(run_id=self.run_id)            
           
        self.llm_usage_tracker.register_adapter(self.context.config.llm_adapter, label="core_adapter")

        tool_agent = self.context.config.tools_agent
        if tool_agent is not None and tool_agent.llm is not None:
            self.llm_usage_tracker.register_adapter(tool_agent.llm, label="tools_agent")

        websearch_config = self.context.config.websearch_config
        if websearch_config is not None and websearch_config.llm is not None:
            if websearch_config.llm.map_adapter is not None:
                self.llm_usage_tracker.register_adapter(websearch_config.llm.map_adapter, label="web_map_adapter")
            if websearch_config.llm.reduce_adapter is not None:
                self.llm_usage_tracker.register_adapter(websearch_config.llm.reduce_adapter, label="web_reduce_adapter")
            if websearch_config.llm.rerank_adapter is not None:
                self.llm_usage_tracker.register_adapter(websearch_config.llm.rerank_adapter, label="web_rerank_adapter")

    
    async def finalize_llm_tracker(
        self,
        request: RuntimeRequest,
        runtime_answer: RuntimeAnswer | None,
    ) -> None:
        if self.llm_usage_tracker is None:
            return

        report = self.llm_usage_tracker.build_report()
        total = report.total  # LLMRunStats

        llm_usage_snapshot = LLMUsageSnapshotDiag(
            run_id=report.run_id,
            calls=total.calls,
            input_tokens=total.input_tokens,
            output_tokens=total.output_tokens,
            total_tokens=total.total_tokens,
            duration_ms=total.duration_ms,
            errors=total.errors,
            adapters_registered=len(report.entries),
            provider_model_groups=len(report.by_provider_model),
        )

        # Always persist snapshot into structured trace, even if run aborted.
        self.trace_event(
            component=TraceComponent.ENGINE,
            step="llm_usage_snapshot",
            level=TraceLevel.INFO,
            message="LLM usage snapshot captured.",
            payload=llm_usage_snapshot,
        )

        # If the run aborted before producing RuntimeAnswer, do not raise and do not collect runs.
        if runtime_answer is None:
            self.trace_event(
                component=TraceComponent.ENGINE,
                step="llm_usage_finalize",
                level=TraceLevel.WARNING,
                message="LLM usage finalized without RuntimeAnswer (run aborted).",
                payload=LLMUsageFinalizeDiag(
                    run_id=self.run_id,
                    session_id=request.session_id,
                    user_id=request.user_id,
                    aborted=True,
                ),
            )
            return

        # Attach report to the answer for API consumers.
        runtime_answer.llm_usage_report = report

        # Optional: store run record for analytics/monitoring
        if self.context.config.enable_llm_usage_collection and runtime_answer.llm_usage_report is not None:
            async with self.context.llm_usage_lock:
                self.context.llm_usage_run_seq += 1
                rec = LLMUsageRunRecord(
                    seq=self.context.llm_usage_run_seq,
                    ts_utc=SystemTimeProvider.utc_now(),
                    run_id=self.run_id,
                    session_id=request.session_id,
                    user_id=request.user_id,
                    report=runtime_answer.llm_usage_report,
                )
                self.context.llm_usage_runs.append(rec)                


