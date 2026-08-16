# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional
from typing import TYPE_CHECKING

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.tracking.llm_usage_track import LLMUsageTracker
from intergrax.contracts.runtime_cost import tokens_to_cost_units
from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.runtime.nexus.engine.contracts.agent_state import AgentState
from intergrax.runtime.nexus.engine.contracts.llm_usage_run_record import LLMUsageRunRecord
from intergrax.runtime.nexus.engine.contracts.runtime_state_contract import RuntimeStateContract

if TYPE_CHECKING:
    from intergrax.runtime.nexus.artifacts.models import ArtifactRef
    from intergrax.runtime.observability.emitter import ObservabilityEmitter

from intergrax.runtime.nexus.context.context_builder import BuiltContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.ingestion.ingestion_service import IngestionResult
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.tracing.adapters.llm_usage_finalize import LLMUsageFinalizeDiag
from intergrax.runtime.nexus.tracing.adapters.llm_usage_snapshot import LLMUsageSnapshotDiag
from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload, ToolCallTrace, TraceComponent, TraceEvent, TraceLevel
from intergrax.utils.time_provider import SystemTimeProvider


@dataclass
class RuntimeState(RuntimeStateContract):
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

    # Run
    run_id: str

    # Typed declarative HITL grant mirror (transport only; scope is per ToolExecutionRequest).
    declarative_hitl_grant: DeclarativeHitlApprovalGrant | None = None

    # Utc
    started_at_utc: str = field(
        default_factory=lambda: SystemTimeProvider.utc_now().isoformat()
    )

    # --- Agent domain state (Tier-2) ---
    agent_state: Optional[AgentState] = None

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
    tool_planner_answer: Optional[str] = None
    tool_planner_allowed_tool_ids: Optional[tuple[str, ...]] = None

    # Typed tool call traces (production runtime artifact).
    tool_traces: List[ToolCallTrace] = field(default_factory=list)
    high_risk_tool_approvals: frozenset[str] = field(default_factory=frozenset)

    # RunBudget mid-run enforcement (RagStep / WebsearchStep entry counts).
    rag_step_invocation_count: int = 0
    websearch_step_invocation_count: int = 0

    # --- Execution Artifacts (runtime infra output) ---
    artifacts: List["ArtifactRef"] = field(default_factory=list)

    # Production trace (append-only structured events)
    trace_events: List[TraceEvent] = field(default_factory=list)
    _trace_seq: int = field(default=0, init=False, repr=False)
    _observability_emitter: Optional["ObservabilityEmitter"] = field(
        default=None, init=False, repr=False
    )

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
    last_llm_adapter_response: Optional[LLMAdapterResponse] = None

    # Filled by Persist step (final runtime output)
    runtime_answer: Optional[RuntimeAnswer] = None

    # Tenant
    @property
    def tenant_id(self) -> str:
        tenant = self.request.tenant_id
        if not tenant:
            raise RuntimeError(
                "tenant_id must be set in RuntimeRequest before RuntimeState is used."
            )
        return tenant

    @property
    def task_id(self) -> str:
        """Typed task identity for policy/HITL matching."""
        typed = self.request.task_id
        if typed is not None and str(typed).strip():
            return str(typed)
        raise RuntimeError("RuntimeRequest.task_id must be set before RuntimeState is used.")


    def _next_trace_seq(self) -> int:
        self._trace_seq += 1
        return self._trace_seq

    def _get_observability_emitter(self) -> ObservabilityEmitter:
        if self._observability_emitter is None:
            from intergrax.contracts.execution_identity import mint_attempt_id, peek_active_execution_identity
            from intergrax.runtime.observability.emitter import ObservabilityEmitter

            active = peek_active_execution_identity()
            attempt_id = active[1] if active is not None else mint_attempt_id()
            self._observability_emitter = ObservabilityEmitter(
                run_id=self.run_id,
                task_id=self.task_id,
                tenant_id=self.tenant_id,
                agent_id=self.request.agent_id or "",
                attempt_id=str(attempt_id),
                trace_writer=self.context.trace_writer,
                event_bus=self.context.config.runtime_event_bus,
                trace_events=self.trace_events,
                production_mode=self.context.config.production_mode,
                next_seq=self._next_trace_seq,
            )
        return self._observability_emitter

    def trace_event(
        self,
        *,
        component: TraceComponent,
        step: str,
        message: str,
        level: TraceLevel = TraceLevel.INFO,
        payload: Optional[DiagnosticPayload] = None,
        artifact_refs: Optional[List["ArtifactRef"]] = None,
    ) -> None:
        self._get_observability_emitter().emit_diagnostic(
            component=component,
            step=step,
            message=message,
            level=level,
            payload=payload,
            artifact_refs=artifact_refs,
        )


    def configure_llm_tracker(self) -> None:

        if self.llm_usage_tracker is None:
           self.llm_usage_tracker = LLMUsageTracker(run_id=self.run_id)

        from intergrax.runtime.nexus.tracing.adapters.model_catalog_miss import (
            wire_catalog_miss_trace_sink,
        )

        wire_catalog_miss_trace_sink(self.trace_event, run_id=self.run_id)

        core_adapter = self.context.config.llm_adapter
        if core_adapter is not None:
            from intergrax.runtime.wiring.llm_resolver import consume_routing_evaluation
            from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
            from intergrax.llm_adapters.routing.evaluating_hooks import wire_routing_evaluating_hooks
            from intergrax.llm_adapters.routing.metering import resolve_metering_adapter
            from intergrax.runtime.nexus.tracing.adapters.llm_routing_attempt import (
                attach_failover_routing_trace_observer,
                emit_llm_routing_allowlist_violation_diag,
                emit_llm_routing_rule_diag,
            )

            def _on_evaluated(evaluation: object) -> None:
                from intergrax.llm_adapters.routing.contracts import RoutingEvaluation

                assert isinstance(evaluation, RoutingEvaluation)
                emit_llm_routing_rule_diag(self.trace_event, evaluation)

            def _on_allowlist_violation(exc: object, context: object) -> None:
                from intergrax.llm_adapters.routing.contracts import RoutingContext
                from intergrax.llm_adapters.routing.evaluator import AllowlistViolationError

                assert isinstance(exc, AllowlistViolationError)
                assert isinstance(context, RoutingContext)
                emit_llm_routing_allowlist_violation_diag(self.trace_event, exc, context)

            def _on_inner_swapped(inner: LLMAdapter) -> None:
                self.llm_usage_tracker.register_adapter(
                    inner,
                    label=f"core_inner_{id(inner)}",
                )
                attach_failover_routing_trace_observer(inner, self.trace_event)

            meter_target = resolve_metering_adapter(core_adapter) or core_adapter

            wire_routing_evaluating_hooks(
                core_adapter,
                on_evaluated=_on_evaluated,
                on_allowlist_violation=_on_allowlist_violation,
                on_inner_swapped=_on_inner_swapped,
                attach_failover_observer=lambda adapter: attach_failover_routing_trace_observer(
                    adapter,
                    self.trace_event,
                ),
            )

            routing_evaluation = consume_routing_evaluation()
            if routing_evaluation is not None:
                emit_llm_routing_rule_diag(self.trace_event, routing_evaluation)
            self.llm_usage_tracker.register_adapter(meter_target, label="core_adapter")

        from intergrax.runtime.nexus.tools.tool_planner_trackable import ToolPlannerTrackable

        tool_planner = self.context.config.tool_planner
        if isinstance(tool_planner, ToolPlannerTrackable):
            self.llm_usage_tracker.register_adapter(
                tool_planner.llm,
                label="tool_planner",
            )

        websearch_config = self.context.config.websearch_config
        if websearch_config is not None and websearch_config.llm is not None:
            if websearch_config.llm.map_adapter is not None:
                self.llm_usage_tracker.register_adapter(websearch_config.llm.map_adapter, label="web_map_adapter")
            if websearch_config.llm.reduce_adapter is not None:
                self.llm_usage_tracker.register_adapter(websearch_config.llm.reduce_adapter, label="web_reduce_adapter")
            if websearch_config.llm.rerank_adapter is not None:
                self.llm_usage_tracker.register_adapter(websearch_config.llm.rerank_adapter, label="web_rerank_adapter")

    

    def add_artifact(self, artifact: "ArtifactRef") -> None:
        """
        Register execution artifact reference in runtime state.

        RuntimeState stores only references, never payloads.
        """
        self.artifacts.append(artifact)



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
        runtime_answer.stats.total_tokens = total.total_tokens
        runtime_answer.stats.input_tokens = total.input_tokens
        runtime_answer.stats.output_tokens = total.output_tokens
        runtime_answer.stats.duration_ms = total.duration_ms
        runtime_answer.stats.extra["cost"] = tokens_to_cost_units(total.total_tokens)

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


