# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from typing import TYPE_CHECKING
import uuid
from intergrax.distributed.contracts.execution_semaphore import DistributedExecutionSemaphore
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.runtime.nexus.artifacts.models import Artifact, ArtifactRef
from intergrax.runtime.nexus.artifacts.store_base import ArtifactStore
from intergrax.runtime.nexus.engine.contracts.llm_usage_run_record import LLMUsageRunRecord
from intergrax.runtime.nexus.tools import RegistryToolExecutor
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.persistence.integration_profile_wiring import open_trace_store_from_profile
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent
from intergrax.runtime.replay.service import ReplayService
from intergrax.runtime.tools.idempotent_invoker import IdempotentToolInvoker
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.registry import ToolRegistry, ToolWiringContext, build_registry_from_profile
if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
    from intergrax.runtime.nexus.config import RuntimeConfig
    from intergrax.runtime.governance.service import GovernanceService
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter
from intergrax.utils.time_provider import SystemTimeProvider
from intergrax.runtime.nexus.context.context_builder import ContextBuilder
from intergrax.runtime.nexus.context.engine_history_layer import HistoryLayer
from intergrax.runtime.nexus.ingestion.ingestion_service import AttachmentIngestionService
from intergrax.runtime.nexus.prompts.history_prompt_builder import DefaultHistorySummaryPromptBuilder, HistorySummaryPromptBuilder
from intergrax.runtime.nexus.prompts.rag_prompt_builder import DefaultRagPromptBuilder, RagPromptBuilder
from intergrax.runtime.nexus.prompts.user_longterm_memory_prompt_builder import DefaultUserLongTermMemoryPromptBuilder, UserLongTermMemoryPromptBuilder
from intergrax.runtime.nexus.prompts.websearch_prompt_builder import DefaultWebSearchPromptBuilder, WebSearchPromptBuilder
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.websearch.service.websearch_executor import WebSearchExecutor


def _enrich_tool_wiring_context(wiring_ctx: ToolWiringContext, config: "RuntimeConfig") -> ToolWiringContext:
    """Fill catalog tool dependencies from RuntimeConfig when Tier-3 omitted explicit wiring."""
    return ToolWiringContext(
        issue_tracker=wiring_ctx.issue_tracker,
        search_provider=wiring_ctx.search_provider,
        wiki_knowledge=wiring_ctx.wiki_knowledge,
        notification_channel=wiring_ctx.notification_channel,
        observability_backend=wiring_ctx.observability_backend,
        rag_manager=wiring_ctx.rag_manager,
        vectorstore_manager=wiring_ctx.vectorstore_manager or config.vectorstore_manager,
        embedding_manager=wiring_ctx.embedding_manager or config.embedding_manager,
        websearch_executor=wiring_ctx.websearch_executor or config.websearch_executor,
        sandbox_session=wiring_ctx.sandbox_session,
        extras=dict(wiring_ctx.extras),
    )


@dataclass(frozen=False)
class RuntimeContext:
    """
    Per-runtime context: resolved dependencies + configuration.

    This object is intended to be:
    - configuration & dependencies are stable; diagnostics mutate
    - reusable in tests (build() can create the same defaults as Runtime.__init__)
    - passed to steps: step.run(state, ctx)

    IMPORTANT:
    - per-request flags/results belong to RuntimeState, not here.
    """
    
    config: "RuntimeConfig"
    session_manager: SessionManager

    replay_service: Optional[ReplayService] = None

    ingestion_service: Optional[AttachmentIngestionService] = None
    context_builder: Optional[ContextBuilder] = None

    rag_prompt_builder: Optional[RagPromptBuilder] = None
    user_longterm_memory_prompt_builder: Optional[UserLongTermMemoryPromptBuilder] = None

    websearch_executor: Optional[WebSearchExecutor] = None
    websearch_prompt_builder: Optional[WebSearchPromptBuilder] = None

    history_prompt_builder: Optional[HistorySummaryPromptBuilder] = None
    history_layer: Optional[HistoryLayer] = None

    llm_usage_run_seq: int = 0
    llm_usage_runs: List[LLMUsageRunRecord] = field(default_factory=list)
    llm_usage_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    trace_writer: Optional[RunTraceWriter] = None
    governance_service: Optional["GovernanceService"] = None
    prompt_registry: Optional[YamlPromptRegistry] = None
    
    artifact_store: Optional[ArtifactStore] = None

    execution_semaphore: Optional[DistributedExecutionSemaphore] = None

    policy_bundle: Optional[RuntimePolicyBundle] = None
    max_parallel_per_tenant: Optional[int] = None

    async def get_llm_usage_runs(self) -> list[LLMUsageRunRecord]:
        async with self.llm_usage_lock:
            return list(self.llm_usage_runs)
            

    async def clear_llm_usage_runs(self) -> None:
        async with self.llm_usage_lock:
            self.llm_usage_runs.clear()
            self.llm_usage_run_seq = 0 


    async def print_usage_runs(self):
        runs = await self.get_llm_usage_runs()
        print("runs:", len(runs))

        # Aggregate totals across all runs
        total_calls = 0
        total_in = 0
        total_out = 0
        total_tokens = 0
        total_ms = 0
        total_errors = 0

        # Aggregate by provider/model string key
        by_key = {}  # key -> dict(calls,in,out,total,ms,err)

        for r in runs:
            # r.total is expected to have these fields (as shown in pretty())
            t = r.report.total
            total_calls += int(t.calls or 0)
            total_in += int(t.input_tokens or 0)
            total_out += int(t.output_tokens or 0)
            total_tokens += int(t.total_tokens or 0)
            total_ms += int(t.duration_ms or 0)
            total_errors += int(t.errors or 0)

            # r.by_provider_model is expected to be iterable of items with key + stats
            # (use exactly what your report object exposes; below assumes dict-like)
            bpm = r.report.by_provider_model

            for k, st in bpm.items():
                agg = by_key.get(k)
                if agg is None:
                    agg = {"calls": 0, "in": 0, "out": 0, "total": 0, "ms": 0, "err": 0}
                    by_key[k] = agg
                agg["calls"] += int(st.calls or 0)
                agg["in"] += int(st.input_tokens or 0)
                agg["out"] += int(st.output_tokens or 0)
                agg["total"] += int(st.total_tokens or 0)
                agg["ms"] += int(st.duration_ms or 0)
                agg["err"] += int(st.errors or 0)

        if runs:
            print("=" * 100)
            print("ALL RUNS (aggregated)")
            print(f"  calls        : {total_calls}")
            print(f"  input_tokens : {total_in}")
            print(f"  output_tokens: {total_out}")
            print(f"  total_tokens : {total_tokens}")
            print(f"  duration_ms  : {total_ms}")
            print(f"  errors       : {total_errors}")

            if by_key:
                print("By provider/model (aggregated):")
                for k, st in by_key.items():
                    print(
                        f"  - {k}: calls={st['calls']} in={st['in']} out={st['out']} "
                        f"total={st['total']} ms={st['ms']} err={st['err']}"
                    )

        for r in runs:
            print("=" * 100)
            print(r.pretty())


    def create_artifact(
        self,
        *,
        state: "RuntimeState",
        kind: str,
        mime_type: str,
        data: bytes,
        step_id: str | None = None,
    ) -> ArtifactRef:
        """
        Runtime infrastructure gateway for execution artifacts.

        This is NOT agent logic.
        This is part of the execution substrate.
        """

        if self.artifact_store is None:
            raise RuntimeError("ArtifactStore not configured.")

        artifact_id = uuid.uuid4().hex

        artifact = Artifact(
            artifact_id=artifact_id,
            run_id=state.run_id,
            step_id=step_id,
            kind=kind,
            mime_type=mime_type,
            created_at_utc=SystemTimeProvider.utc_now(),
            data=data,
            size_bytes=len(data),
        )

        self.artifact_store.put(artifact)

        ref = ArtifactRef(
            artifact_id=artifact_id,
            kind=kind,
            size_bytes=len(data),
        )

        state.add_artifact(ref)

        state.trace_event(
            component=TraceComponent.ENGINE,
            step="artifact_created",
            message=f"Artifact created: {kind}",
            artifact_refs=[ref],
        )

        return ref



    @classmethod
    def build(
        cls,
        *,
        config: "RuntimeConfig",
        session_manager: SessionManager,
        ingestion_service: Optional[AttachmentIngestionService] = None,
        context_builder: Optional[ContextBuilder] = None,
        rag_prompt_builder: Optional[RagPromptBuilder] = None,
        user_longterm_memory_prompt_builder: Optional[UserLongTermMemoryPromptBuilder] = None,
        websearch_prompt_builder: Optional[WebSearchPromptBuilder] = None,
        history_prompt_builder: Optional[HistorySummaryPromptBuilder] = None,
        prompt_registry: Optional[YamlPromptRegistry] = None,
        governance_service: Optional["GovernanceService"] = None,
    ) -> "RuntimeContext":
        """
        Build a fully-resolved RuntimeContext using the same resolution rules as Runtime.__init__:

        - config.validate()
        - context_builder defaults to ContextBuilder(...) when enable_rag and not provided
        - prompt builders default to their Default* implementations
        - websearch_executor resolved from config if enabled and provided
        - history_layer constructed using resolved history_prompt_builder
        """
        config.validate()

        if prompt_registry is None:
            catalog_path = config.prompt_catalog_path
            if catalog_path is not None:
                prompt_registry = YamlPromptRegistry.create_default(
                    path=catalog_path,
                    load=True,
                )
            else:
                prompt_registry = YamlPromptRegistry.create_default(load=True)

        # Resolve ContextBuilder (RAG)
        resolved_context_builder = context_builder
        if resolved_context_builder is None and config.enable_rag:
            resolved_context_builder = ContextBuilder(
                config=config,
                vectorstore_manager=config.vectorstore_manager,
            )

        # Resolve RAG prompt builder
        resolved_rag_prompt_builder: RagPromptBuilder = (
            rag_prompt_builder or DefaultRagPromptBuilder(
                config=config,
                prompt_registry=prompt_registry,
            )
        )

        # Resolve user long-term memory prompt builder
        resolved_user_ltm_prompt_builder: UserLongTermMemoryPromptBuilder = (
            user_longterm_memory_prompt_builder
            or DefaultUserLongTermMemoryPromptBuilder(
                max_entries=config.max_longterm_entries_per_query,
                max_chars=int(config.max_longterm_tokens * 4),
                prompt_registry=prompt_registry,
            )
        )

        # Resolve websearch executor (from config)
        resolved_websearch_executor: Optional[WebSearchExecutor] = None
        if config.enable_websearch and config.websearch_executor:
            resolved_websearch_executor = config.websearch_executor

        # Resolve websearch prompt builder
        resolved_websearch_prompt_builder: Optional[WebSearchPromptBuilder] = (
            websearch_prompt_builder or DefaultWebSearchPromptBuilder(config)
        )

        # Resolve history prompt builder
        resolved_history_prompt_builder : HistorySummaryPromptBuilder = (
            history_prompt_builder or DefaultHistorySummaryPromptBuilder(
                config=config,
                prompt_registry=prompt_registry,
            )
        )
        

        # Build HistoryLayer using resolved builder
        resolved_history_layer = HistoryLayer(
            config=config,
            session_manager=session_manager,
            history_prompt_builder=resolved_history_prompt_builder,
        )

        # --- Runtime Tools ---
        from intergrax.tools.registry.bootstrap import register_default_tools

        register_default_tools()
        registry = ToolRegistry()

        wiring_ctx = config.tool_wiring_context or ToolWiringContext()
        wiring_ctx = _enrich_tool_wiring_context(wiring_ctx, config)
        config.tool_wiring_context = wiring_ctx

        if config.tool_profile is not None:
            build_registry_from_profile(
                config.tool_profile,
                ctx=wiring_ctx,
                registry=registry,
            )

        executor = RegistryToolExecutor(registry)
        base_invoker = RuntimeToolInvoker(registry=registry, executor=executor)

        if config.idempotency_store is None:
            config.idempotency_store = InMemoryIdempotencyStore()

        if config.idempotency_store is not None:
            config.tool_invoker = IdempotentToolInvoker(
                base_invoker=base_invoker,
                idempotency_store=config.idempotency_store,
            )
        else:
            config.tool_invoker = base_invoker


        # Register tools from providers
        for provider in config.tool_providers:
            provider.register_tools(registry, wiring_ctx)

        if config.production_mode and governance_service is None:
            raise ValueError(
                "GovernanceService is required when production_mode=True."
            )
        
        trace_writer: Optional[RunTraceWriter] = None
        if config.production_mode:
            if config.trace_db_path is None:
                raise ValueError("trace_db_path must be set in production_mode.")
            profile = config.integration_profile or IntegrationProfile.lab()
            trace_writer = open_trace_store_from_profile(
                profile,
                db_path=Path(config.trace_db_path),
            )  # type: ignore[assignment]

        if ingestion_service is not None and trace_writer is not None:
            ingestion_service.bind_trace_writer(trace_writer)
        
        return cls(
            config=config,
            session_manager=session_manager,
            policy_bundle=config.policy_bundle,
            ingestion_service=ingestion_service,
            context_builder=resolved_context_builder,
            rag_prompt_builder=resolved_rag_prompt_builder,
            user_longterm_memory_prompt_builder=resolved_user_ltm_prompt_builder,
            websearch_executor=resolved_websearch_executor,
            websearch_prompt_builder=resolved_websearch_prompt_builder,
            history_prompt_builder=resolved_history_prompt_builder,
            history_layer=resolved_history_layer,            
            prompt_registry=prompt_registry,
            governance_service=governance_service,
            trace_writer=trace_writer,
        )