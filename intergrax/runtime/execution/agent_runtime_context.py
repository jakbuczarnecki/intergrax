# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import asyncio  # noqa: I001
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional  # noqa: F401, UP035
from typing import TYPE_CHECKING
import uuid
from intergrax.distributed.contracts.execution_semaphore import (
    DistributedExecutionSemaphore,
)
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.runtime.nexus.artifacts.models import Artifact, TraceArtifactRef
from intergrax.runtime.nexus.artifacts.store_base import ArtifactStore
from intergrax.runtime.nexus.engine.contracts.llm_usage_run_record import (
    LLMUsageRunRecord,
)
from intergrax.runtime.nexus.tools import RegistryToolExecutor
from intergrax.runtime.nexus.tools.runtime_tool_invoker_composition import (
    build_production_runtime_tool_invoker,
)
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.persistence.integration_profile_wiring import (
    open_trace_store_from_profile,
)
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent
from intergrax.runtime.replay.service import ReplayService
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.tools.registry.read import ToolRegistryRead

if TYPE_CHECKING:
    from intergrax.runtime.agent_governance.ports import AgentRuntimeGovernancePort  # noqa: F401, I001
    from intergrax.queueing.worker.registry import BackgroundTaskHandler, TaskExecutionRegistry  # noqa: F401
    from intergrax.runtime.agent_governance.audit import GovernanceAuditRecorder  # noqa: F401
    from intergrax.runtime.nexus.artifacts.governed_access import GovernedArtifactReadPort  # noqa: F401
    from intergrax.runtime.nexus.artifacts.retention import ArtifactRetentionSweeper  # noqa: F401
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState  # noqa: I001
    from intergrax.runtime.nexus.config import RuntimeConfig
    from intergrax.runtime.governance.governance_evidence_recorder import (
        GovernanceEvidenceRecorder,
    )
    from intergrax.runtime.governance.service import GovernanceService
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter  # noqa: I001
from intergrax.utils.time_provider import SystemTimeProvider
from intergrax.runtime.nexus.context.context_builder import ContextBuilder
from intergrax.runtime.nexus.context.engine_history_layer import HistoryLayer
from intergrax.runtime.nexus.ingestion.ingestion_service import (
    AttachmentIngestionService,
)
from intergrax.runtime.nexus.prompts.rag_prompt_builder import (
    DefaultRagPromptBuilder,
    RagPromptBuilder,
)
from intergrax.runtime.nexus.prompts.user_longterm_memory_prompt_builder import (
    DefaultUserLongTermMemoryPromptBuilder,
    UserLongTermMemoryPromptBuilder,
)
from intergrax.runtime.nexus.prompts.websearch_prompt_builder import (
    DefaultWebSearchPromptBuilder,
    WebSearchPromptBuilder,
)
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.websearch.service.websearch_executor import WebSearchExecutor


def _enrich_tool_wiring_context(wiring_ctx: ToolWiringContext, config: "RuntimeConfig") -> ToolWiringContext:
    """Fill catalog tool dependencies from RuntimeConfig when Tier-3 omitted explicit wiring."""
    from dataclasses import replace

    return replace(
        wiring_ctx,
        vectorstore_manager=wiring_ctx.vectorstore_manager or config.vectorstore_manager,
        embedding_manager=wiring_ctx.embedding_manager or config.embedding_manager,
        websearch_executor=wiring_ctx.websearch_executor or config.websearch_executor,
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

    replay_service: ReplayService | None = None

    ingestion_service: AttachmentIngestionService | None = None
    context_builder: ContextBuilder | None = None

    rag_prompt_builder: RagPromptBuilder | None = None
    user_longterm_memory_prompt_builder: UserLongTermMemoryPromptBuilder | None = None

    websearch_executor: WebSearchExecutor | None = None
    websearch_prompt_builder: WebSearchPromptBuilder | None = None

    history_layer: HistoryLayer | None = None

    llm_usage_run_seq: int = 0
    llm_usage_runs: list[LLMUsageRunRecord] = field(default_factory=list)
    llm_usage_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    trace_writer: RunTraceWriter | None = None
    governance_service: Optional["GovernanceService"] = None
    governance_evidence_recorder: Optional["GovernanceEvidenceRecorder"] = None
    prompt_registry: YamlPromptRegistry | None = None

    artifact_store: ArtifactStore | None = None

    execution_semaphore: DistributedExecutionSemaphore | None = None

    policy_bundle: RuntimePolicyBundle | None = None
    max_parallel_per_tenant: int | None = None
    tool_invoker_close_on_context_close: bool = True

    def close(self) -> None:
        """Release runtime-owned resources (tool execution pool)."""
        if not self.tool_invoker_close_on_context_close:
            return
        invoker = self.config.tool_invoker
        if invoker is not None:
            invoker.close()

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
    ) -> TraceArtifactRef:
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

        ref = TraceArtifactRef(
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
        ingestion_service: AttachmentIngestionService | None = None,
        context_builder: ContextBuilder | None = None,
        rag_prompt_builder: RagPromptBuilder | None = None,
        user_longterm_memory_prompt_builder: UserLongTermMemoryPromptBuilder | None = None,
        websearch_prompt_builder: WebSearchPromptBuilder | None = None,
        prompt_registry: YamlPromptRegistry | None = None,
        governance_service: Optional["GovernanceService"] = None,
        governance_evidence_recorder: Optional["GovernanceEvidenceRecorder"] = None,
    ) -> "RuntimeContext":
        """
        Build a fully-resolved RuntimeContext using the same resolution rules as Runtime.__init__:

        - config.validate()
        - context_builder defaults to ContextBuilder(...) when enable_rag and not provided
        - prompt builders default to their Default* implementations
        - websearch_executor resolved from config if enabled and provided
        - history_layer constructed as raw compatibility loader (OFF only)
        """
        config.validate()

        from intergrax.runtime.nexus.budget.production_budget_policy import (
            ensure_production_run_budget,
        )

        ensure_production_run_budget(config)

        if prompt_registry is None:
            from intergrax.prompts.registry.prompt_registry_resolver import (
                resolve_yaml_prompt_registry,
            )

            prompt_registry = resolve_yaml_prompt_registry(
                catalog_path=config.prompt_catalog_path,
            )

        # Resolve ContextBuilder (RAG)
        resolved_context_builder = context_builder
        if resolved_context_builder is None and config.enable_rag:
            resolved_context_builder = ContextBuilder(
                config=config,
                vectorstore_manager=config.vectorstore_manager,
            )

        # Resolve RAG prompt builder
        resolved_rag_prompt_builder: RagPromptBuilder = rag_prompt_builder or DefaultRagPromptBuilder(
            config=config,
            prompt_registry=prompt_registry,
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
        resolved_websearch_executor: WebSearchExecutor | None = None
        if config.enable_websearch and config.websearch_executor:
            resolved_websearch_executor = config.websearch_executor

        # Resolve websearch prompt builder
        resolved_websearch_prompt_builder: WebSearchPromptBuilder | None = (
            websearch_prompt_builder or DefaultWebSearchPromptBuilder(config)
        )

        resolved_history_layer = HistoryLayer(
            config=config,
            session_manager=session_manager,
        )

        # --- Runtime Tools ---
        from intergrax.tools.registry.bootstrap import register_default_tools

        wiring_ctx = config.tool_wiring_context or ToolWiringContext()
        wiring_ctx = _enrich_tool_wiring_context(wiring_ctx, config)
        config.tool_wiring_context = wiring_ctx

        registry: ToolRegistryRead
        if config.tool_registry is not None:
            registry = config.tool_registry
        else:
            register_default_tools()
            mutable_registry = ToolRegistry()

            if config.tool_profile is not None:
                build_registry_from_profile(
                    config.tool_profile,
                    ctx=wiring_ctx,
                    registry=mutable_registry,
                )

            for provider in config.tool_providers:
                provider.register_tools(mutable_registry, wiring_ctx)

            registry = mutable_registry

        executor = RegistryToolExecutor(registry)
        pre_effect_coordinator: IdempotencyPreEffectCoordinator | None = None
        if config.idempotency_store is None:
            config.idempotency_store = InMemoryIdempotencyStore()

        if config.idempotency_store is not None:
            pre_effect_coordinator = IdempotencyPreEffectCoordinator(
                idempotency_store=config.idempotency_store,
            )

        from intergrax.runtime.sandbox.isolation_gate import (  # noqa: I001
            sandbox_availability_provider,
        )

        from intergrax.runtime.resilience.dependency_attempt_boundary_composition import (
            materialize_tool_dependency_attempt_boundary,
        )

        dependency_boundary = materialize_tool_dependency_attempt_boundary(
            config.dependency_concurrency_admission,
            production_mode=config.production_mode,
        )
        base_invoker = build_production_runtime_tool_invoker(
            registry=registry,
            executor=executor,
            scope_policy=config.tool_scope_policy,
            pre_effect_coordinator=pre_effect_coordinator,
            sandbox_availability=sandbox_availability_provider(wiring_ctx),
            agent_runtime_governance=config.agent_runtime_governance,
            inner_execution_guard=config.canonical_inner_execution_guard,
            meaningful_side_effect_authorization=config.meaningful_side_effect_authorization,
            dependency_attempt_boundary=dependency_boundary,
            production_mode=config.production_mode,
            governance_evidence_recorder=governance_evidence_recorder,
        )

        from intergrax.runtime.nexus.tools.planner_bootstrap import (
            wire_catalog_tool_planner_if_enabled,
        )

        wire_catalog_tool_planner_if_enabled(
            config,
            registry,
            prompt_registry=prompt_registry,
        )

        config.tool_invoker = base_invoker

        if config.production_mode and config.agent_runtime_governance is None:
            raise ValueError("agent_runtime_governance is required when production_mode=True.")
        if config.production_mode and governance_service is None:
            raise ValueError("GovernanceService is required when production_mode=True.")

        trace_writer: RunTraceWriter | None = None
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
            history_layer=resolved_history_layer,
            prompt_registry=prompt_registry,
            governance_service=governance_service,
            governance_evidence_recorder=governance_evidence_recorder,
            trace_writer=trace_writer,
        )


# --------------------------------------------------------------------------- D8 composition helpers
#
# ``runtime/execution`` is an execution-engine zone: the EBH-4 gate allows ``runtime.nexus`` imports only
# here and under ``runtime/nexus``. The persistence and worker composition roots therefore reach the
# D8 artifact components through these helpers instead of importing ``runtime.nexus`` themselves.


def open_sqlite_artifact_store(db_path: str) -> ArtifactStore:
    """Open the durable SQLite artifact store (PDC-D8-R003)."""
    from intergrax.runtime.nexus.artifacts.sqlite_artifact_store import SQLiteArtifactStore

    return SQLiteArtifactStore(db_path)


@dataclass(frozen=True)
class WorkerArtifactServices:
    """What a worker root composed for D8: the sweep owner, its handler and, when governed, the read port."""

    sweeper: "ArtifactRetentionSweeper"
    sweep_handler: "BackgroundTaskHandler"
    read_port: Optional["GovernedArtifactReadPort"]


def compose_worker_artifact_services(
    worker_registry: "TaskExecutionRegistry",
    *,
    artifact_store: ArtifactStore,
    governance: Optional["AgentRuntimeGovernancePort"] = None,
    audit: Optional["GovernanceAuditRecorder"] = None,
) -> WorkerArtifactServices:
    """Register the artifact retention sweep and build the governed read port (PDC-D8-R008, R014).

    Wiring of existing D8 components only: the sweep is a handler on the given
    ``TaskExecutionRegistry`` and the port reads through ``ExecutionArtifactSource``. The read
    port needs the governance boundary and its audit recorder together; without them no read
    surface is created.
    """
    from intergrax.contracts.artifact_ref import ArtifactClass
    from intergrax.runtime.nexus.artifacts.governed_access import GovernedArtifactReadPort
    from intergrax.runtime.nexus.artifacts.retention import (
        ArtifactRetentionSweeper,
        make_retention_sweep_handler,
        register_retention_sweep,
    )
    from intergrax.runtime.nexus.artifacts.sources import ExecutionArtifactSource

    if governance is not None and audit is None:
        raise ValueError("a governed artifact read port needs the governance audit recorder as well")
    sweeper = ArtifactRetentionSweeper(artifact_store, audit=audit)
    register_retention_sweep(worker_registry, sweeper)
    read_port: GovernedArtifactReadPort | None = None
    if governance is not None and audit is not None:
        source = ExecutionArtifactSource(artifact_store)
        read_port = GovernedArtifactReadPort(
            governance=governance,
            audit=audit,
            sources={ArtifactClass.EXECUTION_EVIDENCE: source, ArtifactClass.TOOL_OUTPUT: source},
        )
    # The same handler the registry now holds, for an in-process host that runs the sweep directly.
    return WorkerArtifactServices(
        sweeper=sweeper,
        sweep_handler=make_retention_sweep_handler(sweeper),
        read_port=read_port,
    )
