# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from typing import TYPE_CHECKING
from intergrax.llm_adapters.llm_usage_track import LLMUsageReport
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter
if TYPE_CHECKING:
    from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_builder import ContextBuilder
from intergrax.runtime.nexus.context.engine_history_layer import HistoryLayer
from intergrax.runtime.nexus.ingestion.ingestion_service import AttachmentIngestionService
from intergrax.runtime.nexus.prompts.history_prompt_builder import DefaultHistorySummaryPromptBuilder, HistorySummaryPromptBuilder
from intergrax.runtime.nexus.prompts.rag_prompt_builder import DefaultRagPromptBuilder, RagPromptBuilder
from intergrax.runtime.nexus.prompts.user_longterm_memory_prompt_builder import DefaultUserLongTermMemoryPromptBuilder, UserLongTermMemoryPromptBuilder
from intergrax.runtime.nexus.prompts.websearch_prompt_builder import DefaultWebSearchPromptBuilder, WebSearchPromptBuilder
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.websearch.service.websearch_executor import WebSearchExecutor


@dataclass(frozen=True)
class LLMUsageRunRecord:
    seq: int
    ts_utc: datetime
    run_id: str
    session_id: str
    user_id: str
    report: LLMUsageReport

    def pretty(self) -> str:
        lines: List[str] = []
        lines.append(f"Run #{self.seq}")
        lines.append(f"  ts_utc     : {self.ts_utc.isoformat()}")
        lines.append(f"  run_id     : {self.run_id}")
        lines.append(f"  session_id : {self.session_id}")
        lines.append(f"  user_id    : {self.user_id}")
        lines.append(self.report.pretty())        

        return "\n".join(lines)

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

    prompt_registry: Optional[YamlPromptRegistry] = None

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
            # Default production registry
            prompt_registry = YamlPromptRegistry(
                catalog_dir=Path("intergrax/prompts/catalog")
            )
            prompt_registry.load_all()

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

        return cls(
            config=config,
            session_manager=session_manager,
            ingestion_service=ingestion_service,
            context_builder=resolved_context_builder,
            rag_prompt_builder=resolved_rag_prompt_builder,
            user_longterm_memory_prompt_builder=resolved_user_ltm_prompt_builder,
            websearch_executor=resolved_websearch_executor,
            websearch_prompt_builder=resolved_websearch_prompt_builder,
            history_prompt_builder=resolved_history_prompt_builder,
            history_layer=resolved_history_layer,            
            prompt_registry=prompt_registry,
        )