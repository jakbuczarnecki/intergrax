# © Artur Czarnecki. All rights reserved.

"""Default Nexus context engine (CE-3.1)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from intergrax.context.contracts import (
    AssembledContext,
    ContextAssemblyProvenance,
    ContextAssemblyRequest,
    ContextFragment,
    ContextProviderContext,
)
from intergrax.context.planner import ContextPlanner
from intergrax.context.session_history import (
    HandleSessionHistoryProvider,
    SessionHistorySnapshot,
)
from intergrax.context.dedup import dedup_fragments_by_hash
from intergrax.context.formatter import DefaultContextFormatter, merge_fragment_messages
from intergrax.context.ranker import DefaultContextRanker
from intergrax.context.registry import ContextPluginRegistry
from intergrax.context.tracking.context_spans import context_span
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.observability.context_counters import get_context_counters
from intergrax.runtime.policy.context_assembly_policy import run_pre_context_policy_gate

logger = logging.getLogger("intergrax.context.engine")
from intergrax.runtime.nexus.context.compile_service import compile_chat_messages
from intergrax.runtime.nexus.context.context_compiler import ContextCompiler, classify_candidates
from intergrax.runtime.nexus.context.context_validator import DefaultContextValidator
from intergrax.runtime.nexus.context.fragment_bridge import fragment_from_candidate

if TYPE_CHECKING:
    from intergrax.llm.messages import ChatMessage
    from intergrax.runtime.nexus.config import RuntimeConfig


class DefaultNexusContextEngine:
    """Shipped CE engine — provider collect + rank + ContextCompiler budget (CE-3)."""

    def __init__(
        self,
        *,
        engine_id: str = "default",
        registry: ContextPluginRegistry | None = None,
        compiler: ContextCompiler | None = None,
        validator: DefaultContextValidator | None = None,
        ranker: DefaultContextRanker | None = None,
        formatter: DefaultContextFormatter | None = None,
    ) -> None:
        self._engine_id = engine_id
        self._registry = registry or ContextPluginRegistry()
        self._compiler = compiler or ContextCompiler()
        self._validator = validator or DefaultContextValidator()
        self._ranker = ranker or DefaultContextRanker()
        self._formatter = formatter or DefaultContextFormatter()

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @property
    def registry(self) -> ContextPluginRegistry:
        return self._registry

    async def assemble(
        self,
        request: ContextAssemblyRequest,
        *,
        provider_ctx: ContextProviderContext | None = None,
    ) -> AssembledContext:
        counters = get_context_counters()
        counters.record_assemble(self._engine_id)
        logger.info(
            "assemble scope=%s task_id=%s step_kind=%s",
            request.assembly_scope,
            request.task_id,
            request.step_kind,
        )
        with context_span("context.engine.assemble"):
            return await self._assemble_inner(request, provider_ctx=provider_ctx)

    async def _assemble_inner(
        self,
        request: ContextAssemblyRequest,
        *,
        provider_ctx: ContextProviderContext | None = None,
    ) -> AssembledContext:
        ctx = provider_ctx or ContextProviderContext(engine_id=self._engine_id)
        runtime_config: RuntimeConfig | None = ctx.handles.get("runtime_config")
        raw_messages: list[ChatMessage] = list(ctx.handles.get("messages") or [])
        max_output_tokens = ctx.handles.get("max_output_tokens")

        if runtime_config is None:
            raise ValueError("ContextProviderContext.handles must include runtime_config")

        event_bus = _event_bus_from_handles(ctx)
        event_ctx = _assembly_event_context(request, ctx)

        pre_gate = run_pre_context_policy_gate(request)
        if not pre_gate.allowed:
            _record_validation_failed(event_bus, event_ctx, pre_gate.errors, stage="pre_context_policy")
            raise ValueError("; ".join(pre_gate.errors))

        collected_fragments: list = []
        fragments_excluded: list[tuple[ContextFragment, str]] = []
        with context_span("context.provider.collect"):
            for provider in self._registry.list_providers():
                fragments = await provider.collect(request, ctx)
                if fragments:
                    counters = get_context_counters()
                    counters.candidate_collected_total += len(fragments)
                    collected_fragments.extend(fragments)
                    if event_bus is not None:
                        from intergrax.runtime.events.context_skill_recording import (
                            record_context_candidate_collected,
                        )

                        record_context_candidate_collected(
                            event_bus,
                            provider_id=provider.provider_id,
                            fragment_count=len(fragments),
                            engine_id=self._engine_id,
                            **event_ctx,
                        )

        unique, dropped = dedup_fragments_by_hash(collected_fragments)
        collected_fragments = unique
        fragments_excluded.extend(dropped)
        if dropped:
            counters = get_context_counters()
            counters.candidate_dropped_total += len(dropped)
            if event_bus is not None:
                from intergrax.runtime.events.context_skill_recording import (
                    record_context_candidate_dropped,
                )

                for fragment, reason in dropped:
                    record_context_candidate_dropped(
                        event_bus,
                        provider_id=fragment.source_id or "dedup",
                        drop_reason=reason,
                        engine_id=self._engine_id,
                        **event_ctx,
                    )

        post_gate = run_pre_context_policy_gate(request, collected=tuple(collected_fragments))
        if not post_gate.allowed:
            get_context_counters().validation_failed_total += 1
            _record_validation_failed(event_bus, event_ctx, post_gate.errors, stage="post_collect_policy")
            raise ValueError("; ".join(post_gate.errors))

        ranked_fragments: list[ContextFragment] = []
        if collected_fragments:
            with context_span("context.budget.allocate"):
                ranked_fragments, quality_excluded = self._ranker.rank_with_exclusions(
                    collected_fragments,
                    request,
                )
                fragments_excluded.extend(quality_excluded)
                if quality_excluded and event_bus is not None:
                    from intergrax.runtime.events.context_skill_recording import (
                        record_context_candidate_dropped,
                    )

                    for fragment, reason in quality_excluded:
                        record_context_candidate_dropped(
                            event_bus,
                            provider_id=fragment.source_id or fragment.source.value,
                            drop_reason=reason,
                            engine_id=self._engine_id,
                            **event_ctx,
                        )

        formatter = self._registry.formatter or self._formatter
        fragment_messages = formatter.format(ranked_fragments, request)
        messages_for_compile = merge_fragment_messages(raw_messages, fragment_messages)

        resolved_budget = self._compiler.resolve_global_input_budget(
            runtime_config,
            max_output_tokens=max_output_tokens,
        )
        session_history = await _load_session_history_snapshot(request, ctx)
        optimization_policy = ctx.handles.get("context_optimization_policy")
        planner = ContextPlanner(count_tokens=self._compiler.count_tokens)
        context_plan = planner.plan(
            request,
            messages_for_compile=messages_for_compile,
            ranked_fragments=ranked_fragments,
            session_history=session_history,
            resolved_global_budget_tokens=resolved_budget,
            optimization_policy=optimization_policy,
            model_family=getattr(runtime_config.llm_adapter, "model", None),
        )

        compile_result = compile_chat_messages(
            messages_for_compile,
            runtime_config,
            compiler=self._compiler,
            max_output_tokens=max_output_tokens,
            run_preflight=False,
        )
        messages = tuple(compile_result.messages)

        candidates = classify_candidates(
            list(messages),
            count_tokens=self._compiler.count_tokens,
        )
        if ranked_fragments:
            fragments_included = tuple(ranked_fragments)
        else:
            fragments_included = tuple(
                fragment_from_candidate(candidate, messages[candidate.message_index])
                for candidate in candidates
                if candidate.message_index < len(messages)
            )
        provenance = tuple(
            ContextAssemblyProvenance(
                source_type=fragment.source.value,
                source_id=fragment.source_id,
                fragment_id=fragment.fragment_id,
            )
            for fragment in fragments_included
        )

        assembled = AssembledContext(
            messages=messages,
            fragments_included=fragments_included,
            fragments_excluded=tuple(fragments_excluded),
            provenance=provenance,
            total_tokens=compile_result.total_tokens,
            budget_tokens=compile_result.budget_tokens,
            degradation_steps=compile_result.degradation_steps,
            context_plan=context_plan,
        )

        validation = self._validator.validate(
            assembled,
            request,
            runtime_config=runtime_config,
            max_output_tokens=max_output_tokens,
        )
        if not validation.valid:
            get_context_counters().validation_failed_total += 1
            _record_validation_failed(event_bus, event_ctx, validation.errors, stage="assembled_validation")
            raise ValueError("; ".join(validation.errors))

        if event_bus is not None:
            from intergrax.runtime.events.context_skill_recording import (
                record_context_assembled_from_engine,
            )

            record_context_assembled_from_engine(
                event_bus,
                assembled=assembled,
                task_id=request.task_id,
                run_id=request.run_id,
                node_id=str(event_ctx.get("node_id") or request.graph_node_id or ""),
                agent_id=event_ctx.get("agent_id") if isinstance(event_ctx.get("agent_id"), str) else None,
                engine_id=self._engine_id,
                step_kind=request.step_kind,
            )

        return assembled


async def _load_session_history_snapshot(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> SessionHistorySnapshot | None:
    provider = HandleSessionHistoryProvider()
    return await provider.load_snapshot(request, ctx)


def _event_bus_from_handles(ctx: ContextProviderContext) -> RuntimeEventBus | None:
    bus = ctx.handles.get("event_bus")
    if isinstance(bus, RuntimeEventBus):
        return bus
    return None


def _assembly_event_context(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> dict[str, str | None]:
    node_id = ctx.handles.get("node_id")
    agent_id = ctx.handles.get("agent_id")
    return {
        "task_id": request.task_id,
        "run_id": request.run_id,
        "node_id": node_id if isinstance(node_id, str) else (request.graph_node_id or ""),
        "agent_id": agent_id if isinstance(agent_id, str) else None,
        "correlation_id": request.trace_id or request.task_id,
    }


def _record_validation_failed(
    event_bus: RuntimeEventBus | None,
    event_ctx: dict[str, str | None],
    errors: tuple[str, ...] | list[str],
    *,
    stage: str,
) -> None:
    if event_bus is None or not errors:
        return
    from intergrax.runtime.events.context_skill_recording import (
        record_context_validation_failed,
    )

    record_context_validation_failed(
        event_bus,
        errors=tuple(errors),
        stage=stage,
        **event_ctx,
    )
