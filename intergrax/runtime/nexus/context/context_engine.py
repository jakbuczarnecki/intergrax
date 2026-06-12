# © Artur Czarnecki. All rights reserved.

"""Default Nexus context engine (CE-3.1)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from intergrax.context.contracts import (
    AssembledContext,
    ContextAssemblyProvenance,
    ContextAssemblyRequest,
    ContextProviderContext,
)
from intergrax.context.dedup import dedup_fragments_by_hash
from intergrax.context.ranker import DefaultContextRanker
from intergrax.context.registry import ContextPluginRegistry
from intergrax.context.tracking.context_spans import context_span
from intergrax.runtime.observability.context_counters import get_context_counters
from intergrax.runtime.policy.context_assembly_policy import run_pre_context_policy_gate

logger = logging.getLogger("intergrax.context.engine")
from intergrax.runtime.nexus.context.compile_service import compile_chat_messages
from intergrax.runtime.nexus.context.context_compiler import ContextCompiler
from intergrax.runtime.nexus.context.context_validator import DefaultContextValidator
from intergrax.runtime.nexus.context.fragment_bridge import fragment_from_candidate
from intergrax.runtime.nexus.context.context_compiler import classify_candidates

if TYPE_CHECKING:
    from intergrax.llm.messages import ChatMessage
    from intergrax.runtime.nexus.config import RuntimeConfig


class DefaultNexusContextEngine:
    """Shipped CE engine — provider collect stub + ContextCompiler budget (CE-3)."""

    def __init__(
        self,
        *,
        engine_id: str = "default",
        registry: ContextPluginRegistry | None = None,
        compiler: ContextCompiler | None = None,
        validator: DefaultContextValidator | None = None,
        ranker: DefaultContextRanker | None = None,
    ) -> None:
        self._engine_id = engine_id
        self._registry = registry or ContextPluginRegistry()
        self._compiler = compiler or ContextCompiler()
        self._validator = validator or DefaultContextValidator()
        self._ranker = ranker or DefaultContextRanker()

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

        pre_gate = run_pre_context_policy_gate(request)
        if not pre_gate.allowed:
            raise ValueError("; ".join(pre_gate.errors))

        collected_fragments: list = []
        with context_span("context.provider.collect"):
            for provider in self._registry.list_providers():
                fragments = await provider.collect(request, ctx)
                if fragments:
                    counters = get_context_counters()
                    counters.candidate_collected_total += len(fragments)
                    collected_fragments.extend(fragments)

        unique, dropped = dedup_fragments_by_hash(collected_fragments)
        collected_fragments = unique
        if dropped:
            counters = get_context_counters()
            counters.candidate_dropped_total += len(dropped)

        post_gate = run_pre_context_policy_gate(request, collected=tuple(collected_fragments))
        if not post_gate.allowed:
            get_context_counters().validation_failed_total += 1
            raise ValueError("; ".join(post_gate.errors))

        if collected_fragments:
            with context_span("context.budget.allocate"):
                collected_fragments = self._ranker.rank(collected_fragments, request)

        compile_result = compile_chat_messages(
            raw_messages,
            runtime_config,
            compiler=self._compiler,
            max_output_tokens=max_output_tokens,
            run_preflight=False,
        )
        messages = tuple(compile_result.messages)

        candidates = classify_candidates(
            list(messages),
            count_tokens=self._compiler._count_tokens,  # noqa: SLF001
        )
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
            fragments_excluded=(),
            provenance=provenance,
            total_tokens=compile_result.total_tokens,
            budget_tokens=compile_result.budget_tokens,
            degradation_steps=compile_result.degradation_steps,
        )

        validation = self._validator.validate(
            assembled,
            request,
            runtime_config=runtime_config,
            max_output_tokens=max_output_tokens,
        )
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))

        return assembled
