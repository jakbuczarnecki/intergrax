# © Artur Czarnecki. All rights reserved.

"""Bounded multi-hop context collection (CE-8.1)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from intergrax.context.contracts import ContextAssemblyRequest, ContextFragment, ContextProviderContext
from intergrax.context.protocols import ContextEngine


@dataclass(frozen=True, slots=True)
class ContextOrchestratorConfig:
    max_hops: int = 2
    latency_budget_ms: int = 500


class ContextOrchestrator:
    """Runs bounded collect hops for codebase preset only (CE-8.2)."""

    def __init__(self, engine: ContextEngine, *, config: ContextOrchestratorConfig | None = None) -> None:
        self._engine = engine
        self._config = config or ContextOrchestratorConfig()

    async def assemble_with_hops(
        self,
        request: ContextAssemblyRequest,
        *,
        provider_ctx: ContextProviderContext,
    ):
        started = time.perf_counter()
        last = await self._engine.assemble(request, provider_ctx=provider_ctx)
        hops = 1
        while hops < self._config.max_hops:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            if elapsed_ms >= self._config.latency_budget_ms:
                break
            if not last.fragments_included:
                break
            hops += 1
            provider_ctx.handles["orchestrator_hop"] = hops
            last = await self._engine.assemble(request, provider_ctx=provider_ctx)
        return last
