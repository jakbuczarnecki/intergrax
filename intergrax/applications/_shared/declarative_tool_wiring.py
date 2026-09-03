# © Artur Czarnecki. All rights reserved.

"""Tier-3 wiring for ACP declarative catalog tool invoker."""

from __future__ import annotations

from intergrax.agents.persistence.catalog_declarative_invoker import (
    CatalogDeclarativeToolInvoker,
)
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.sandbox.isolation_gate import sandbox_availability_provider
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)


def build_declarative_invoker_from_tool_wiring(
    tool_wiring: ApplicationToolWiring,
    *,
    idempotency_store: IdempotencyStore | None = None,
) -> CatalogDeclarativeToolInvoker | None:
    """Materialize catalog invoker when host tool profile enables catalog tools."""
    if not tool_wiring.profile.enabled and not tool_wiring.profile.enabled_bundles:
        return None
    coordinator = (
        IdempotencyPreEffectCoordinator(idempotency_store=idempotency_store)
        if idempotency_store is not None
        else None
    )
    invoker = RuntimeToolInvoker(
        registry=tool_wiring.registry,
        executor=RegistryToolExecutor(tool_wiring.registry),
        pre_effect_coordinator=coordinator,
        sandbox_availability=sandbox_availability_provider(tool_wiring.wiring_context),
    )
    return CatalogDeclarativeToolInvoker(tool_invoker=invoker)
