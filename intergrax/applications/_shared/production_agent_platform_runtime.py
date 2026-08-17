# © Artur Czarnecki. All rights reserved.

"""Process-local Agent Platform lifecycle store composition for production hosts (AC-3-FIX-2).

Production chain (reference single-process host):

    Agent Platform lifecycle composition (activation / projection)
        ↓
    ApplicationEnvironmentServingStore + RuntimeRegistryProjectionStore
        ↓
    active traffic-serving RuntimeRevision id
        ↓
    MaterializedRegistryProjection
        ↓
    production application factory → Nexus

Host startup is a **consumer** of stores populated before traffic serving. It does
**not** build registry projections from manifest and does **not** instantiate fresh
lifecycle stores per bootstrap call.

Current adapters are process-local in-memory implementations. They are suitable for
reference-host production semantics only; durable multi-instance persistence remains
deferred.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryApplicationEnvironmentServingStore,
)
from intergrax.agent_distribution.stores import ApplicationEnvironmentServingStore
from intergrax.applications._shared.registry_projection import (
    InMemoryRuntimeRegistryProjectionStore,
    RuntimeRegistryProjectionStore,
)


@dataclass(frozen=True, slots=True)
class AgentPlatformRuntimeStores:
    """Typed AP-9/AP-10 store bundle shared by activation and production host startup."""

    serving_store: ApplicationEnvironmentServingStore
    registry_projection_store: RuntimeRegistryProjectionStore


@dataclass(frozen=True, slots=True)
class ProductionAgentPlatformRuntime:
    """Canonical process-lifetime owner for reference production AP lifecycle stores."""

    distribution_state: AgentDistributionStoreState
    stores: AgentPlatformRuntimeStores


def build_production_agent_platform_runtime() -> ProductionAgentPlatformRuntime:
    """Build one process-local AP lifecycle store composition for a production host."""
    state = AgentDistributionStoreState()
    return ProductionAgentPlatformRuntime(
        distribution_state=state,
        stores=AgentPlatformRuntimeStores(
            serving_store=InMemoryApplicationEnvironmentServingStore(state),
            registry_projection_store=InMemoryRuntimeRegistryProjectionStore(),
        ),
    )


__all__ = [
    "AgentPlatformRuntimeStores",
    "ProductionAgentPlatformRuntime",
    "build_production_agent_platform_runtime",
]
