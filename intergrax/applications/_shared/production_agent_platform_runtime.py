# © Artur Czarnecki. All rights reserved.

"""Process-local Agent Platform lifecycle store bundle (AGENT-CONSOLIDATION-3-ARCH).

Reference production V1 chain (single process, one composition root):

    ProductionProcessComposition
        ↓
    ProductionAgentPlatformRuntime.stores
        ↓
    AP-9 activation → ApplicationEnvironmentServingStore
    AP-10 projection → RuntimeRegistryProjectionStore
        ↓
    active traffic-serving RuntimeRevision id
        ↓
    MaterializedRegistryProjection
        ↓
    production application host → Nexus

``build_production_agent_platform_runtime()`` constructs **one new process-local
lifecycle universe**. It does **not** resolve already-active production state.
Only ``ProductionProcessComposition`` (or an equivalent process composition root)
may call it once per process.

Host startup is a **consumer** of stores populated by deploy/activate. It does
**not** build registry projections from manifest and does **not** instantiate fresh
lifecycle stores per bootstrap call.

Current adapters are process-local in-memory implementations. They support
**reference single-process production semantics** only — restart loses lifecycle
state and multi-instance deployment is not supported at this adapter tier.
Durable multi-instance production remains deferred (see ADR-AGENT-005).
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agent_distribution.effective_roster_authority import (
    EffectiveRosterAuthorityService,
)
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryApplicationEnvironmentServingStore,
    InMemoryEffectiveRosterSnapshotStore,
    InMemoryMaterializedRuntimeLockStore,
    InMemoryRuntimeMaterializationStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.stores import (
    ApplicationEnvironmentServingStore,
    EffectiveRosterSnapshotStore,
    MaterializedRuntimeLockStore,
    RuntimeMaterializationStore,
    RuntimeRevisionStore,
)
from intergrax.applications._shared.registry_projection import (
    InMemoryRuntimeRegistryProjectionStore,
    RuntimeRegistryProjectionStore,
)


@dataclass(frozen=True, slots=True)
class AgentPlatformRuntimeStores:
    """Typed AP-9/AP-10 store bundle shared by activation and production host startup."""

    serving_store: ApplicationEnvironmentServingStore
    registry_projection_store: RuntimeRegistryProjectionStore
    revision_store: RuntimeRevisionStore
    lock_store: MaterializedRuntimeLockStore
    materialization_store: RuntimeMaterializationStore
    effective_roster_snapshot_store: EffectiveRosterSnapshotStore


@dataclass(frozen=True, slots=True)
class ProductionAgentPlatformRuntime:
    """Canonical process-lifetime owner for reference production AP lifecycle stores."""

    distribution_state: AgentDistributionStoreState
    stores: AgentPlatformRuntimeStores
    effective_roster_authority: EffectiveRosterAuthorityService


def build_production_agent_platform_runtime() -> ProductionAgentPlatformRuntime:
    """Construct one new process-local AP lifecycle store bundle for a composition root.

    Callers MUST be the process composition root (``ProductionProcessComposition``).
    Application ``main.py`` and factories MUST NOT be canonical owners.
    """
    state = AgentDistributionStoreState()
    effective_roster_snapshot_store = InMemoryEffectiveRosterSnapshotStore(state)
    return ProductionAgentPlatformRuntime(
        distribution_state=state,
        stores=AgentPlatformRuntimeStores(
            serving_store=InMemoryApplicationEnvironmentServingStore(state),
            registry_projection_store=InMemoryRuntimeRegistryProjectionStore(),
            revision_store=InMemoryRuntimeRevisionStore(state),
            lock_store=InMemoryMaterializedRuntimeLockStore(state),
            materialization_store=InMemoryRuntimeMaterializationStore(state),
            effective_roster_snapshot_store=effective_roster_snapshot_store,
        ),
        effective_roster_authority=EffectiveRosterAuthorityService(
            snapshot_store=effective_roster_snapshot_store,
        ),
    )


create_process_local_agent_platform_runtime = build_production_agent_platform_runtime

__all__ = [
    "AgentPlatformRuntimeStores",
    "EffectiveRosterAuthorityService",
    "ProductionAgentPlatformRuntime",
    "build_production_agent_platform_runtime",
    "create_process_local_agent_platform_runtime",
]
