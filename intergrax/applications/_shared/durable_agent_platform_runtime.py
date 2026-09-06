# © Artur Czarnecki. All rights reserved.

"""Durable single-host Agent Platform runtime composition (enterprise reference)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.agent_distribution.effective_roster_authority import (
    EffectiveRosterAuthorityService,
)
from intergrax.agent_distribution.sqlite_stores import (
    SqliteAgentDistributionStoreBundle,
    build_sqlite_agent_distribution_store_bundle,
)
from intergrax.applications._shared.production_agent_platform_runtime import (
    AgentPlatformRuntimeStores,
    ProductionAgentPlatformRuntime,
)
from intergrax.applications._shared.registry_projection import (
    InMemoryRuntimeRegistryProjectionStore,
)
from intergrax.applications._shared.registry_projection_authority_resolver import (
    RegistryProjectionAuthorityResolver,
)


@dataclass(frozen=True, slots=True)
class DurableAgentPlatformRuntime:
    """Process composition root for durable reference production lifecycle."""

    db_path: Path
    distribution_store_bundle: SqliteAgentDistributionStoreBundle
    stores: AgentPlatformRuntimeStores
    effective_roster_authority: EffectiveRosterAuthorityService
    registry_projection_authority: RegistryProjectionAuthorityResolver

    @property
    def agent_platform_runtime(self) -> ProductionAgentPlatformRuntime:
        """Adapter view for existing production composition helpers."""
        return ProductionAgentPlatformRuntime(
            distribution_state=self.distribution_store_bundle.installation_store.state,
            stores=self.stores,
            effective_roster_authority=self.effective_roster_authority,
            registry_projection_authority=self.registry_projection_authority,
        )


def build_durable_production_agent_platform_runtime(
    db_path: Path,
) -> DurableAgentPlatformRuntime:
    """Construct one durable AP lifecycle store bundle backed by SQLite."""
    distribution_bundle = build_sqlite_agent_distribution_store_bundle(db_path)
    effective_roster_snapshot_store = distribution_bundle.effective_roster_snapshot_store
    revision_store = distribution_bundle.revision_store
    lock_store = distribution_bundle.lock_store
    materialization_store = distribution_bundle.materialization_store
    effective_roster_authority = EffectiveRosterAuthorityService(
        snapshot_store=effective_roster_snapshot_store,
    )
    stores = AgentPlatformRuntimeStores(
        serving_store=distribution_bundle.serving_store,
        registry_projection_store=InMemoryRuntimeRegistryProjectionStore(),
        revision_store=revision_store,
        lock_store=lock_store,
        materialization_store=materialization_store,
        effective_roster_snapshot_store=effective_roster_snapshot_store,
    )
    return DurableAgentPlatformRuntime(
        db_path=db_path,
        distribution_store_bundle=distribution_bundle,
        stores=stores,
        effective_roster_authority=effective_roster_authority,
        registry_projection_authority=RegistryProjectionAuthorityResolver(
            revision_store=revision_store,
            effective_roster_authority=effective_roster_authority,
            lock_store=lock_store,
            materialization_store=materialization_store,
        ),
    )


__all__ = [
    "DurableAgentPlatformRuntime",
    "build_durable_production_agent_platform_runtime",
]
