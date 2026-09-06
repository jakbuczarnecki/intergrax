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
from intergrax.applications._shared.registry_projection_descriptor import (
    RuntimeRegistryProjectionDescriptorStore,
)
from intergrax.applications._shared.registry_projection_rehydrator import (
    RegistryProjectionRehydrationResult,
    RuntimeRegistryProjectionRehydrator,
    rehydrate_serving_registry_projection,
)


@dataclass(frozen=True, slots=True)
class DurableAgentPlatformRuntime:
    """Process composition root for durable reference production lifecycle."""

    db_path: Path
    distribution_store_bundle: SqliteAgentDistributionStoreBundle
    platform_persistence: ProductionPlatformPersistence
    stores: AgentPlatformRuntimeStores
    effective_roster_authority: EffectiveRosterAuthorityService
    registry_projection_authority: RegistryProjectionAuthorityResolver
    projection_descriptor_store: RuntimeRegistryProjectionDescriptorStore
    registry_projection_rehydrator: RuntimeRegistryProjectionRehydrator

    @property
    def agent_platform_runtime(self) -> ProductionAgentPlatformRuntime:
        """Adapter view for existing production composition helpers."""
        return ProductionAgentPlatformRuntime(
            distribution_state=self.distribution_store_bundle.installation_store.state,
            stores=self.stores,
            platform_persistence=self.platform_persistence,
            effective_roster_authority=self.effective_roster_authority,
            registry_projection_authority=self.registry_projection_authority,
        )

    def rehydrate_serving_registry_projection(
        self,
        *,
        application_id: str,
        application_environment_id: str,
    ) -> RegistryProjectionRehydrationResult:
        """Deterministically rebuild process-local serving projection from durable authority."""
        return rehydrate_serving_registry_projection(
            application_id=application_id,
            application_environment_id=application_environment_id,
            rehydrator=self.registry_projection_rehydrator,
        )


def build_durable_production_agent_platform_runtime(
    db_path: Path,
) -> DurableAgentPlatformRuntime:
    """Construct one durable AP lifecycle store bundle backed by SQLite."""
    distribution_bundle = build_sqlite_agent_distribution_store_bundle(db_path)
    platform_persistence = build_reference_production_platform_persistence(db_path=db_path)
    effective_roster_snapshot_store = distribution_bundle.effective_roster_snapshot_store
    revision_store = distribution_bundle.revision_store
    lock_store = distribution_bundle.lock_store
    materialization_store = distribution_bundle.materialization_store
    effective_roster_authority = EffectiveRosterAuthorityService(
        snapshot_store=effective_roster_snapshot_store,
    )
    projection_store = InMemoryRuntimeRegistryProjectionStore()
    stores = AgentPlatformRuntimeStores(
        serving_store=distribution_bundle.serving_store,
        registry_projection_store=projection_store,
        revision_store=revision_store,
        lock_store=lock_store,
        materialization_store=materialization_store,
        effective_roster_snapshot_store=effective_roster_snapshot_store,
    )
    registry_projection_authority = RegistryProjectionAuthorityResolver(
        revision_store=revision_store,
        effective_roster_authority=effective_roster_authority,
        lock_store=lock_store,
        materialization_store=materialization_store,
    )
    descriptor_store = distribution_bundle.projection_descriptor_store
    rehydrator = RuntimeRegistryProjectionRehydrator(
        serving_store=distribution_bundle.serving_store,
        descriptor_store=descriptor_store,
        authority=registry_projection_authority,
        projection_store=projection_store,
    )
    return DurableAgentPlatformRuntime(
        db_path=db_path,
        distribution_store_bundle=distribution_bundle,
        platform_persistence=platform_persistence,
        stores=stores,
        effective_roster_authority=effective_roster_authority,
        registry_projection_authority=registry_projection_authority,
        projection_descriptor_store=descriptor_store,
        registry_projection_rehydrator=rehydrator,
    )


__all__ = [
    "DurableAgentPlatformRuntime",
    "build_durable_production_agent_platform_runtime",
]
