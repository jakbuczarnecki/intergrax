# © Artur Czarnecki. All rights reserved.

"""Enterprise durable agent lifecycle proof composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.agent_distribution.agent_manager_query_service import (
    AgentManagerQueryService,
)
from intergrax.agent_distribution.effective_roster import EffectiveRosterBuilder
from intergrax.agent_distribution.agent_project_metadata import AgentProjectMetadata
from intergrax.agent_distribution.catalog import AgentCatalogEntry, CatalogSourceIdentity
from intergrax.applications._shared.durable_agent_platform_runtime import (
    DurableAgentPlatformRuntime,
    build_durable_production_agent_platform_runtime,
)
from intergrax.applications._shared.production_agent_capability_runtime import (
    build_production_agent_capability_runtime,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleLauncher,
    wire_durable_reference_production_lifecycle_services,
)
from intergrax.applications._shared.reference_production_governance_wiring import (
    ReferenceProductionControlPlaneGovernance,
    build_reference_production_control_plane_governance,
)
from testing_support.canonical_agent_lifecycle_composition import (
    CanonicalAgentLifecycleProofStack,
    CanonicalLifecycleProofConfig,
    CanonicalLifecycleProofResult,
    _MetadataProvider,
    _StaticCatalogProvider,
    _build_application_composition,
    _stage15_proof_environment,
    default_stage15_proof_config,
)
from intergrax.applications._shared.registry_projection_rehydrator import (
    rehydrate_serving_registry_projection,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest


@dataclass(frozen=True, slots=True)
class EnterpriseAgentLifecycleProofStack:
    """Durable-backed specialization of the Stage 15 canonical lifecycle proof."""

    durable_runtime: DurableAgentPlatformRuntime
    canonical: CanonicalAgentLifecycleProofStack

    @property
    def config(self) -> CanonicalLifecycleProofConfig:
        return self.canonical.config

    @property
    def composition(self) -> ProductionProcessComposition:
        return self.canonical.composition

    @property
    def admin(self):
        return self.canonical.admin

    @property
    def launcher(self) -> ReferenceProductionLifecycleLauncher:
        return self.canonical.launcher

    @property
    def agent_manager_query(self) -> AgentManagerQueryService:
        return self.canonical.agent_manager_query

    @classmethod
    def build(
        cls,
        tmp_path: Path,
        *,
        db_path: Path | None = None,
        config: CanonicalLifecycleProofConfig | None = None,
    ) -> EnterpriseAgentLifecycleProofStack:
        resolved = config or default_stage15_proof_config()
        lifecycle_db = db_path or (tmp_path / "enterprise-lifecycle.db")
        durable_runtime = build_durable_production_agent_platform_runtime(lifecycle_db)
        environment = _stage15_proof_environment(resolved.environment_id)
        catalog_entry = AgentCatalogEntry(
            catalog_entry_id=resolved.catalog_entry_id,
            catalog_source=CatalogSourceIdentity(
                catalog_source_id=resolved.catalog_source_id,
                provider_kind=resolved.catalog_provider_kind,
            ),
            display_name="Canonical Ping",
            package_id_line=resolved.distribution_package_id,
        )
        catalog_provider = _StaticCatalogProvider((catalog_entry,))
        metadata_provider = _MetadataProvider(
            {
                resolved.metadata_ref: AgentProjectMetadata(
                    distribution_package_id=resolved.distribution_package_id,
                    dependencies=(),
                ),
            }
        )
        application_composition = _build_application_composition(
            resolved,
            tmp_path,
            catalog_provider,
            metadata_provider,
        )
        agent_platform_runtime = durable_runtime.agent_platform_runtime
        lifecycle_services = wire_durable_reference_production_lifecycle_services(
            ProductionProcessComposition(
                agent_platform_runtime=agent_platform_runtime,
                agent_capability_runtime=None,
            ),
            durable_store_bundle=durable_runtime.distribution_store_bundle,
        )
        capability_runtime = build_production_agent_capability_runtime(
            agent_platform_runtime=agent_platform_runtime,
            application_composition=application_composition,
            lifecycle_services=lifecycle_services,
            durable_store_bundle=durable_runtime.distribution_store_bundle,
        )
        composition = ProductionProcessComposition(
            agent_platform_runtime=agent_platform_runtime,
            agent_capability_runtime=capability_runtime,
        )
        governance = build_reference_production_control_plane_governance(environment)
        launcher = ReferenceProductionLifecycleLauncher(
            composition,
            services=lifecycle_services,
            mutation_authorization_boundary=governance.mutation_authorization_boundary,
            environment_tenant_resolver=governance.environment_tenant_resolver,
        )
        bundle = durable_runtime.distribution_store_bundle
        agent_manager_query = AgentManagerQueryService(
            catalog_provider=catalog_provider,
            installation_store=bundle.installation_store,
            binding_store=bundle.binding_store,
            revision_store=bundle.revision_store,
            serving_store=bundle.serving_store,
            roster_builder=EffectiveRosterBuilder(bundle.installation_store),
        )
        manifest = ApplicationManifest.lab(
            app_id=resolved.application_id,
            name="Enterprise Durable Lifecycle Proof",
            agents=[
                AgentBinding(
                    contract_id=resolved.logical_agent_id,
                    builder_key=resolved.logical_agent_id,
                ),
            ],
        )
        canonical = CanonicalAgentLifecycleProofStack(
            config=resolved,
            runtime_root=tmp_path,
            composition=composition,
            launcher=launcher,
            governance=governance,
            catalog_provider=catalog_provider,
            manifest=manifest,
            environment=environment,
            agent_manager_query=agent_manager_query,
        )
        return cls(durable_runtime=durable_runtime, canonical=canonical)

    @classmethod
    def reopen(cls, tmp_path: Path, db_path: Path, config: CanonicalLifecycleProofConfig) -> EnterpriseAgentLifecycleProofStack:
        stack = cls.build(tmp_path, db_path=db_path, config=config)
        stack.rehydrate_serving_runtime()
        return stack

    def rehydrate_serving_runtime(self) -> None:
        """Rebuild process-local serving projection from durable revision-bound authority."""
        rehydrate_serving_registry_projection(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
            rehydrator=self.durable_runtime.registry_projection_rehydrator,
        )

    def run_happy_path(self) -> CanonicalLifecycleProofResult:
        return self.canonical.run_happy_path()

    def discover_catalog_entry(self) -> AgentCatalogEntry:
        return self.canonical.discover_catalog_entry()

    def install_from_catalog(self, *, mutation_id: str = "mut-enterprise-install") -> None:
        self.canonical.install_from_catalog(mutation_id=mutation_id)

    def bind_enabled_agent(self, *, mutation_id: str = "mut-enterprise-bind") -> int:
        return self.canonical.bind_enabled_agent(mutation_id=mutation_id)

    def build_revision(self, **kwargs: object):
        return self.canonical.build_revision(**kwargs)

    def register_projection_and_activate(self, built) -> str:
        return self.canonical.register_projection_and_activate(built)

    def resolve_serving_projection(self):
        return self.canonical.resolve_serving_projection()

    async def execute_canonical(self):
        return await self.canonical.execute_canonical()


__all__ = [
    "EnterpriseAgentLifecycleProofStack",
]
