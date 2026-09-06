# © Artur Czarnecki. All rights reserved.

"""Reference Production V1 lifecycle proof composition for AI Incident Investigation."""

from __future__ import annotations

import asyncio
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    BindAgentRequest,
    BuildApplicationRevisionRequest,
    BuildRevisionResult,
    InstallAgentRequest,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
from intergrax.agent_distribution.agent_manager_query_service import (
    AgentManagerQueryService,
)
from intergrax.agent_distribution.agent_project_metadata import AgentProjectMetadata
from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    CatalogProviderKind,
    CatalogSourceIdentity,
    CatalogSourceProvider,
)
from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
)
from intergrax.agent_distribution.dependency import RepositoryDependencyDeclaration
from intergrax.agent_distribution.effective_roster import EffectiveRosterBuilder
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.in_memory_stores import (
    InMemoryAgentInstallationStore,
    InMemoryApplicationAgentBindingStore,
)
from intergrax.agent_distribution.materialization import (
    MaterializationInput,
    MaterializationOutput,
)
from intergrax.agent_distribution.materialization_service import (
    RuntimeMaterializationService,
)
from intergrax.agent_distribution.runtime_context_staging import (
    RUNTIME_LOCK_MANIFEST_FILENAME,
    directory_content_digest,
)
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.agent_distribution.task_capability_resolution import (
    build_deterministic_task_capability_resolver,
    build_task_capability_rule,
)
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from intergrax.applications._shared.application_owned_tool_conformance import (
    application_owned_tool_declarations,
)
from intergrax.applications._shared.production_agent_capability_runtime import (
    AgentCapabilityApplicationComposition,
    ProductionAgentPlatformAdminConfig,
    build_production_agent_capability_runtime,
)
from intergrax.applications._shared.production_delegated_subtask_plans import (
    ProductionDelegatedSubtaskPlanConfig,
)
from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
    create_reference_production_process_composition,
)
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    build_production_registry_projection_input_bundle_for_revision,
)
from intergrax.applications._shared.reference_production_governance_wiring import (
    ReferenceProductionControlPlaneGovernance,
    build_reference_production_control_plane_governance,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleLauncher,
    wire_reference_production_lifecycle_services,
)
from intergrax.applications._shared.registry_projection import (
    MaterializedRegistryProjection,
)
from intergrax.applications._shared.registry_projection_input_bundle import (
    reference_admission_mutation_id,
)
from intergrax.applications._shared.scenario_runtime_baseline import (
    build_scenario_runtime_from_environment,
)
from intergrax.applications._shared.scenario_runtime_profiles import ScenarioRuntimeMode
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.core.qualification import QualificationStatus
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.registry.presets import OTEL
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_contract import (
    INVESTIGATOR_AGENT_ID,
    INVESTIGATOR_CAPABILITY,
)
from platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition import (
    build_scenario_environment_profile,
    prepare_incident_execution_runtime,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    OUTCOME_RESOLVED,
    ScenarioExecutionResult,
    ScenarioRuntimeBundle,
    STANDALONE_SCENARIO_TENANT_ID,
    execute_resolved_skeleton,
)
from platform_proofs.scenarios.ai_incident_investigation.application.tools import (
    SCENARIO_TOOL_IDS,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import (
    IncidentFixture,
    build_resolved_fixture,
)
from platform_proofs.scenarios.ai_incident_investigation.integration.agent_factory import (
    IncidentInvestigatorProductionSettings,
    build_default_production_settings,
)
from platform_proofs.scenarios.ai_incident_investigation.integration.package_identity import (
    INCIDENT_INVESTIGATOR_APPLICATION_BINDING_ID,
    INCIDENT_INVESTIGATOR_APPLICATION_ID,
    INCIDENT_INVESTIGATOR_CATALOG_ENTRY_ID,
    INCIDENT_INVESTIGATOR_CATALOG_SOURCE_ID,
    INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID,
    INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
    INCIDENT_INVESTIGATOR_FACTORY_REFERENCE,
    INCIDENT_INVESTIGATOR_INSTALLATION_ID,
    INCIDENT_INVESTIGATOR_INSTALLATION_SLOT_ID,
    INCIDENT_INVESTIGATOR_METADATA_REF,
    INCIDENT_INVESTIGATOR_PACKAGE_DIGEST,
    INCIDENT_INVESTIGATOR_PACKAGE_VERSION,
    INCIDENT_INVESTIGATOR_RUNTIME_REVISION_ID,
)
from testing_support.agent_platform_admin_harness import (
    admin_test_principal,
    allow_mutation_boundary,
)
from testing_support.agent_platform_dependency_resolver import (
    make_identity_dependency_resolver,
)

if TYPE_CHECKING:
    from intergrax.agent_distribution.admin_models import EffectiveRosterView


class _MetadataProvider:
    def __init__(self, records: dict[str, AgentProjectMetadata]) -> None:
        self._records = records

    def get_metadata(self, metadata_ref: str) -> AgentProjectMetadata | None:
        return self._records.get(metadata_ref)


class _StaticCatalogProvider:
    def __init__(self, entries: tuple[AgentCatalogEntry, ...]) -> None:
        self._entries = entries

    @property
    def catalog_source_id(self) -> str:
        return self._entries[0].catalog_source.catalog_source_id

    def list_entries(self, filters: object | None = None) -> list[AgentCatalogEntry]:
        del filters
        return list(self._entries)

    def resolve_package(
        self, entry: AgentCatalogEntry, *, version_selector: str
    ) -> object:
        del entry, version_selector
        raise NotImplementedError

    def health(self) -> None:
        return None


class _IncidentTrustRecordFactory:
    def build_trust_record(
        self,
        *,
        package_digest: str,
        package_id: str,
    ) -> AgentInstallationTrustRecord:
        del package_id
        return AgentInstallationTrustRecord(
            qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
            package_digest=package_digest,
            publisher_identity_ref="publisher:ai-incident-investigator",
            source_provider_id=INCIDENT_INVESTIGATOR_CATALOG_SOURCE_ID,
            trust_evidence_refs=(
                AgentTrustEvidenceRef(
                    evidence_id="evidence:aipv-1",
                    kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                ),
            ),
        )


class _IncidentInvestigatorVenvBundleMaterializer:
    topology = MaterializationTopology.VENV_BUNDLE
    materializer_id = "intergrax.ai-incident-investigator"
    materializer_version = "1.0.0"

    def __init__(self, artifact_root_base: Path) -> None:
        self._artifact_root_base = artifact_root_base

    def materialize(
        self, materialization_input: MaterializationInput
    ) -> MaterializationOutput:
        revision_id = materialization_input.runtime_revision.runtime_revision_id
        artifact_root = self._artifact_root_base / revision_id
        site_packages = artifact_root / "site-packages"
        site_packages.mkdir(parents=True, exist_ok=True)
        package_dir = site_packages / "incident_investigator_agent"
        package_dir.mkdir(exist_ok=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        (package_dir / "factory.py").write_text(
            textwrap.dedent(
                """
                from platform_proofs.scenarios.ai_incident_investigation.integration.agent_factory import (
                    build_agent,
                )

                __all__ = ["build_agent"]
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        lock = materialization_input.materialized_runtime_lock
        (artifact_root / RUNTIME_LOCK_MANIFEST_FILENAME).write_text(
            json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (artifact_root / ".intergrax-runtime-graph.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        digest = directory_content_digest(artifact_root)
        return MaterializationOutput(
            materialization_artifact_digest=digest,
            artifact_locator=f"test://{artifact_root.resolve().as_posix()}",
            health_check_evidence_ref=f"test://{artifact_root.resolve().as_posix()}",
            runtime_graph_manifest_path=".intergrax-runtime-graph.json",
            topology=self.topology,
        )


def _incident_proof_environment() -> ApplicationEnvironmentProfile:
    environment = build_scenario_environment_profile()
    integration = environment.integration_profile
    assert integration is not None
    return environment.model_copy(
        update={
            "profile_id": INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
            "capabilities": environment.capabilities.model_copy(
                update={
                    "integrations": integration.model_copy(
                        update={"observability_backend": OTEL},
                    ),
                },
            ),
        },
    )


def _incident_production_manifest() -> ApplicationManifest:
    environment = _incident_proof_environment()
    return ApplicationManifest(
        app_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
        name="AI Incident Investigation Production Validation",
        route_prefix="/v1/scenario/ai_incident_investigation",
        env_prefix="SCENARIO_AI_INCIDENT_",
        agents=[
            AgentBinding.reference(
                contract_id=INVESTIGATOR_AGENT_ID,
                capabilities=[INVESTIGATOR_CAPABILITY],
            ),
        ],
        application_owned_tools=application_owned_tool_declarations(SCENARIO_TOOL_IDS),
        environment=environment,
    )


def _build_application_composition(
    tmp_path: Path,
    catalog_provider: CatalogSourceProvider,
    metadata_provider: _MetadataProvider,
) -> AgentCapabilityApplicationComposition:
    return AgentCapabilityApplicationComposition(
        capability_resolver=build_deterministic_task_capability_resolver(
            rules=(
                build_task_capability_rule(
                    rule_id="rule.incident.investigate.v1",
                    task_kind="scenario.incident.investigate",
                    required=(INVESTIGATOR_CAPABILITY,),
                ),
            ),
        ),
        catalog_providers=(catalog_provider,),
        package_metadata_refs={
            INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID: INCIDENT_INVESTIGATOR_METADATA_REF
        },
        package_logical_agents={
            INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID: INVESTIGATOR_AGENT_ID
        },
        trust_record_factory=_IncidentTrustRecordFactory(),
        admin_config=ProductionAgentPlatformAdminConfig(
            metadata_provider=metadata_provider,
            materialization_service=RuntimeMaterializationService(
                {
                    MaterializationTopology.VENV_BUNDLE: _IncidentInvestigatorVenvBundleMaterializer(
                        tmp_path,
                    ),
                }
            ),
            dependency_resolver=make_identity_dependency_resolver(),
            mutation_authorization_boundary=allow_mutation_boundary(),
            environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
                "tenant-test",
            ),
            catalog_provider=catalog_provider,
        ),
        delegated_plan_config=ProductionDelegatedSubtaskPlanConfig(
            application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
            application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
            application_release_id="rel-ai-incident-aipv1",
            package_metadata_refs={
                INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID: INCIDENT_INVESTIGATOR_METADATA_REF
            },
            package_logical_agents={
                INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID: INVESTIGATOR_AGENT_ID
            },
        ),
    )


@dataclass(frozen=True, slots=True)
class IncidentInvestigatorLifecycleProofResult:
    package_digest: str
    runtime_revision_id: str
    traffic_serving_revision_id: str
    execution_outcome: str


@dataclass
class IncidentInvestigatorAgentPlatformProofStack:
    runtime_root: Path
    composition: ProductionProcessComposition
    launcher: ReferenceProductionLifecycleLauncher
    governance: ReferenceProductionControlPlaneGovernance
    catalog_provider: CatalogSourceProvider
    manifest: ApplicationManifest
    environment: ApplicationEnvironmentProfile
    agent_manager_query: AgentManagerQueryService
    production_settings: IncidentInvestigatorProductionSettings
    fixture: IncidentFixture

    @property
    def admin(self) -> AgentPlatformAdminService:
        capability_runtime = self.composition.agent_capability_runtime
        assert capability_runtime is not None
        return capability_runtime.admin_service

    @classmethod
    def build(
        cls,
        tmp_path: Path,
        *,
        fixture: IncidentFixture | None = None,
    ) -> IncidentInvestigatorAgentPlatformProofStack:
        resolved_fixture = fixture or build_resolved_fixture()
        production_settings = build_default_production_settings(
            resolved_fixture.to_operational_data(),
        )
        environment = _incident_proof_environment()
        catalog_entry = AgentCatalogEntry(
            catalog_entry_id=INCIDENT_INVESTIGATOR_CATALOG_ENTRY_ID,
            catalog_source=CatalogSourceIdentity(
                catalog_source_id=INCIDENT_INVESTIGATOR_CATALOG_SOURCE_ID,
                provider_kind=CatalogProviderKind.BUILTIN,
            ),
            display_name="AI Incident Investigator",
            package_id_line=INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID,
        )
        catalog_provider = _StaticCatalogProvider((catalog_entry,))
        metadata_provider = _MetadataProvider(
            {
                INCIDENT_INVESTIGATOR_METADATA_REF: AgentProjectMetadata(
                    distribution_package_id=INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID,
                    dependencies=(),
                ),
            }
        )
        application_composition = _build_application_composition(
            tmp_path,
            catalog_provider,
            metadata_provider,
        )
        base_composition = create_reference_production_process_composition()
        lifecycle_services = wire_reference_production_lifecycle_services(
            base_composition
        )
        capability_runtime = build_production_agent_capability_runtime(
            agent_platform_runtime=base_composition.agent_platform_runtime,
            application_composition=application_composition,
            lifecycle_services=lifecycle_services,
        )
        composition = ProductionProcessComposition(
            agent_platform_runtime=base_composition.agent_platform_runtime,
            agent_capability_runtime=capability_runtime,
        )
        governance = build_reference_production_control_plane_governance(environment)
        launcher = ReferenceProductionLifecycleLauncher(
            composition,
            services=lifecycle_services,
            mutation_authorization_boundary=governance.mutation_authorization_boundary,
            environment_tenant_resolver=governance.environment_tenant_resolver,
        )
        state = composition.agent_platform_runtime.distribution_state
        stores = composition.agent_platform_runtime.stores
        installation_store = InMemoryAgentInstallationStore(state)
        agent_manager_query = AgentManagerQueryService(
            catalog_provider=catalog_provider,
            installation_store=installation_store,
            binding_store=InMemoryApplicationAgentBindingStore(state),
            revision_store=stores.revision_store,
            serving_store=stores.serving_store,
            roster_builder=EffectiveRosterBuilder(installation_store),
        )
        manifest = _incident_production_manifest()
        return cls(
            runtime_root=tmp_path,
            composition=composition,
            launcher=launcher,
            governance=governance,
            catalog_provider=catalog_provider,
            manifest=manifest,
            environment=environment,
            agent_manager_query=agent_manager_query,
            production_settings=production_settings,
            fixture=resolved_fixture,
        )

    def build_context(self) -> ApplicationBuildContext:
        return ApplicationBuildContext.for_manifest(
            self.manifest,
            settings=self.production_settings,
            environment=self.environment,
            tool_registry=self.production_settings.composition.tool_registry,
        )

    def install_from_catalog(self, *, mutation_id: str = "mut-aipv1-install") -> None:
        principal = admin_test_principal()
        self.admin.install_agent(
            application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
            application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
            request=InstallAgentRequest(
                mutation_id=mutation_id,
                installation_id=INCIDENT_INVESTIGATOR_INSTALLATION_ID,
                installation_slot_id=INCIDENT_INVESTIGATOR_INSTALLATION_SLOT_ID,
                package_identity=AgentPackageIdentity(
                    distribution_package_id=INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID,
                    package_version=INCIDENT_INVESTIGATOR_PACKAGE_VERSION,
                    package_digest=INCIDENT_INVESTIGATOR_PACKAGE_DIGEST,
                ),
                artifact_store_ref=f"store://artifacts/{INCIDENT_INVESTIGATOR_INSTALLATION_ID}",
                trust_record=AgentInstallationTrustRecord(
                    qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
                    package_digest=INCIDENT_INVESTIGATOR_PACKAGE_DIGEST,
                    publisher_identity_ref="publisher:ai-incident-investigator",
                    source_provider_id=INCIDENT_INVESTIGATOR_CATALOG_SOURCE_ID,
                    trust_evidence_refs=(
                        AgentTrustEvidenceRef(
                            evidence_id="evidence:aipv-1",
                            kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                        ),
                    ),
                ),
                agent_project_metadata_ref=INCIDENT_INVESTIGATOR_METADATA_REF,
            ),
            principal=principal,
        )

    def bind_enabled_agent(self, *, mutation_id: str = "mut-aipv1-bind") -> int:
        principal = admin_test_principal()
        self.admin.bind_agent(
            application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
            application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
            request=BindAgentRequest(
                mutation_id=mutation_id,
                application_binding_id=INCIDENT_INVESTIGATOR_APPLICATION_BINDING_ID,
                logical_agent_id=INVESTIGATOR_AGENT_ID,
                installation_slot_id=INCIDENT_INVESTIGATOR_INSTALLATION_SLOT_ID,
                factory_reference=INCIDENT_INVESTIGATOR_FACTORY_REFERENCE,
                enablement=True,
            ),
            principal=principal,
        )
        binding = self.admin.list_bindings(
            application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
            application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
        ).bindings[0]
        self.admin.enable_binding(
            application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
            application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
            application_binding_id=INCIDENT_INVESTIGATOR_APPLICATION_BINDING_ID,
            request=SetAgentEnablementRequest(
                mutation_id=f"{mutation_id}:enable",
                expected_revision=binding.binding_revision,
            ),
            principal=principal,
        )
        enabled_binding = self.admin.list_bindings(
            application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
            application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
        ).bindings[0]
        return enabled_binding.binding_revision

    def inspect_effective_roster(self) -> EffectiveRosterView:
        roster = self.admin.inspect_effective_roster(
            application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
            application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
        )
        entry = next(
            item
            for item in roster.entries
            if item.logical_agent_id == INVESTIGATOR_AGENT_ID
        )
        assert entry.installation_slot_id == INCIDENT_INVESTIGATOR_INSTALLATION_SLOT_ID
        assert entry.effective_enablement is True
        assert entry.package_digest == INCIDENT_INVESTIGATOR_PACKAGE_DIGEST
        assert (
            entry.distribution_package_id
            == INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID
        )
        return roster

    def build_revision(
        self,
        *,
        mutation_id: str = "mut-aipv1-build",
    ) -> BuildRevisionResult:
        return self.admin.build_application_revision(
            application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
            application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
            request=BuildApplicationRevisionRequest(
                mutation_id=mutation_id,
                runtime_revision_id=INCIDENT_INVESTIGATOR_RUNTIME_REVISION_ID,
                application_release_id="rel-ai-incident-aipv1",
                platform_version="0.1.0",
                python_version="3.12",
                source_context_root="/tmp/src",
                output_root="/tmp/out",
                application_source_root=f"applications/{INCIDENT_INVESTIGATOR_APPLICATION_ID}",
                materialization_topology=MaterializationTopology.VENV_BUNDLE,
                repository_declaration=RepositoryDependencyDeclaration(
                    application_release_id="rel-ai-incident-aipv1",
                    direct_dependencies=(),
                ),
                resolver_algorithm_id="intergrax.ai-incident-aipv1-resolver",
                resolver_algorithm_version="1.0.0",
            ),
            principal=admin_test_principal(),
        )

    def register_projection_and_activate(self, built: BuildRevisionResult) -> str:
        assert built.artifact_locator is not None
        assert built.materialization_artifact_digest is not None
        assert built.materialized_runtime_lock_digest is not None
        assert built.runtime_graph_digest is not None

        bundle = build_production_registry_projection_input_bundle_for_revision(
            application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
            application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
            runtime_revision_id=built.runtime_revision_id,
            manifest=self.manifest,
            build_context=self.build_context(),
            authority=self.composition.agent_platform_runtime.registry_projection_authority,
        )
        serving_before = self.admin.inspect_serving(
            application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
            application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
        )
        activation_request = ActivateRuntimeRevisionRequest(
            mutation_id=f"mut-aipv1-activate:{built.runtime_revision_id}",
            runtime_revision_id=built.runtime_revision_id,
            artifact_locator=built.artifact_locator,
            expected_artifact_digest=built.materialization_artifact_digest,
            expected_serving_pointer_revision=serving_before.serving_pointer_revision,
            expected_prior_traffic_revision_id=serving_before.traffic_serving_revision_id,
        )
        self.launcher.deploy_and_activate(
            projection_input=bundle,
            activation_request=activation_request,
            principal=self.governance.principal,
            admission_mutation_id=reference_admission_mutation_id(
                built.runtime_revision_id,
            ),
        )
        serving = self.admin.inspect_serving(
            application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
            application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
        )
        assert serving.traffic_serving_revision_id == built.runtime_revision_id
        return serving.traffic_serving_revision_id

    def resolve_serving_projection(self) -> MaterializedRegistryProjection:
        serving = self.admin.inspect_serving(
            application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
            application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
        )
        assert serving.traffic_serving_revision_id is not None
        projection = bootstrap_production_registry_projection(
            application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
            application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
            stores=self.composition.agent_platform_runtime.stores,
        )
        assert (
            projection.evidence.runtime_revision_id
            == serving.traffic_serving_revision_id
        )
        return projection

    def resolve_registry_read(self) -> AgentRegistryRead:
        projection = self.resolve_serving_projection()
        registry = projection.agent_registry
        assert registry.has(INVESTIGATOR_AGENT_ID)
        return registry

    def attach_production_scenario_runtime(self) -> ScenarioRuntimeBundle:
        projection = self.resolve_serving_projection()
        registry = projection.agent_registry
        composition = self.production_settings.composition
        platform = build_scenario_runtime_from_environment(
            environment=self.environment,
            registry=registry,
            tenant_id=STANDALONE_SCENARIO_TENANT_ID,
            manifest=self.manifest,
            runtime_events_db_path=self.runtime_root / "runtime_events.db",
            trace_db_path=self.runtime_root / "trace.db",
            document_store=InMemoryDocumentStore(),
            settings=self.production_settings,
            runtime_mode=ScenarioRuntimeMode.PRODUCTION_ATTACHED,
            require_runtime_event_persistence=True,
            application_tool_registry=self.production_settings.composition.tool_registry,
        )
        composition.attach_platform(platform)
        composition.tool_registry = composition.platform.env_wiring.tool_wiring.registry
        investigator = registry.get(INVESTIGATOR_AGENT_ID)
        evidence_store = self.production_settings.evidence_store
        if evidence_store is None:
            raise RuntimeError(
                "incident_investigator_evidence_store_missing_from_factory_projection"
            )
        return ScenarioRuntimeBundle(
            operational_data=self.production_settings.operational_data,
            registry=composition.tool_registry,
            investigator=investigator,
            runtime_composition=composition,
            evidence_store=evidence_store,
            investigation_input=self.production_settings.investigation_input,
        )

    async def execute_incident_scenario(self) -> ScenarioExecutionResult:
        bundle = self.attach_production_scenario_runtime()
        prepare_incident_execution_runtime(bundle.runtime_composition)
        return await execute_resolved_skeleton(bundle)

    def run_happy_path(self) -> IncidentInvestigatorLifecycleProofResult:
        self.install_from_catalog()
        self.bind_enabled_agent()
        roster = self.inspect_effective_roster()
        built = self.build_revision()
        assert built.materialized_runtime_lock_digest is not None
        traffic_revision_id = self.register_projection_and_activate(built)
        registry = self.resolve_registry_read()
        assert isinstance(registry, AgentRegistryRead)
        assert registry.has(INVESTIGATOR_AGENT_ID)
        assert roster.effective_roster_revision_id is not None
        execution = asyncio.run(self.execute_incident_scenario())
        assert execution.outcome == OUTCOME_RESOLVED
        return IncidentInvestigatorLifecycleProofResult(
            package_digest=INCIDENT_INVESTIGATOR_PACKAGE_DIGEST,
            runtime_revision_id=built.runtime_revision_id,
            traffic_serving_revision_id=traffic_revision_id,
            execution_outcome=execution.outcome,
        )
