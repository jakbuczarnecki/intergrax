# © Artur Czarnecki. All rights reserved.

"""Reusable canonical agent lifecycle proof composition (Stage 15)."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    BindAgentRequest,
    BuildApplicationRevisionRequest,
    BuildRevisionResult,
    InstallAgentRequest,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.agent_manager_query_service import AgentManagerQueryService
from intergrax.agent_distribution.agent_project_metadata import AgentProjectMetadata
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
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
from intergrax.agent_distribution.materialization_service import RuntimeMaterializationService
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.agent_distribution.runtime_context_staging import (
    RUNTIME_LOCK_MANIFEST_FILENAME,
    directory_content_digest,
)
from intergrax.agent_distribution.task_capability_resolution import (
    build_deterministic_task_capability_resolver,
    build_task_capability_rule,
)
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.harness_registry_authority import (
    RegistryAssemblyMode,
    resolve_harness_host_registry,
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
    build_production_registry_projection_for_revision,
    build_production_registry_projection_input_bundle_for_revision,
)
from intergrax.applications._shared.registry_projection_input_bundle import (
    reference_admission_mutation_id,
)
from intergrax.applications._shared.reference_production_governance_wiring import (
    ReferenceProductionControlPlaneGovernance,
    build_reference_production_control_plane_governance,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleLauncher,
    wire_reference_production_lifecycle_services,
)
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.core.qualification import QualificationStatus
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.registry.presets import OTEL
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.agent_platform_dependency_resolver import make_identity_dependency_resolver
from testing_support.canonical_lifecycle_ping_agent import (
    CANONICAL_PING_CAPABILITY,
    CANONICAL_PING_CONTRACT_ID,
)
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    admin_test_principal,
    allow_mutation_boundary,
)

_DEFAULT_DIGEST = "sha256:" + ("a" * 64)


def _stage15_proof_environment(profile_id: str) -> ApplicationEnvironmentProfile:
    """Product defaults with observability backend required by host assembly validation."""
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=profile_id)
    integration = env.integration_profile
    assert integration is not None
    return env.model_copy(
        update={
            "capabilities": env.capabilities.model_copy(
                update={
                    "integrations": integration.model_copy(
                        update={"observability_backend": OTEL},
                    ),
                },
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class CanonicalLifecycleProofConfig:
    application_id: str
    environment_id: str
    logical_agent_id: str
    catalog_source_id: str
    catalog_entry_id: str
    catalog_provider_kind: CatalogProviderKind
    distribution_package_id: str
    package_version: str
    package_digest: str
    installation_slot_id: str
    application_binding_id: str
    installation_id: str
    metadata_ref: str
    factory_reference: AgentBindingFactoryReference
    revision_id: str
    test_input: str
    expected_output: str
    capability: str = CANONICAL_PING_CAPABILITY


@dataclass(frozen=True, slots=True)
class CanonicalLifecycleProofResult:
    catalog_source_id: str
    distribution_package_id: str
    package_digest: str
    installation_id: str
    application_binding_id: str
    binding_revision: int
    effective_roster_revision_id: str
    runtime_revision_id: str
    materialization_artifact_digest: str
    materialized_runtime_lock_digest: str
    runtime_graph_digest: str
    traffic_serving_revision_id: str
    execution_agent_id: str
    execution_answer: str


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

    def resolve_package(self, entry: AgentCatalogEntry, *, version_selector: str) -> object:
        del entry, version_selector
        raise NotImplementedError

    def health(self) -> None:
        return None


class _Stage15TrustRecordFactory:
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
            publisher_identity_ref="publisher:stage15",
            source_provider_id="builtin-stage15",
            trust_evidence_refs=(
                AgentTrustEvidenceRef(
                    evidence_id="evidence:stage15",
                    kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                ),
            ),
        )


def _build_application_composition(
    config: CanonicalLifecycleProofConfig,
    tmp_path: Path,
    catalog_provider: CatalogSourceProvider,
    metadata_provider: _MetadataProvider,
) -> AgentCapabilityApplicationComposition:
    return AgentCapabilityApplicationComposition(
        capability_resolver=build_deterministic_task_capability_resolver(
            rules=(
                build_task_capability_rule(
                    rule_id="rule.canonical.ping.v1",
                    task_kind="canonical.ping",
                    required=(CANONICAL_PING_CAPABILITY,),
                ),
            ),
        ),
        catalog_providers=(catalog_provider,),
        package_metadata_refs={config.distribution_package_id: config.metadata_ref},
        package_logical_agents={config.distribution_package_id: config.logical_agent_id},
        trust_record_factory=_Stage15TrustRecordFactory(),
        admin_config=ProductionAgentPlatformAdminConfig(
            metadata_provider=metadata_provider,
            materialization_service=RuntimeMaterializationService(
                {
                    MaterializationTopology.VENV_BUNDLE: _LifecycleVenvBundleMaterializer(
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
            application_id=config.application_id,
            application_environment_id=config.environment_id,
            application_release_id="rel-stage15",
            package_metadata_refs={config.distribution_package_id: config.metadata_ref},
            package_logical_agents={
                config.distribution_package_id: config.logical_agent_id,
            },
        ),
    )


class _LifecycleVenvBundleMaterializer:
    topology = MaterializationTopology.VENV_BUNDLE
    materializer_id = "intergrax.stage15-lifecycle-proof"
    materializer_version = "1.0.0"

    def __init__(self, artifact_root_base: Path) -> None:
        self._artifact_root_base = artifact_root_base

    def materialize(self, materialization_input: MaterializationInput) -> MaterializationOutput:
        revision_id = materialization_input.runtime_revision.runtime_revision_id
        artifact_root = self._artifact_root_base / revision_id
        site_packages = artifact_root / "site-packages"
        site_packages.mkdir(parents=True, exist_ok=True)
        package_dir = site_packages / "example_agent"
        package_dir.mkdir(exist_ok=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")

        for entry in materialization_input.effective_roster.entries:
            if not entry.effective_enablement:
                continue
            factory_reference = entry.factory_reference
            if factory_reference is None or factory_reference.factory_path is None:
                continue
            module_name, function_name = factory_reference.factory_path.rsplit(".", 1)
            relative_module = module_name.removeprefix("example_agent.")
            (package_dir / f"{relative_module}.py").write_text(
                textwrap.dedent(
                    f"""
                    from intergrax.agents.authoring.patterns.reflex import ReflexAgent
                    from intergrax.agents.authoring.patterns.types import (
                        AgentEvaluation,
                        CognitiveEvaluation,
                        Observation,
                        ReasoningResult,
                    )
                    from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
                    from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
                    from intergrax.contracts.agent_run_enums import CognitivePattern
                    from intergrax.contracts.agent_step_context import AgentStepContext
                    from intergrax.contracts.capability import CapabilityMatchResult
                    from intergrax.agents.reference_harness import (
                        build_lab_agent_runtime_context,
                        default_reference_harness,
                    )
                    from testing_support.builder import MeteringFakeLLMAdapter
                    from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
                    from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
                    from intergrax.runtime.task.task import TaskContext

                    _CONTRACT_ID = "canonical-ping-agent"
                    _CAPABILITY = "canonical.ping"
                    _PING_INPUT = "ping"
                    _PING_OUTPUT = "canonical-agent-ok"


                    class _MaterializedCanonicalPingAgent(ReflexAgent):
                        contract_id = _CONTRACT_ID
                        capabilities = (_CAPABILITY,)
                        agent_name = "Canonical Ping Agent"
                        agent_description = "Stage 15 materialized lifecycle proof agent."
                        agent_version = "1.0.0"
                        risk_level = AgentRiskLevel.LOW
                        max_steps = 1
                        cognitive_pattern = CognitivePattern.REFLEX

                        def get_contract(self) -> AgentContract:
                            return AgentContract(
                                id=self.contract_id,
                                name=self.agent_name,
                                description=self.agent_description,
                                version=self.agent_version,
                                capabilities=list(self.capabilities),
                                skills=[],
                                extra_tools=[],
                                risk_level=self.risk_level,
                                lifecycle_state=AgentLifecycleState.PRODUCTION,
                                production_eligible=True,
                                owner_team="platform",
                                owner_contact="harness@intergrax",
                                on_call_contact="harness@intergrax",
                                runbook_ref="docs/project/architecture/AGENT_DISTRIBUTION.md",
                                modality_profile_id="lab.default",
                                output_schema={{"type": "object", "properties": {{"answer": {{"type": "string"}}}}}},
                                validation_rules=["structured_output"],
                                max_steps=self.max_steps,
                                cognitive_pattern=self.cognitive_pattern,
                                pattern_version=self.pattern_version,
                            )

                        def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
                            capability = task_context.capability
                            if capability in (None, _CAPABILITY):
                                return CapabilityMatchResult(
                                    matched=True,
                                    agent_id=self.contract_id,
                                    matched_capabilities=[_CAPABILITY],
                                    score=1.0,
                                    rationale="canonical lifecycle proof agent",
                                )
                            return CapabilityMatchResult(
                                matched=False,
                                rationale="capability not supported",
                            )

                        def build_context(self, request: RuntimeRequest) -> RuntimeContext:
                            return build_lab_agent_runtime_context(
                                request=request,
                                llm_adapter=MeteringFakeLLMAdapter(),
                                harness=default_reference_harness(),
                            )

                        async def perceive(self, step_ctx: AgentStepContext) -> Observation:
                            message = self.read_run_input(step_ctx)
                            return Observation(summary=message or "")

                        async def reason(
                            self,
                            step_ctx: AgentStepContext,
                            observation: Observation,
                        ) -> ReasoningResult:
                            del step_ctx
                            return ReasoningResult(thought=observation.summary)

                        async def act(
                            self,
                            step_ctx: AgentStepContext,
                            reasoning: ReasoningResult,
                        ) -> dict[str, object]:
                            del step_ctx
                            if reasoning.thought == _PING_INPUT:
                                return {{"summary": _PING_OUTPUT, "answer": _PING_OUTPUT}}
                            return {{
                                "summary": f"unexpected-input:{{reasoning.thought}}",
                                "answer": f"unexpected-input:{{reasoning.thought}}",
                            }}

                        def evaluate(
                            self,
                            step_ctx: AgentStepContext,
                            output: dict[str, object],
                        ) -> AgentEvaluation:
                            del step_ctx, output
                            return AgentEvaluation(
                                verdict=CognitiveEvaluation.COMPLETE,
                                reason="canonical_ping_goal_met",
                            )


                    def {function_name}(ctx, binding):
                        del ctx, binding
                        return _MaterializedCanonicalPingAgent()
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


def default_stage15_proof_config(
    *,
    catalog_provider_kind: CatalogProviderKind = CatalogProviderKind.BUILTIN,
    catalog_source_id: str = "builtin-stage15",
) -> CanonicalLifecycleProofConfig:
    return CanonicalLifecycleProofConfig(
        application_id="stage15_app",
        environment_id="env_stage15",
        logical_agent_id=CANONICAL_PING_CONTRACT_ID,
        catalog_source_id=catalog_source_id,
        catalog_entry_id="cat-canonical-ping",
        catalog_provider_kind=catalog_provider_kind,
        distribution_package_id="intergrax-canonical-ping-agent",
        package_version="1.0.0",
        package_digest=_DEFAULT_DIGEST,
        installation_slot_id="slot-canonical-ping",
        application_binding_id="bind-canonical-ping",
        installation_id="inst-canonical-ping",
        metadata_ref="meta://canonical-ping",
        factory_reference=AgentBindingFactoryReference(
            factory_path="example_agent.factory.build_agent",
        ),
        revision_id="rev-stage15-canonical",
        test_input="ping",
        expected_output="canonical-agent-ok",
    )


@dataclass(frozen=True, slots=True)
class CanonicalAgentLifecycleProofStack:
    config: CanonicalLifecycleProofConfig
    runtime_root: Path
    composition: ProductionProcessComposition
    launcher: ReferenceProductionLifecycleLauncher
    governance: ReferenceProductionControlPlaneGovernance
    catalog_provider: CatalogSourceProvider
    manifest: ApplicationManifest
    environment: ApplicationEnvironmentProfile
    agent_manager_query: AgentManagerQueryService

    @property
    def admin(self) -> AgentPlatformAdminService:
        capability_runtime = self.composition.agent_capability_runtime
        assert capability_runtime is not None
        return capability_runtime.admin_service

    @classmethod
    def build(
        cls,
        tmp_path: Path,
        config: CanonicalLifecycleProofConfig | None = None,
    ) -> CanonicalAgentLifecycleProofStack:
        resolved = config or default_stage15_proof_config()
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
        base_composition = create_reference_production_process_composition()
        lifecycle_services = wire_reference_production_lifecycle_services(base_composition)
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
        manifest = ApplicationManifest.lab(
            app_id=resolved.application_id,
            name="Stage 15 Canonical Lifecycle Proof",
            agents=[
                AgentBinding(
                    contract_id=resolved.logical_agent_id,
                    builder_key=resolved.logical_agent_id,
                ),
            ],
        )
        return cls(
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

    def discover_catalog_entry(self) -> AgentCatalogEntry:
        entries = self.catalog_provider.list_entries()
        assert len(entries) == 1
        entry = entries[0]
        assert entry.catalog_source.catalog_source_id == self.config.catalog_source_id
        assert entry.package_id_line == self.config.distribution_package_id
        return entry

    def install_from_catalog(self, *, mutation_id: str = "mut-stage15-install") -> None:
        catalog_entry = self.discover_catalog_entry()
        assert catalog_entry.package_id_line == self.config.distribution_package_id
        principal = admin_test_principal()
        self.admin.install_agent(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
            request=InstallAgentRequest(
                mutation_id=mutation_id,
                installation_id=self.config.installation_id,
                installation_slot_id=self.config.installation_slot_id,
                package_identity=AgentPackageIdentity(
                    distribution_package_id=self.config.distribution_package_id,
                    package_version=self.config.package_version,
                    package_digest=self.config.package_digest,
                ),
                artifact_store_ref=f"store://artifacts/{self.config.installation_id}",
                trust_record=AgentInstallationTrustRecord(
                    qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
                    package_digest=self.config.package_digest,
                    publisher_identity_ref="publisher:stage15",
                    source_provider_id=catalog_entry.catalog_source.catalog_source_id,
                    trust_evidence_refs=(
                        AgentTrustEvidenceRef(
                            evidence_id="evidence:stage15",
                            kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                        ),
                    ),
                ),
                agent_project_metadata_ref=self.config.metadata_ref,
            ),
            principal=principal,
        )

    def bind_enabled_agent(self, *, mutation_id: str = "mut-stage15-bind") -> int:
        principal = admin_test_principal()
        self.admin.bind_agent(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
            request=BindAgentRequest(
                mutation_id=mutation_id,
                application_binding_id=self.config.application_binding_id,
                logical_agent_id=self.config.logical_agent_id,
                installation_slot_id=self.config.installation_slot_id,
                factory_reference=self.config.factory_reference,
                enablement=True,
            ),
            principal=principal,
        )
        binding = self.admin.list_bindings(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
        ).bindings[0]
        self.admin.enable_binding(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
            application_binding_id=self.config.application_binding_id,
            request=SetAgentEnablementRequest(
                mutation_id=f"{mutation_id}:enable",
                expected_revision=binding.binding_revision,
            ),
            principal=principal,
        )
        enabled_binding = self.admin.list_bindings(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
        ).bindings[0]
        return enabled_binding.binding_revision

    def inspect_effective_roster(self):
        roster = self.admin.inspect_effective_roster(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
        )
        entry = next(
            item
            for item in roster.entries
            if item.logical_agent_id == self.config.logical_agent_id
        )
        assert entry.installation_slot_id == self.config.installation_slot_id
        assert entry.effective_enablement is True
        assert entry.package_digest == self.config.package_digest
        assert entry.distribution_package_id == self.config.distribution_package_id
        return roster

    def build_revision(
        self,
        *,
        revision_id: str | None = None,
        mutation_id: str = "mut-stage15-build",
    ) -> BuildRevisionResult:
        resolved_revision_id = revision_id or self.config.revision_id
        return self.admin.build_application_revision(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
            request=BuildApplicationRevisionRequest(
                mutation_id=mutation_id,
                runtime_revision_id=resolved_revision_id,
                application_release_id="rel-stage15",
                platform_version="0.1.0",
                python_version="3.12",
                source_context_root="/tmp/src",
                output_root="/tmp/out",
                application_source_root=f"applications/{self.config.application_id}",
                materialization_topology=MaterializationTopology.VENV_BUNDLE,
                repository_declaration=RepositoryDependencyDeclaration(
                    application_release_id="rel-stage15",
                    direct_dependencies=(),
                ),
                resolver_algorithm_id="intergrax.stage15-resolver",
                resolver_algorithm_version="1.0.0",
            ),
            principal=admin_test_principal(),
        )

    def register_projection_and_activate(self, built: BuildRevisionResult) -> str:
        bundle = build_production_registry_projection_input_bundle_for_revision(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
            runtime_revision_id=built.runtime_revision_id,
            manifest=self.manifest,
            build_context=ApplicationBuildContext.for_manifest(self.manifest),
            authority=self.composition.agent_platform_runtime.registry_projection_authority,
        )
        result = self.launcher.deploy_and_activate(
            bundle,
            ActivateRuntimeRevisionRequest(
                mutation_id=f"mut-stage15-activate:{built.runtime_revision_id}",
                runtime_revision_id=built.runtime_revision_id,
                artifact_locator=built.artifact_locator or "test://artifact",
                expected_artifact_digest=built.materialization_artifact_digest or "",
                expected_serving_pointer_revision=0,
            ),
            principal=self.governance.principal,
            admission_mutation_id=reference_admission_mutation_id(
                built.runtime_revision_id,
            ),
        )
        assert result.runtime_revision_id == built.runtime_revision_id
        assert result.application_id == self.config.application_id
        assert result.application_environment_id == self.config.environment_id
        assert (
            result.resolved_projection.evidence.runtime_revision_id
            == built.runtime_revision_id
        )
        serving = self.admin.inspect_serving(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
        )
        assert serving.traffic_serving_revision_id == built.runtime_revision_id
        return serving.traffic_serving_revision_id or built.runtime_revision_id

    def resolve_serving_projection(self) -> MaterializedRegistryProjection:
        serving = self.admin.inspect_serving(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
        )
        assert serving.traffic_serving_revision_id is not None
        projection = bootstrap_production_registry_projection(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
            stores=self.composition.agent_platform_runtime.stores,
        )
        assert projection.evidence.runtime_revision_id == serving.traffic_serving_revision_id
        return projection

    def resolve_registry_read(self) -> AgentRegistryRead:
        projection = self.resolve_serving_projection()
        registry, evidence = resolve_harness_host_registry(
            manifest=self.manifest,
            build_context=ApplicationBuildContext.for_manifest(self.manifest),
            environment=self.environment,
            assembly_mode=RegistryAssemblyMode.REVISION_BOUND,
            registry_projection=projection,
        )
        assert evidence is projection.evidence
        assert registry.has(self.config.logical_agent_id)
        return registry

    def resolve_projection_for_revision(
        self,
        runtime_revision_id: str,
    ) -> MaterializedRegistryProjection:
        return build_production_registry_projection_for_revision(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
            runtime_revision_id=runtime_revision_id,
            manifest=self.manifest,
            build_context=ApplicationBuildContext.for_manifest(self.manifest),
            authority=self.composition.agent_platform_runtime.registry_projection_authority,
        )

    async def execute_canonical(self) -> tuple[str, str]:
        projection = self.resolve_serving_projection()
        host_runtime = build_harness_host_runtime(
            self.manifest,
            self.environment,
            registry_projection=projection,
            trace_db_path=self.runtime_root / "trace.db",
            runtime_events_db_path=self.runtime_root / "runtime_events.db",
            document_store=InMemoryDocumentStore(),
        )
        task = Task(
            tenant_id="tenant-test",
            user_id="proof-user",
            message=self.config.test_input,
            agent_id=self.config.logical_agent_id,
            context=TaskContext(capability=self.config.capability),
        )
        result = await host_runtime.execution.execute(task)
        assert result.agent_id == self.config.logical_agent_id
        assert result.answer == self.config.expected_output
        return result.agent_id or "", result.answer or ""

    def run_happy_path(self) -> CanonicalLifecycleProofResult:
        self.install_from_catalog()
        binding_revision = self.bind_enabled_agent()
        roster = self.inspect_effective_roster()
        built = self.build_revision()
        traffic_revision_id = self.register_projection_and_activate(built)
        projection = self.resolve_serving_projection()
        registry = self.resolve_registry_read()
        assert isinstance(registry, AgentRegistryRead)
        assert registry.has(self.config.logical_agent_id)
        materialization = (
            self.composition.agent_platform_runtime.stores.materialization_store.get_by_revision(
                built.runtime_revision_id,
            )
        )
        assert materialization is not None
        assert materialization.runtime_revision_id == built.runtime_revision_id
        assert materialization.materialization_artifact_digest == built.materialization_artifact_digest
        return CanonicalLifecycleProofResult(
            catalog_source_id=self.config.catalog_source_id,
            distribution_package_id=self.config.distribution_package_id,
            package_digest=self.config.package_digest,
            installation_id=self.config.installation_id,
            application_binding_id=self.config.application_binding_id,
            binding_revision=binding_revision,
            effective_roster_revision_id=roster.effective_roster_revision_id or "",
            runtime_revision_id=built.runtime_revision_id,
            materialization_artifact_digest=built.materialization_artifact_digest or "",
            materialized_runtime_lock_digest=built.materialized_runtime_lock_digest or "",
            runtime_graph_digest=built.runtime_graph_digest or "",
            traffic_serving_revision_id=traffic_revision_id,
            execution_agent_id="",
            execution_answer="",
        )


__all__ = [
    "CanonicalAgentLifecycleProofStack",
    "CanonicalLifecycleProofConfig",
    "CanonicalLifecycleProofResult",
    "default_stage15_proof_config",
]
