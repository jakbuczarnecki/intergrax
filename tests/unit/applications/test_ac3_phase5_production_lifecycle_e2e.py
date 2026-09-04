# © Artur Czarnecki. All rights reserved.

"""AC-3 Phase 5 — Reference Production V1 lifecycle E2E (N → N+1 serving switch).

Proves one ``ProductionProcessComposition`` owns the canonical chain:

    desired state → build_application_revision → historical authority
    → projection → PREPARE/READY → COMMIT → serving → Nexus

Uses the admin build path and production projection authority resolver on the
**same** store universe wired through ``ProductionAgentPlatformRuntime``.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.agent_distribution.admin_models import (
    BindAgentRequest,
    BuildApplicationRevisionRequest,
    BuildRevisionResult,
    InstallAgentRequest,
    SetAgentEnablementRequest,
    UpdateAgentBindingRequest,
)
from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
from intergrax.agent_distribution.agent_project_metadata import AgentProjectMetadata
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.dependency import RepositoryDependencyDeclaration
from intergrax.agent_distribution.deployment import DeploymentInstanceState
from intergrax.agent_distribution.effective_roster import (
    EffectiveRosterBuilder,
    InstalledAgentRequirementSetBuilder,
)
from intergrax.agent_distribution.errors import RuntimeReadinessError
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.in_memory_stores import (
    InMemoryAgentArtifactMetadataStore,
    InMemoryAgentInstallationStore,
    InMemoryApplicationAgentBindingStore,
)
from intergrax.agent_distribution.installation_service import InstallationService
from intergrax.agent_distribution.binding_service import BindingService
from intergrax.agent_distribution.materialization import (
    MaterializationInput,
    MaterializationOutput,
)
from intergrax.agent_distribution.materialization_service import (
    RuntimeMaterializationService,
)
from intergrax.agent_distribution.runtime_graph_service import (
    CandidateRuntimeGraphBuilder,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.runtime_context_staging import (
    RUNTIME_LOCK_MANIFEST_FILENAME,
    directory_content_digest,
)
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from intergrax.applications._shared.harness_host_runtime import (
    build_harness_host_runtime,
)
from intergrax.applications._shared.harness_registry_authority import (
    HarnessHostRegistryAuthorityError,
)
from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
    create_reference_production_process_composition,
)
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    ProductionRegistryProjectionInputError,
    build_production_registry_projection_for_revision,
    build_production_registry_projection_input_bundle_for_revision,
)
from intergrax.applications._shared.reference_production_governance_wiring import (
    ReferenceProductionControlPlaneGovernance,
    wire_governed_reference_production_launcher,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleLauncher,
)
from intergrax.applications._shared.registry_projection import RegistryProjectionError
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications._shared.harness_host_runtime_compat import resolve_harness_host_nexus_loop_legacy
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.core.qualification import QualificationStatus
from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
)
from testing_support.agent_platform_dependency_resolver import (
    make_identity_dependency_resolver,
)
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    _FakeCatalog,
    _MetadataProvider,
    admin_test_principal,
    allow_mutation_boundary,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        if agent_override is not None:
            return agent_override
        return adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.package_wiring.assert_manifest_package_closure",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.diagnostic_assembly_resolver.assert_diagnostic_assembly_valid",
        lambda *args, **kwargs: None,
    )


_APP = "app_a"
_ENV = "env-prod"
_RELEASE = "rel-1"
_LOGICAL_AGENT_ID = "search"
_SLOT_ID = "slot-search"
_BINDING_ID = "bind-search"
_META_REF_A = "meta://package-a"
_META_REF_B = "meta://package-b"
_PACKAGE_ID_A = "package-a"
_PACKAGE_ID_B = "package-b"
_DIGEST_A = "sha256:" + ("a" * 64)
_DIGEST_B = "sha256:" + ("b" * 64)
_FACTORY_REF_A = AgentBindingFactoryReference(
    factory_path="example_agent.factory_a.build_agent",
)
_FACTORY_REF_B = AgentBindingFactoryReference(
    factory_path="example_agent.factory_b.build_agent",
)
_CONFIG_A = {"model": {"temperature": 0.1}}
_CONFIG_B = {"model": {"temperature": 0.9}}
_REVISION_N = "rev-phase5-n"
_REVISION_N_PLUS_1 = "rev-phase5-n-plus-1"


@dataclass(frozen=True, slots=True)
class _DesiredState:
    installation_id: str
    package_digest: str
    distribution_package_id: str
    metadata_ref: str
    factory_reference: AgentBindingFactoryReference
    merged_config: dict[str, object]


_STATE_A = _DesiredState(
    installation_id="inst-phase5-a",
    package_digest=_DIGEST_A,
    distribution_package_id=_PACKAGE_ID_A,
    metadata_ref=_META_REF_A,
    factory_reference=_FACTORY_REF_A,
    merged_config=_CONFIG_A,
)
_STATE_B = _DesiredState(
    installation_id="inst-phase5-b",
    package_digest=_DIGEST_B,
    distribution_package_id=_PACKAGE_ID_B,
    metadata_ref=_META_REF_B,
    factory_reference=_FACTORY_REF_B,
    merged_config=_CONFIG_B,
)


class _Phase5VenvBundleMaterializer:
    topology = MaterializationTopology.VENV_BUNDLE
    materializer_id = "intergrax.phase5-lifecycle-proof"
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
            marker = entry.package_digest[-8:]
            (package_dir / f"{relative_module}.py").write_text(
                textwrap.dedent(
                    f"""
                    from echo.echo_agent import EchoAgent

                    MARKER = {marker!r}

                    def {function_name}(ctx, binding):
                        return EchoAgent()
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
            "{}\n", encoding="utf-8"
        )
        digest = directory_content_digest(artifact_root)
        return MaterializationOutput(
            materialization_artifact_digest=digest,
            artifact_locator=f"test://{artifact_root.resolve().as_posix()}",
            health_check_evidence_ref=f"test://{artifact_root.resolve().as_posix()}",
            runtime_graph_manifest_path=".intergrax-runtime-graph.json",
            topology=self.topology,
        )


@dataclass(frozen=True, slots=True)
class Phase5Harness:
    composition: ProductionProcessComposition
    launcher: ReferenceProductionLifecycleLauncher
    governance: ReferenceProductionControlPlaneGovernance
    admin: AgentPlatformAdminService
    manifest: ApplicationManifest
    environment: ApplicationEnvironmentProfile


def _manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id=_APP,
        name="AC-3 Phase 5 Lifecycle Proof",
        agents=[
            AgentBinding(
                import_path="echo.echo_agent.EchoAgent",
                contract_id=_LOGICAL_AGENT_ID,
                builder_key=_LOGICAL_AGENT_ID,
            ),
        ],
    )


def _trust_record(package_digest: str) -> AgentInstallationTrustRecord:
    return AgentInstallationTrustRecord(
        qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
        package_digest=package_digest,
        publisher_identity_ref="publisher:acme",
        source_provider_id="builtin",
        trust_evidence_refs=(
            AgentTrustEvidenceRef(
                evidence_id=f"evidence:{package_digest[-8:]}",
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
            ),
        ),
    )


def _build_request(
    revision_id: str, *, mutation_id: str
) -> BuildApplicationRevisionRequest:
    return BuildApplicationRevisionRequest(
        mutation_id=mutation_id,
        runtime_revision_id=revision_id,
        application_release_id=_RELEASE,
        platform_version="0.1.0",
        python_version="3.12",
        source_context_root="/tmp/src",
        output_root="/tmp/out",
        application_source_root=f"applications/{_APP}",
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        repository_declaration=RepositoryDependencyDeclaration(
            application_release_id=_RELEASE,
            direct_dependencies=(),
        ),
        resolver_algorithm_id="intergrax.phase5-resolver",
        resolver_algorithm_version="1.0.0",
    )


def _build_phase5_harness(tmp_path: Path) -> Phase5Harness:
    composition = create_reference_production_process_composition()
    environment = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    launcher, governance = wire_governed_reference_production_launcher(
        composition, environment
    )
    stores = composition.agent_platform_runtime.stores
    state = composition.agent_platform_runtime.distribution_state
    activation = launcher.services.activation_service

    metadata_provider = _MetadataProvider(
        {
            _META_REF_A: AgentProjectMetadata(
                distribution_package_id=_PACKAGE_ID_A,
                dependencies=(),
            ),
            _META_REF_B: AgentProjectMetadata(
                distribution_package_id=_PACKAGE_ID_B,
                dependencies=(),
            ),
        }
    )
    catalog = _FakeCatalog(
        [
            AgentCatalogEntry(
                catalog_entry_id="cat-search",
                catalog_source=CatalogSourceIdentity(
                    catalog_source_id="builtin-1",
                    provider_kind=CatalogProviderKind.BUILTIN,
                ),
                display_name="Search",
                package_id_line=_PACKAGE_ID_A,
            )
        ]
    )
    installation_store = InMemoryAgentInstallationStore(state)
    binding_store = InMemoryApplicationAgentBindingStore(state)
    artifact_store = InMemoryAgentArtifactMetadataStore(state)
    installation_service = InstallationService(installation_store)
    binding_service = BindingService(binding_store, installation_service)
    admin = AgentPlatformAdminService(
        installation_store=installation_store,
        binding_store=binding_store,
        revision_store=stores.revision_store,
        serving_store=stores.serving_store,
        deployment_instance_store=activation._deployment_instance_store,  # noqa: SLF001
        lock_store=stores.lock_store,
        materialization_store=stores.materialization_store,
        effective_roster_snapshot_store=stores.effective_roster_snapshot_store,
        effective_roster_authority=composition.agent_platform_runtime.effective_roster_authority,
        artifact_metadata_store=artifact_store,
        installation_service=installation_service,
        binding_service=binding_service,
        revision_service=RuntimeRevisionService(stores.revision_store),
        roster_builder=EffectiveRosterBuilder(installation_store),
        requirement_set_builder=InstalledAgentRequirementSetBuilder(artifact_store),
        activation_service=activation,
        graph_builder=CandidateRuntimeGraphBuilder(metadata_provider),
        materialization_service=RuntimeMaterializationService(
            {
                MaterializationTopology.VENV_BUNDLE: _Phase5VenvBundleMaterializer(
                    tmp_path
                )
            }
        ),
        metadata_provider=metadata_provider,
        catalog_provider=catalog,
        dependency_resolver=make_identity_dependency_resolver(),
        mutation_authorization_boundary=allow_mutation_boundary(),
        environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
            "tenant-test",
        ),
    )
    return Phase5Harness(
        composition=composition,
        launcher=launcher,
        governance=governance,
        admin=admin,
        manifest=_manifest(),
        environment=environment,
    )


def _install_desired_state(
    harness: Phase5Harness,
    desired: _DesiredState,
    *,
    mutation_id: str,
) -> None:
    harness.admin.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=InstallAgentRequest(
            mutation_id=mutation_id,
            installation_id=desired.installation_id,
            installation_slot_id=_SLOT_ID,
            package_identity=AgentPackageIdentity(
                distribution_package_id=desired.distribution_package_id,
                package_version="1.0.0",
                package_digest=desired.package_digest,
            ),
            artifact_store_ref=f"store://artifacts/{desired.installation_id}",
            trust_record=_trust_record(desired.package_digest),
            agent_project_metadata_ref=desired.metadata_ref,
        ),
        principal=admin_test_principal(),
    )
    harness.admin.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=BindAgentRequest(
            mutation_id=f"{mutation_id}:bind",
            application_binding_id=_BINDING_ID,
            logical_agent_id=_LOGICAL_AGENT_ID,
            installation_slot_id=_SLOT_ID,
            config=desired.merged_config,
            factory_reference=desired.factory_reference,
            enablement=True,
        ),
        principal=admin_test_principal(),
    )
    binding = harness.admin._binding_store.get_binding(_BINDING_ID)  # noqa: SLF001
    assert binding is not None
    harness.admin.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id=_BINDING_ID,
        request=SetAgentEnablementRequest(
            mutation_id=f"{mutation_id}:enable",
            expected_revision=binding.binding_revision,
        ),
        principal=admin_test_principal(),
    )


def _mutate_desired_state_a_to_b(harness: Phase5Harness) -> None:
    _install_desired_state(harness, _STATE_B, mutation_id="mut-phase5-install-b")
    binding = harness.admin._binding_store.get_binding(_BINDING_ID)  # noqa: SLF001
    assert binding is not None
    updated_binding = binding.model_copy(
        update={
            "factory_reference": _STATE_B.factory_reference,
            "binding_revision": binding.binding_revision + 1,
        }
    )
    harness.admin._binding_store.persist_binding(  # noqa: SLF001
        updated_binding,
        expected_revision=binding.binding_revision,
    )
    harness.admin.update_binding_config(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id=_BINDING_ID,
        request=UpdateAgentBindingRequest(
            mutation_id="mut-phase5-config-b",
            expected_revision=updated_binding.binding_revision,
            config=_STATE_B.merged_config,
        ),
        principal=admin_test_principal(),
    )
    harness.admin.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id=_BINDING_ID,
        request=SetAgentEnablementRequest(
            mutation_id="mut-phase5-enable-b",
            expected_revision=updated_binding.binding_revision + 1,
        ),
        principal=admin_test_principal(),
    )


def _build_revision(
    harness: Phase5Harness,
    revision_id: str,
    *,
    mutation_id: str,
) -> BuildRevisionResult:
    return harness.admin.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request(revision_id, mutation_id=mutation_id),
        principal=admin_test_principal(),
    )


def _projection_bundle(harness: Phase5Harness, built: BuildRevisionResult):
    return build_production_registry_projection_input_bundle_for_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=built.runtime_revision_id,
        manifest=harness.manifest,
        build_context=ApplicationBuildContext.for_manifest(harness.manifest),
        authority=harness.composition.agent_platform_runtime.registry_projection_authority,
    )


def _active_serving_revision(harness: Phase5Harness) -> str | None:
    serving = harness.composition.agent_platform_runtime.stores.serving_store.get_serving_record(
        _APP,
        _ENV,
    )
    if serving is None:
        return None
    return serving.traffic_serving_revision_id


def _prepare_ready(harness: Phase5Harness, built: BuildRevisionResult) -> None:
    bundle = _projection_bundle(harness, built)
    harness.launcher.services.projection_input_store.register(bundle)
    harness.launcher.services.activation_service.prepare_candidate(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=built.runtime_revision_id,
        artifact_locator=built.artifact_locator,
    )


def _commit_revision(
    harness: Phase5Harness,
    built: BuildRevisionResult,
    *,
    expected_prior_traffic_revision_id: str | None,
    expected_serving_pointer_revision: int,
) -> None:
    harness.launcher.services.activation_service.commit_activation(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=built.runtime_revision_id,
        expected_prior_traffic_revision_id=expected_prior_traffic_revision_id,
        expected_serving_pointer_revision=expected_serving_pointer_revision,
        expected_artifact_digest=built.materialization_artifact_digest,
    )


def _resolve_active_projection(harness: Phase5Harness):
    return bootstrap_production_registry_projection(
        application_id=_APP,
        application_environment_id=_ENV,
        stores=harness.composition.agent_platform_runtime.stores,
    )


def _nexus_registry_agent_ids(harness: Phase5Harness) -> tuple[str, ...]:
    projection = _resolve_active_projection(harness)
    runtime = build_harness_host_runtime(
        harness.manifest,
        harness.environment,
        registry_projection=projection,
        use_in_memory_trace=True,
        document_store=InMemoryDocumentStore(),
    )
    return tuple(resolve_harness_host_nexus_loop_legacy(runtime).registry.list_agent_ids())


def _assert_store_universe(harness: Phase5Harness) -> None:
    stores = harness.composition.agent_platform_runtime.stores
    services = harness.launcher.services
    activation = services.activation_service
    assert activation._revision_store is stores.revision_store  # noqa: SLF001
    assert activation._serving_store is stores.serving_store  # noqa: SLF001
    assert (
        services.projection_coordinator._projection_store
        is stores.registry_projection_store
    )  # noqa: SLF001
    assert harness.admin._revision_store is stores.revision_store  # noqa: SLF001
    assert harness.admin._materialization_store is stores.materialization_store  # noqa: SLF001
    assert harness.admin._lock_store is stores.lock_store  # noqa: SLF001
    assert (
        harness.admin._effective_roster_snapshot_store
        is stores.effective_roster_snapshot_store
    )  # noqa: SLF001


def test_phase5_initial_activation_build_ready_commit_serve_nexus(
    tmp_path: Path,
) -> None:
    harness = _build_phase5_harness(tmp_path)
    _assert_store_universe(harness)
    _install_desired_state(harness, _STATE_A, mutation_id="mut-phase5-setup-a")

    built = _build_revision(harness, _REVISION_N, mutation_id="mut-phase5-build-n")
    assert built.revision_state is RuntimeRevisionState.VALIDATED
    assert _active_serving_revision(harness) is None

    _prepare_ready(harness, built)
    assert _active_serving_revision(harness) is None

    _commit_revision(
        harness,
        built,
        expected_prior_traffic_revision_id=None,
        expected_serving_pointer_revision=0,
    )

    resolved = _resolve_active_projection(harness)
    assert resolved.evidence.runtime_revision_id == _REVISION_N
    assert resolved.agent_registry.list_agent_ids() == [_LOGICAL_AGENT_ID]
    assert _nexus_registry_agent_ids(harness) == (_LOGICAL_AGENT_ID,)
    assert _active_serving_revision(harness) == _REVISION_N


def test_phase5_n_plus_one_switch_preserves_n_until_commit(tmp_path: Path) -> None:
    harness = _build_phase5_harness(tmp_path)
    _install_desired_state(harness, _STATE_A, mutation_id="mut-phase5-setup-a")
    built_n = _build_revision(harness, _REVISION_N, mutation_id="mut-phase5-build-n")
    _prepare_ready(harness, built_n)
    _commit_revision(
        harness,
        built_n,
        expected_prior_traffic_revision_id=None,
        expected_serving_pointer_revision=0,
    )

    _mutate_desired_state_a_to_b(harness)
    built_n1 = _build_revision(
        harness,
        _REVISION_N_PLUS_1,
        mutation_id="mut-phase5-build-n-plus-1",
    )
    assert _active_serving_revision(harness) == _REVISION_N

    _prepare_ready(harness, built_n1)
    assert _active_serving_revision(harness) == _REVISION_N
    resolved_before = _resolve_active_projection(harness)
    assert resolved_before.evidence.runtime_revision_id == _REVISION_N
    assert _nexus_registry_agent_ids(harness) == (_LOGICAL_AGENT_ID,)

    _commit_revision(
        harness,
        built_n1,
        expected_prior_traffic_revision_id=_REVISION_N,
        expected_serving_pointer_revision=1,
    )

    resolved_after = _resolve_active_projection(harness)
    assert resolved_after.evidence.runtime_revision_id == _REVISION_N_PLUS_1
    assert resolved_after.evidence.materialization_artifact_digest == (
        built_n1.materialization_artifact_digest
    )
    assert _active_serving_revision(harness) == _REVISION_N_PLUS_1
    assert _nexus_registry_agent_ids(harness) == (_LOGICAL_AGENT_ID,)

    superseded = (
        harness.composition.agent_platform_runtime.stores.revision_store.get_revision(
            _REVISION_N
        )
    )
    assert superseded is not None
    assert superseded.revision_state is RuntimeRevisionState.SUPERSEDED

    prior_instance = harness.launcher.services.activation_service._deployment_instance_store.get_instance(  # noqa: SLF001
        _APP,
        _ENV,
        _REVISION_N,
    )
    assert prior_instance is not None
    assert prior_instance.instance_state is DeploymentInstanceState.DRAINING


def test_phase5_failed_prepare_preserves_active_n(tmp_path: Path) -> None:
    harness = _build_phase5_harness(tmp_path)
    _install_desired_state(harness, _STATE_A, mutation_id="mut-phase5-setup-a")
    built_n = _build_revision(harness, _REVISION_N, mutation_id="mut-phase5-build-n")
    _prepare_ready(harness, built_n)
    _commit_revision(
        harness,
        built_n,
        expected_prior_traffic_revision_id=None,
        expected_serving_pointer_revision=0,
    )

    _mutate_desired_state_a_to_b(harness)
    built_n1 = _build_revision(
        harness,
        _REVISION_N_PLUS_1,
        mutation_id="mut-phase5-build-n-plus-1",
    )
    harness.launcher.services.projection_input_store.register(
        _projection_bundle(harness, built_n1)
    )
    deployment_adapter = (
        harness.launcher.services.activation_service._deployment_adapter
    )  # noqa: SLF001
    deployment_adapter.fail_readiness(_REVISION_N_PLUS_1)

    with pytest.raises(
        RuntimeReadinessError, match="candidate readiness validation failed"
    ):
        harness.launcher.services.activation_service.prepare_candidate(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=_REVISION_N_PLUS_1,
            artifact_locator="artifact://wrong-digest",
        )

    assert _active_serving_revision(harness) == _REVISION_N
    resolved = _resolve_active_projection(harness)
    assert resolved.evidence.runtime_revision_id == _REVISION_N


def test_phase5_no_active_revision_fails_closed() -> None:
    harness = _build_phase5_harness(Path("/tmp/unused"))
    with pytest.raises(
        HarnessHostRegistryAuthorityError, match="no active traffic-serving"
    ):
        bootstrap_production_registry_projection(
            application_id=_APP,
            application_environment_id=_ENV,
            stores=harness.composition.agent_platform_runtime.stores,
        )


def test_phase5_ready_without_commit_does_not_serve(tmp_path: Path) -> None:
    harness = _build_phase5_harness(tmp_path)
    _install_desired_state(harness, _STATE_A, mutation_id="mut-phase5-setup-a")
    built = _build_revision(harness, _REVISION_N, mutation_id="mut-phase5-build-n")
    _prepare_ready(harness, built)

    with pytest.raises(
        HarnessHostRegistryAuthorityError, match="no active traffic-serving"
    ):
        bootstrap_production_registry_projection(
            application_id=_APP,
            application_environment_id=_ENV,
            stores=harness.composition.agent_platform_runtime.stores,
        )


def test_phase5_historical_n_resolvable_after_n_plus_one_commit(tmp_path: Path) -> None:
    harness = _build_phase5_harness(tmp_path)
    _install_desired_state(harness, _STATE_A, mutation_id="mut-phase5-setup-a")
    built_n = _build_revision(harness, _REVISION_N, mutation_id="mut-phase5-build-n")
    _prepare_ready(harness, built_n)
    _commit_revision(
        harness,
        built_n,
        expected_prior_traffic_revision_id=None,
        expected_serving_pointer_revision=0,
    )
    bundle_n = _projection_bundle(harness, built_n)
    roster_n_revision_id = bundle_n.effective_roster.effective_roster_revision_id

    _mutate_desired_state_a_to_b(harness)
    built_n1 = _build_revision(
        harness,
        _REVISION_N_PLUS_1,
        mutation_id="mut-phase5-build-n-plus-1",
    )
    _prepare_ready(harness, built_n1)
    _commit_revision(
        harness,
        built_n1,
        expected_prior_traffic_revision_id=_REVISION_N,
        expected_serving_pointer_revision=1,
    )

    authority = harness.composition.agent_platform_runtime.registry_projection_authority
    historical = authority.require_for_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=_REVISION_N,
    )
    assert historical.runtime_revision.runtime_revision_id == _REVISION_N
    assert (
        historical.effective_roster.effective_roster_revision_id == roster_n_revision_id
    )
    assert historical.effective_roster.entries[0].package_digest == _DIGEST_A
    assert historical.effective_roster.entries[0].merged_config == _CONFIG_A

    historical_projection = build_production_registry_projection_for_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=_REVISION_N,
        manifest=harness.manifest,
        build_context=ApplicationBuildContext.for_manifest(harness.manifest),
        authority=authority,
    )
    assert historical_projection.evidence.runtime_revision_id == _REVISION_N
    assert historical_projection.evidence.materialization_artifact_digest == (
        built_n.materialization_artifact_digest
    )

    current_bundle = _projection_bundle(harness, built_n1)
    assert current_bundle.effective_roster.entries[0].package_digest == _DIGEST_B
    assert current_bundle.effective_roster.entries[0].merged_config == _CONFIG_B


def test_phase5_commit_projection_failure_leaves_serving_on_n(tmp_path: Path) -> None:
    harness = _build_phase5_harness(tmp_path)
    _install_desired_state(harness, _STATE_A, mutation_id="mut-phase5-setup-a")
    built_n = _build_revision(harness, _REVISION_N, mutation_id="mut-phase5-build-n")
    _prepare_ready(harness, built_n)
    _commit_revision(
        harness,
        built_n,
        expected_prior_traffic_revision_id=None,
        expected_serving_pointer_revision=0,
    )

    _mutate_desired_state_a_to_b(harness)
    built_n1 = _build_revision(
        harness,
        _REVISION_N_PLUS_1,
        mutation_id="mut-phase5-build-n-plus-1",
    )
    harness.launcher.services.activation_service.prepare_candidate(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=_REVISION_N_PLUS_1,
        artifact_locator=built_n1.artifact_locator,
    )

    with pytest.raises(
        RegistryProjectionError, match="missing frozen projection inputs"
    ):
        harness.launcher.services.activation_service.commit_activation(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=_REVISION_N_PLUS_1,
            expected_prior_traffic_revision_id=_REVISION_N,
            expected_serving_pointer_revision=1,
            expected_artifact_digest=built_n1.materialization_artifact_digest,
        )

    assert _active_serving_revision(harness) == _REVISION_N
    resolved = _resolve_active_projection(harness)
    assert resolved.evidence.runtime_revision_id == _REVISION_N


def test_phase5_build_writes_into_composition_revision_store(tmp_path: Path) -> None:
    harness = _build_phase5_harness(tmp_path)
    _install_desired_state(harness, _STATE_A, mutation_id="mut-phase5-setup-a")
    built = _build_revision(harness, _REVISION_N, mutation_id="mut-phase5-build-n")

    stores = harness.composition.agent_platform_runtime.stores
    persisted = stores.revision_store.get_revision(built.runtime_revision_id)
    assert persisted is not None
    assert persisted.revision_state is RuntimeRevisionState.VALIDATED

    with pytest.raises(ProductionRegistryProjectionInputError):
        build_production_registry_projection_input_bundle_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id="rev-missing",
            manifest=harness.manifest,
            build_context=ApplicationBuildContext.for_manifest(harness.manifest),
            authority=harness.composition.agent_platform_runtime.registry_projection_authority,
        )
