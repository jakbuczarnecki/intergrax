# © Artur Czarnecki. All rights reserved.

"""Reusable agent platform admin stack for agent_distribution qualification harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.agent_distribution.activation import (
    ActivationService,
    FakeRuntimeServingProjectionCoordinator,
)
from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    BindAgentRequest,
    BuildApplicationRevisionRequest,
    BuildRevisionResult,
    InstallAgentRequest,
)
from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
from intergrax.agent_distribution.agent_project_metadata import AgentProjectMetadata
from intergrax.agent_distribution.binding_service import BindingService
from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
)
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter
from intergrax.agent_distribution.dependency import RepositoryDependencyDeclaration
from intergrax.agent_distribution.effective_roster import (
    EffectiveRosterBuilder,
    InstalledAgentRequirementSetBuilder,
)
from intergrax.agent_distribution.effective_roster_authority import (
    EffectiveRosterAuthorityService,
)
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentArtifactMetadataStore,
    InMemoryAgentInstallationStore,
    InMemoryApplicationAgentBindingStore,
    InMemoryApplicationEnvironmentActivationStore,
    InMemoryApplicationEnvironmentServingStore,
    InMemoryDeploymentInstanceStore,
    InMemoryEffectiveRosterSnapshotStore,
    InMemoryMaterializedRuntimeLockStore,
    InMemoryRuntimeMaterializationStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.installation_service import InstallationService
from intergrax.agent_distribution.materialization_service import (
    RuntimeMaterializationService,
)
from intergrax.agent_distribution.runtime_graph_service import (
    CandidateRuntimeGraphBuilder,
)
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from intergrax.core.qualification import QualificationStatus
from testing_support.agent_platform_admin_harness import (
    ADMIN_TEST_MATERIALIZATION_ARTIFACT_DIGEST,
    AgentProjectMetadataTestProvider,
    DeterministicAgentDistributionAdapter,
    FakeAgentCatalog,
    admin_test_principal,
    allow_mutation_boundary,
)
from testing_support.agent_platform_dependency_resolver import (
    make_identity_dependency_resolver,
)

QUALIFICATION_APPLICATION_ID = "app-a"
QUALIFICATION_ENVIRONMENT_ID = "env-prod"
QUALIFICATION_PACKAGE_DIGEST = "sha256:" + ("a" * 64)
QUALIFICATION_PACKAGE_ID = "intergrax-local-search-agent"
QUALIFICATION_METADATA_REF = "meta://search"
QUALIFICATION_ARTIFACT_DIGEST = ADMIN_TEST_MATERIALIZATION_ARTIFACT_DIGEST

_QUALIFIED_AT = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)
_PACKAGE = AgentPackageIdentity(
    distribution_package_id=QUALIFICATION_PACKAGE_ID,
    package_version="1.0.0",
    package_digest=QUALIFICATION_PACKAGE_DIGEST,
)


@dataclass
class AgentPlatformAdminQualificationStack:
    service: AgentPlatformAdminService
    state: AgentDistributionStoreState
    catalog: FakeAgentCatalog
    materialization_store: InMemoryRuntimeMaterializationStore
    effective_roster_snapshot_store: InMemoryEffectiveRosterSnapshotStore


def qualification_trust_record() -> AgentInstallationTrustRecord:
    return AgentInstallationTrustRecord(
        qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
        package_digest=QUALIFICATION_PACKAGE_DIGEST,
        publisher_identity_ref="publisher:acme",
        source_provider_id="builtin",
        qualification_qualified_at=_QUALIFIED_AT,
        trust_evidence_refs=(
            AgentTrustEvidenceRef(
                evidence_id="evidence:service:0",
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
            ),
        ),
    )


def qualification_install_request(
    mutation_id: str = "mut-install",
) -> InstallAgentRequest:
    return InstallAgentRequest(
        mutation_id=mutation_id,
        installation_id="inst-1",
        installation_slot_id="slot-search",
        package_identity=_PACKAGE,
        artifact_store_ref="store://artifacts/inst-1",
        trust_record=qualification_trust_record(),
        agent_project_metadata_ref=QUALIFICATION_METADATA_REF,
    )


def qualification_bind_request(mutation_id: str = "mut-bind") -> BindAgentRequest:
    return BindAgentRequest(
        mutation_id=mutation_id,
        application_binding_id="bind-search",
        logical_agent_id="researcher",
        installation_slot_id="slot-search",
    )


def qualification_build_request(
    revision_id: str,
    *,
    mutation_id: str = "mut-build",
) -> BuildApplicationRevisionRequest:
    return BuildApplicationRevisionRequest(
        mutation_id=mutation_id,
        runtime_revision_id=revision_id,
        application_release_id="rel-1",
        platform_version="0.1.0",
        python_version="3.12",
        source_context_root="/tmp/src",
        output_root="/tmp/out",
        application_source_root="applications/app-a",
        materialization_topology=MaterializationTopology.OCI_IMAGE,
        repository_declaration=RepositoryDependencyDeclaration(
            application_release_id="rel-1",
            direct_dependencies=(),
        ),
        resolver_algorithm_id="intergrax.test-resolver",
        resolver_algorithm_version="1.0.0",
    )


def qualification_activate_request(
    revision_id: str,
    *,
    pointer_revision: int = 0,
    prior_revision_id: str | None = None,
    mutation_id: str = "mut-activate",
) -> ActivateRuntimeRevisionRequest:
    return ActivateRuntimeRevisionRequest(
        mutation_id=mutation_id,
        runtime_revision_id=revision_id,
        artifact_locator="test://artifact",
        expected_artifact_digest=QUALIFICATION_ARTIFACT_DIGEST,
        expected_serving_pointer_revision=pointer_revision,
        expected_prior_traffic_revision_id=prior_revision_id,
    )


def build_agent_platform_admin_qualification_stack(
    *,
    with_catalog: bool = True,
) -> AgentPlatformAdminQualificationStack:
    state = AgentDistributionStoreState()
    installation_store = InMemoryAgentInstallationStore(state)
    binding_store = InMemoryApplicationAgentBindingStore(state)
    revision_store = InMemoryRuntimeRevisionStore(state)
    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    deployment_store = InMemoryDeploymentInstanceStore(state)
    lock_store = InMemoryMaterializedRuntimeLockStore(state)
    materialization_store = InMemoryRuntimeMaterializationStore(state)
    effective_roster_snapshot_store = InMemoryEffectiveRosterSnapshotStore(state)
    effective_roster_authority = EffectiveRosterAuthorityService(
        snapshot_store=effective_roster_snapshot_store,
    )
    artifact_store = InMemoryAgentArtifactMetadataStore(state)
    installation_service = InstallationService(installation_store)
    binding_service = BindingService(binding_store, installation_service)
    revision_service = RuntimeRevisionService(revision_store)
    metadata_provider = AgentProjectMetadataTestProvider(
        {
            QUALIFICATION_METADATA_REF: AgentProjectMetadata(
                distribution_package_id=QUALIFICATION_PACKAGE_ID,
                dependencies=(),
            )
        }
    )
    catalog = FakeAgentCatalog(
        [
            AgentCatalogEntry(
                catalog_entry_id="cat-researcher",
                catalog_source=CatalogSourceIdentity(
                    catalog_source_id="builtin-1",
                    provider_kind=CatalogProviderKind.BUILTIN,
                ),
                display_name="Researcher",
                package_id_line=QUALIFICATION_PACKAGE_ID,
            )
        ]
    )
    service = AgentPlatformAdminService(
        installation_store=installation_store,
        binding_store=binding_store,
        revision_store=revision_store,
        serving_store=serving_store,
        deployment_instance_store=deployment_store,
        lock_store=lock_store,
        materialization_store=materialization_store,
        effective_roster_snapshot_store=effective_roster_snapshot_store,
        effective_roster_authority=effective_roster_authority,
        artifact_metadata_store=artifact_store,
        installation_service=installation_service,
        binding_service=binding_service,
        revision_service=revision_service,
        roster_builder=EffectiveRosterBuilder(installation_store),
        requirement_set_builder=InstalledAgentRequirementSetBuilder(artifact_store),
        activation_service=ActivationService(
            revision_store=revision_store,
            deployment_instance_store=deployment_store,
            serving_store=serving_store,
            activation_store=InMemoryApplicationEnvironmentActivationStore(state),
            deployment_adapter=FakeInMemoryRuntimeDeploymentAdapter(),
            projection_coordinator=FakeRuntimeServingProjectionCoordinator(),
        ),
        graph_builder=CandidateRuntimeGraphBuilder(metadata_provider),
        materialization_service=RuntimeMaterializationService(
            {MaterializationTopology.OCI_IMAGE: DeterministicAgentDistributionAdapter()}
        ),
        metadata_provider=metadata_provider,
        catalog_provider=catalog if with_catalog else None,
        dependency_resolver=make_identity_dependency_resolver(),
        mutation_authorization_boundary=allow_mutation_boundary(),
        environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
            "tenant-test"
        ),
    )
    return AgentPlatformAdminQualificationStack(
        service=service,
        state=state,
        catalog=catalog,
        materialization_store=materialization_store,
        effective_roster_snapshot_store=effective_roster_snapshot_store,
    )


def qualification_build_revision(
    stack: AgentPlatformAdminQualificationStack,
    revision_id: str,
    *,
    mutation_id: str = "mut-build",
) -> BuildRevisionResult:
    return stack.service.build_application_revision(
        application_id=QUALIFICATION_APPLICATION_ID,
        application_environment_id=QUALIFICATION_ENVIRONMENT_ID,
        request=qualification_build_request(revision_id, mutation_id=mutation_id),
        principal=admin_test_principal(),
    )
