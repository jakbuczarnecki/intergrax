# © Artur Czarnecki. All rights reserved.

"""Shared in-memory harness for the PDC-X B-D1 lifecycle qualification.

Builds the real ``AgentPlatformAdminService`` over the in-memory stores (no mocks of
the unit under test). Packages are digest-pinned releases of one package line with
real ``AgentContract`` authority records so the upgrade compatibility check runs on
genuine data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
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
    SetAgentEnablementRequest,
    UpgradeAgentRequest,
)
from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
from intergrax.agent_distribution.agent_contract_authority import (
    PackageAgentContractAuthorityRecord,
)
from intergrax.agent_distribution.agent_project_metadata import (
    AgentPackageContractDeclaration,
    AgentProjectMetadata,
)
from intergrax.agent_distribution.binding import AgentBindingPolicyOverrides
from intergrax.agent_distribution.binding_service import BindingService
from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
)
from intergrax.agent_distribution.dependency import RepositoryDependencyDeclaration
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter
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
    AgentPackageTrustPolicy,
    AgentPackageTrustRevocationState,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.core.qualification import QualificationStatus
from intergrax.skills.core.contracts import SkillManifest
from testing_support.agent_platform_admin_harness import (
    ADMIN_TEST_MATERIALIZATION_ARTIFACT_DIGEST,
    AgentProjectMetadataTestProvider,
    DeterministicAgentDistributionAdapter,
    admin_test_principal,
    allow_mutation_boundary,
)
from testing_support.agent_platform_dependency_resolver import (
    make_identity_dependency_resolver,
)

APP = "app-a"
APP_B = "app-b"
ENV = "env-prod"
TENANT = "tenant-test"
OTHER_TENANT = "tenant-other"
PACKAGE_LINE = "intergrax-local-search-agent"
SLOT = "slot-search"
CONTRACT_ID = "researcher"
ARTIFACT_DIGEST = ADMIN_TEST_MATERIALIZATION_ARTIFACT_DIGEST
EVALUATED_AT = datetime(2026, 8, 5, 12, 0, 0, tzinfo=UTC)

DIGESTS = {
    "1.0.0": "sha256:" + ("a" * 64),
    "1.1.0": "sha256:" + ("b" * 64),
    "1.2.0": "sha256:" + ("c" * 64),
    "2.0.0": "sha256:" + ("e" * 64),
    "0.9.0": "sha256:" + ("f" * 64),
}

OTHER_TENANT_PRINCIPAL = RequestIdentity(
    tenant_id=OTHER_TENANT,
    user_id="other-admin",
    principal_type=PrincipalType.USER,
    auth_subject="other-admin",
)


def principal() -> RequestIdentity:
    return admin_test_principal()


def package(version: str, *, line: str = PACKAGE_LINE) -> AgentPackageIdentity:
    return AgentPackageIdentity(
        distribution_package_id=line,
        package_version=version,
        package_digest=DIGESTS[version],
    )


def meta_ref(version: str) -> str:
    return f"meta://search/{version}"


def installation_id(version: str) -> str:
    return f"inst-{version}"


def trust(version: str) -> AgentInstallationTrustRecord:
    return AgentInstallationTrustRecord(
        qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
        package_digest=DIGESTS[version],
        publisher_identity_ref="publisher:acme",
        source_provider_id="builtin",
        trust_evidence_refs=(
            AgentTrustEvidenceRef(
                evidence_id=f"evidence:{version}",
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
            ),
        ),
    )


def contract(
    *,
    version: str = "1.0.0",
    capabilities: tuple[str, ...] = ("search",),
    tools: tuple[str, ...] = ("tool.search",),
    contract_id: str = CONTRACT_ID,
) -> AgentContract:
    skills = (
        [
            SkillManifest(
                skill_id="skill.search",
                description="search skill",
                tool_ids=tools,
            )
        ]
        if tools
        else []
    )
    return AgentContract(
        id=contract_id,
        name=contract_id,
        description="researcher agent",
        version=version,
        capabilities=list(capabilities),
        skills=skills,
    )


def declaration(contract_value: AgentContract) -> AgentPackageContractDeclaration:
    return AgentPackageContractDeclaration(
        contract_id=contract_value.id,
        contract_version=contract_value.version,
        capabilities=tuple(contract_value.capabilities),
        skill_ids=tuple(skill.skill_id for skill in contract_value.skills),
    )


def project_metadata(
    package_version: str,
    contracts: tuple[AgentContract, ...],
    *,
    line: str = PACKAGE_LINE,
) -> AgentProjectMetadata:
    return AgentProjectMetadata(
        distribution_package_id=line,
        package_version=package_version,
        dependencies=(),
        declared_contracts=tuple(declaration(item) for item in contracts),
    )


def authority_records(
    package_version: str,
    contracts: tuple[AgentContract, ...],
    *,
    line: str = PACKAGE_LINE,
) -> tuple[PackageAgentContractAuthorityRecord, ...]:
    return tuple(
        PackageAgentContractAuthorityRecord.from_validated_contract(
            package_digest=DIGESTS[package_version],
            distribution_package_id=line,
            artifact_store_ref=f"store://artifacts/{installation_id(package_version)}",
            agent_project_metadata_ref=meta_ref(package_version),
            contract=item,
        )
        for item in contracts
    )


@dataclass
class Release:
    """One package release: identity, declared contracts and its authority records."""

    version: str
    contracts: tuple[AgentContract, ...]
    line: str = PACKAGE_LINE

    @property
    def identity(self) -> AgentPackageIdentity:
        return package(self.version, line=self.line)

    def metadata(self) -> AgentProjectMetadata:
        return project_metadata(self.version, self.contracts, line=self.line)

    def records(self) -> tuple[PackageAgentContractAuthorityRecord, ...]:
        return authority_records(self.version, self.contracts, line=self.line)

    def install_request(self, *, mutation_id: str | None = None, slot: str = SLOT) -> InstallAgentRequest:
        return InstallAgentRequest(
            mutation_id=mutation_id or f"mut-install-{self.version}",
            installation_id=installation_id(self.version),
            installation_slot_id=slot,
            package_identity=self.identity,
            artifact_store_ref=f"store://artifacts/{installation_id(self.version)}",
            trust_record=trust(self.version),
            agent_project_metadata_ref=meta_ref(self.version),
            package_contract_authority=self.records(),
        )

    def upgrade_request(
        self,
        *,
        expected_active: str,
        mutation_id: str | None = None,
        slot: str = SLOT,
    ) -> UpgradeAgentRequest:
        return UpgradeAgentRequest(
            mutation_id=mutation_id or f"mut-upgrade-{self.version}",
            installation_id=installation_id(self.version),
            installation_slot_id=slot,
            package_identity=self.identity,
            artifact_store_ref=f"store://artifacts/{installation_id(self.version)}",
            trust_record=trust(self.version),
            agent_project_metadata_ref=meta_ref(self.version),
            package_contract_authority=self.records(),
            expected_active_installation_id=expected_active,
        )


def release(
    version: str,
    *,
    contract_version: str = "1.0.0",
    capabilities: tuple[str, ...] = ("search",),
    tools: tuple[str, ...] = ("tool.search",),
    line: str = PACKAGE_LINE,
) -> Release:
    return Release(
        version=version,
        contracts=(contract(version=contract_version, capabilities=capabilities, tools=tools),),
        line=line,
    )


class MutableRevocation:
    """Revocation snapshot the test can flip while the stack is running."""

    def __init__(self) -> None:
        self.state = AgentPackageTrustRevocationState()

    def __call__(self) -> AgentPackageTrustRevocationState:
        return self.state


@dataclass
class D1Stack:
    service: AgentPlatformAdminService
    state: AgentDistributionStoreState
    adapter: FakeInMemoryRuntimeDeploymentAdapter
    revocation: MutableRevocation
    installation_service: InstallationService
    binding_service: BindingService
    installation_store: InMemoryAgentInstallationStore
    serving_store: InMemoryApplicationEnvironmentServingStore
    revision_store: InMemoryRuntimeRevisionStore
    deployment_store: InMemoryDeploymentInstanceStore
    roster_authority: EffectiveRosterAuthorityService
    activation: ActivationService
    revision_service: RuntimeRevisionService
    metadata_provider: AgentProjectMetadataTestProvider
    releases: dict[str, Release] = field(default_factory=dict)

    def register(self, *releases: Release) -> None:
        for item in releases:
            self.releases[item.version] = item
            self.metadata_provider._records[meta_ref(item.version)] = item.metadata()


def build_stack(
    *,
    tenant: str = TENANT,
    policy_source: Callable[[], AgentPackageTrustPolicy] | None = None,
    time_source: Callable[[], datetime] | None = None,
    installation_store_factory: Callable[[AgentDistributionStoreState], InMemoryAgentInstallationStore] | None = None,
) -> D1Stack:
    state = AgentDistributionStoreState()
    installation_store = (
        installation_store_factory(state)
        if installation_store_factory is not None
        else InMemoryAgentInstallationStore(state)
    )
    binding_store = InMemoryApplicationAgentBindingStore(state)
    revision_store = InMemoryRuntimeRevisionStore(state)
    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    deployment_store = InMemoryDeploymentInstanceStore(state)
    lock_store = InMemoryMaterializedRuntimeLockStore(state)
    materialization_store = InMemoryRuntimeMaterializationStore(state)
    roster_snapshot_store = InMemoryEffectiveRosterSnapshotStore(state)
    roster_authority = EffectiveRosterAuthorityService(snapshot_store=roster_snapshot_store)
    artifact_store = InMemoryAgentArtifactMetadataStore(state)
    installation_service = InstallationService(installation_store)
    binding_service = BindingService(binding_store, installation_service)
    revision_service = RuntimeRevisionService(revision_store)
    metadata_provider = AgentProjectMetadataTestProvider({})
    adapter = FakeInMemoryRuntimeDeploymentAdapter()
    activation = ActivationService(
        revision_store=revision_store,
        deployment_instance_store=deployment_store,
        serving_store=serving_store,
        activation_store=InMemoryApplicationEnvironmentActivationStore(state),
        deployment_adapter=adapter,
        projection_coordinator=FakeRuntimeServingProjectionCoordinator(),
        materialization_store=materialization_store,
    )
    revocation = MutableRevocation()
    service = AgentPlatformAdminService(
        installation_store=installation_store,
        binding_store=binding_store,
        revision_store=revision_store,
        serving_store=serving_store,
        deployment_instance_store=deployment_store,
        lock_store=lock_store,
        materialization_store=materialization_store,
        effective_roster_snapshot_store=roster_snapshot_store,
        effective_roster_authority=roster_authority,
        artifact_metadata_store=artifact_store,
        installation_service=installation_service,
        binding_service=binding_service,
        revision_service=revision_service,
        roster_builder=EffectiveRosterBuilder(installation_store),
        requirement_set_builder=InstalledAgentRequirementSetBuilder(artifact_store),
        activation_service=activation,
        graph_builder=CandidateRuntimeGraphBuilder(metadata_provider),
        materialization_service=RuntimeMaterializationService(
            {MaterializationTopology.OCI_IMAGE: DeterministicAgentDistributionAdapter()}
        ),
        metadata_provider=metadata_provider,
        dependency_resolver=make_identity_dependency_resolver(),
        mutation_authorization_boundary=allow_mutation_boundary(),
        environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(tenant),
        package_trust_revocation_state_source=revocation,
        package_trust_policy_source=policy_source,
        package_trust_evaluation_time_source=time_source or (lambda: EVALUATED_AT),
    )
    return D1Stack(
        service=service,
        state=state,
        adapter=adapter,
        revocation=revocation,
        installation_service=installation_service,
        binding_service=binding_service,
        installation_store=installation_store,
        serving_store=serving_store,
        revision_store=revision_store,
        deployment_store=deployment_store,
        roster_authority=roster_authority,
        activation=activation,
        revision_service=revision_service,
        metadata_provider=metadata_provider,
    )


def bind_request(
    *,
    binding_id: str = "bind-search",
    logical_agent_id: str = CONTRACT_ID,
    slot: str = SLOT,
    tool_allowlist: tuple[str, ...] = (),
    mutation_id: str = "mut-bind",
) -> BindAgentRequest:
    return BindAgentRequest(
        mutation_id=mutation_id,
        application_binding_id=binding_id,
        logical_agent_id=logical_agent_id,
        installation_slot_id=slot,
        policy_overrides=(AgentBindingPolicyOverrides(tool_allowlist=tool_allowlist) if tool_allowlist else None),
    )


def install_bind_enable(
    stack: D1Stack,
    first: Release,
    *,
    tool_allowlist: tuple[str, ...] = (),
) -> None:
    """Install ``first`` into the slot, bind and enable the agent (desired state)."""
    stack.register(first)
    stack.service.install_agent(
        application_id=APP,
        application_environment_id=ENV,
        request=first.install_request(),
        principal=principal(),
    )
    stack.service.bind_agent(
        application_id=APP,
        application_environment_id=ENV,
        request=bind_request(tool_allowlist=tool_allowlist),
        principal=principal(),
    )
    stack.service.enable_binding(
        application_id=APP,
        application_environment_id=ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(mutation_id="mut-enable", expected_revision=0),
        principal=principal(),
    )


def build_request(revision_id: str, *, mutation_id: str | None = None) -> BuildApplicationRevisionRequest:
    return BuildApplicationRevisionRequest(
        mutation_id=mutation_id or f"mut-build-{revision_id}",
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


def build_revision(stack: D1Stack, revision_id: str) -> BuildRevisionResult:
    return stack.service.build_application_revision(
        application_id=APP,
        application_environment_id=ENV,
        request=build_request(revision_id),
        principal=principal(),
    )


def activate_revision(
    stack: D1Stack,
    revision_id: str,
    *,
    expected_prior: str | None = None,
    expected_pointer: int = 0,
    mutation_id: str | None = None,
    principal_value: RequestIdentity | None = None,
):
    built = stack.revision_store.get_revision(revision_id)
    assert built is not None
    return stack.service.activate_revision(
        application_id=APP,
        application_environment_id=ENV,
        principal=principal_value or principal(),
        request=ActivateRuntimeRevisionRequest(
            mutation_id=mutation_id or f"mut-activate-{revision_id}",
            runtime_revision_id=revision_id,
            artifact_locator="test://artifact",
            expected_artifact_digest=ARTIFACT_DIGEST,
            expected_prior_traffic_revision_id=expected_prior,
            expected_serving_pointer_revision=expected_pointer,
        ),
    )
