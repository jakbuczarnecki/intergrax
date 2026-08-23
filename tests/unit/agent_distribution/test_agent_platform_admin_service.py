# © Artur Czarnecki. All rights reserved.

"""AP-11 Agent Platform admin facade tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.core.qualification import QualificationStatus
from pydantic import ValidationError

from intergrax.agent_distribution.activation import (
    ActivationService,
    FakeRuntimeServingProjectionCoordinator,
)
from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    AgentPlatformAdminBlockedError,
    BindAgentRequest,
    BuildApplicationRevisionRequest,
    InstallAgentRequest,
    RollbackRuntimeRevisionRequest,
    SetAgentEnablementRequest,
    UpdateAgentBindingRequest,
)
from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
from intergrax.agent_distribution.agent_project_metadata import AgentProjectMetadata
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.agent_distribution.binding_service import BindingService
from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.dependency import RepositoryDependencyDeclaration
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter
from intergrax.agent_distribution.effective_roster import (
    EffectiveRosterBuilder,
    InstalledAgentRequirementSetBuilder,
)
from intergrax.agent_distribution.errors import (
    AgentDistributionNotFoundError,
    BindingRevisionConflict,
    RuntimeActivationConflict,
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
    InMemoryMaterializedRuntimeLockStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.installation import InstallationState
from intergrax.agent_distribution.installation_service import InstallationService
from intergrax.agent_distribution.materialization import MaterializationOutput
from intergrax.agent_distribution.materialization_service import RuntimeMaterializationService
from intergrax.agent_distribution.runtime_graph_service import CandidateRuntimeGraphBuilder
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from testing_support.agent_platform_dependency_resolver import make_identity_dependency_resolver

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_APP = "app-a"
_APP_B = "app-b"
_ENV = "env-prod"
_DIGEST = "sha256:" + ("a" * 64)
_ARTIFACT = "sha256:" + ("d" * 64)
_PACKAGE_ID = "intergrax-local-search-agent"
_META_REF = "meta://search"
_PACKAGE = AgentPackageIdentity(
    distribution_package_id=_PACKAGE_ID,
    package_version="1.0.0",
    package_digest=_DIGEST,
)
_TEST_PRINCIPAL = RequestIdentity(
    tenant_id="tenant-test",
    user_id="admin-1",
    principal_type=PrincipalType.USER,
    auth_subject="admin-1",
)


@dataclass
class _AllowEvaluator:
    def evaluate(self, request: object) -> PolicyDecision:
        del request
        return PolicyDecision(action=PolicyAction.ALLOW, reason="test_allow")


def admin_test_principal() -> RequestIdentity:
    return _TEST_PRINCIPAL


def allow_mutation_boundary() -> ControlPlaneMutationAuthorizationBoundary:
    return ControlPlaneMutationAuthorizationBoundary(evaluator=_AllowEvaluator())


class _MetadataProvider:
    def __init__(self, records: dict[str, AgentProjectMetadata]) -> None:
        self._records = records

    def get_metadata(self, metadata_ref: str) -> AgentProjectMetadata | None:
        return self._records.get(metadata_ref)


class _DeterministicAdapter:
    topology = MaterializationTopology.OCI_IMAGE
    materializer_id = "intergrax.admin-test"
    materializer_version = "1.0.0"

    def materialize(self, materialization_input: object) -> MaterializationOutput:
        del materialization_input
        return MaterializationOutput(
            materialization_artifact_digest=_ARTIFACT,
            artifact_locator="test://artifact",
            health_check_evidence_ref="test://health",
            runtime_graph_manifest_path=".intergrax-runtime-graph.json",
            topology=self.topology,
        )


class _FakeCatalog:
    def __init__(self, entries: list[AgentCatalogEntry]) -> None:
        self._entries = entries

    @property
    def catalog_source_id(self) -> str:
        return "builtin-1"

    def list_entries(self, filters: object | None = None) -> list[AgentCatalogEntry]:
        del filters
        return list(self._entries)

    def resolve_package(self, entry: AgentCatalogEntry, *, version_selector: str) -> object:
        del entry, version_selector
        raise NotImplementedError

    def health(self) -> None:
        return None


def _trust() -> AgentInstallationTrustRecord:
    return AgentInstallationTrustRecord(
        qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
        package_digest=_DIGEST,
        publisher_identity_ref="publisher:acme",
        source_provider_id="builtin",
        trust_evidence_refs=(
            AgentTrustEvidenceRef(
                evidence_id="evidence:service:0",
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
            ),
        ),
    )


def _install_request() -> InstallAgentRequest:
    return InstallAgentRequest(
        installation_id="inst-1",
        installation_slot_id="slot-search",
        package_identity=_PACKAGE,
        artifact_store_ref="store://artifacts/inst-1",
        trust_record=_trust(),
        agent_project_metadata_ref=_META_REF,
    )


def _bind_request() -> BindAgentRequest:
    return BindAgentRequest(
        application_binding_id="bind-search",
        logical_agent_id="researcher",
        installation_slot_id="slot-search",
    )


def _build_request(revision_id: str) -> BuildApplicationRevisionRequest:
    return BuildApplicationRevisionRequest(
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


@dataclass
class AdminStack:
    service: AgentPlatformAdminService
    state: AgentDistributionStoreState
    catalog: _FakeCatalog


def build_admin_stack(*, with_catalog: bool = True) -> AdminStack:
    state = AgentDistributionStoreState()
    installation_store = InMemoryAgentInstallationStore(state)
    binding_store = InMemoryApplicationAgentBindingStore(state)
    revision_store = InMemoryRuntimeRevisionStore(state)
    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    deployment_store = InMemoryDeploymentInstanceStore(state)
    lock_store = InMemoryMaterializedRuntimeLockStore(state)
    artifact_store = InMemoryAgentArtifactMetadataStore(state)
    installation_service = InstallationService(installation_store)
    binding_service = BindingService(binding_store, installation_service)
    revision_service = RuntimeRevisionService(revision_store)
    metadata_provider = _MetadataProvider(
        {
            _META_REF: AgentProjectMetadata(
                distribution_package_id=_PACKAGE_ID,
                dependencies=(),
            )
        }
    )
    catalog = _FakeCatalog(
        [
            AgentCatalogEntry(
                catalog_entry_id="cat-researcher",
                catalog_source=CatalogSourceIdentity(
                    catalog_source_id="builtin-1",
                    provider_kind=CatalogProviderKind.BUILTIN,
                ),
                display_name="Researcher",
                package_id_line=_PACKAGE_ID,
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
            {MaterializationTopology.OCI_IMAGE: _DeterministicAdapter()}
        ),
        metadata_provider=metadata_provider,
        catalog_provider=catalog if with_catalog else None,
        dependency_resolver=make_identity_dependency_resolver(),
        mutation_authorization_boundary=allow_mutation_boundary(),
    )
    return AdminStack(service=service, state=state, catalog=catalog)


def _activate_request(
    *,
    runtime_revision_id: str,
    artifact_locator: str,
    expected_artifact_digest: str,
    expected_serving_pointer_revision: int = 0,
    mutation_id: str = "mut-activate",
) -> ActivateRuntimeRevisionRequest:
    return ActivateRuntimeRevisionRequest(
        mutation_id=mutation_id,
        runtime_revision_id=runtime_revision_id,
        artifact_locator=artifact_locator,
        expected_artifact_digest=expected_artifact_digest,
        expected_serving_pointer_revision=expected_serving_pointer_revision,
    )


def _rollback_request(
    *,
    expected_current_traffic_revision_id: str,
    expected_serving_pointer_revision: int,
    mutation_id: str = "mut-rollback",
) -> RollbackRuntimeRevisionRequest:
    return RollbackRuntimeRevisionRequest(
        mutation_id=mutation_id,
        expected_current_traffic_revision_id=expected_current_traffic_revision_id,
        expected_serving_pointer_revision=expected_serving_pointer_revision,
    )


def _install_bind(stack: AdminStack) -> None:
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(),
    )
    stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_bind_request(),
    )


def test_list_installed_bound_and_desired_state() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    installed = stack.service.list_installed(application_id=_APP, application_environment_id=_ENV)
    assert len(installed.installations) == 1
    assert installed.installations[0].installation_state is InstallationState.INSTALLED_ACTIVE
    bindings = stack.service.list_bindings(application_id=_APP, application_environment_id=_ENV)
    assert bindings.bindings[0].logical_agent_id == "researcher"
    assert bindings.bindings[0].enablement is False
    roster = stack.service.inspect_effective_roster(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert roster.entries[0].logical_agent_id == "researcher"
    assert roster.entries[0].effective_enablement is False


def test_status_read_model_distinguishes_desired_vs_serving() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    enabled = stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(expected_revision=0),
    )
    assert enabled.binding.enablement is True
    status = stack.service.inspect_agent_status(
        application_id=_APP,
        application_environment_id=_ENV,
        logical_agent_id="researcher",
    )
    assert status.available is True
    assert status.installed is True
    assert status.bound is True
    assert status.enabled_in_desired_state is True
    assert status.included_in_active_revision is False
    assert status.traffic_serving_revision_id is None
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id is None


def test_enable_does_not_change_serving_revision() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    before = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(expected_revision=0),
    )
    after = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert after.traffic_serving_revision_id == before.traffic_serving_revision_id
    assert after.serving_pointer_revision == before.serving_pointer_revision


def test_disable_does_not_change_serving_revision() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(expected_revision=0),
    )
    built = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-17"),
    )
    activated = stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_activate_request(
            runtime_revision_id=built.runtime_revision_id,
            artifact_locator=built.artifact_locator or "test://artifact",
            expected_artifact_digest=built.materialization_artifact_digest or _ARTIFACT,
        ),
    )
    serving_id = activated.traffic_serving_revision_id
    stack.service.disable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(expected_revision=1),
    )
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id == serving_id
    status = stack.service.inspect_agent_status(
        application_id=_APP,
        application_environment_id=_ENV,
        logical_agent_id="researcher",
    )
    assert status.enabled_in_desired_state is False
    assert status.traffic_serving_revision_id == serving_id


def test_build_creates_candidate_but_does_not_activate() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(expected_revision=0),
    )
    built = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-18"),
    )
    assert built.runtime_revision_id == "rev-18"
    assert built.revision_state is RuntimeRevisionState.VALIDATED
    assert built.effective_roster_revision_id
    assert built.materialized_runtime_lock_id
    assert built.runtime_graph_digest
    assert built.materialization_artifact_digest == _ARTIFACT
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id is None
    candidate = stack.service.inspect_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-18",
    )
    assert candidate.revision_state is RuntimeRevisionState.VALIDATED
    status = stack.service.inspect_agent_status(
        application_id=_APP,
        application_environment_id=_ENV,
        logical_agent_id="researcher",
    )
    assert status.pending_candidate_revision_id == "rev-18"
    assert status.included_in_active_revision is False


def test_activate_delegates_ap9_and_changes_serving_revision() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(expected_revision=0),
    )
    built = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-18"),
    )
    activated = stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_activate_request(
            runtime_revision_id="rev-18",
            artifact_locator=built.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
        ),
    )
    assert activated.traffic_serving_revision_id == "rev-18"
    assert activated.revision_state is RuntimeRevisionState.ACTIVE
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id == "rev-18"
    status = stack.service.inspect_agent_status(
        application_id=_APP,
        application_environment_id=_ENV,
        logical_agent_id="researcher",
    )
    assert status.included_in_active_revision is True
    assert status.pending_candidate_revision_id is None


def test_rollback_delegates_immutable_ap9_rollback() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(expected_revision=0),
    )
    first = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-17"),
    )
    stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_activate_request(
            runtime_revision_id="rev-17",
            artifact_locator=first.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
        ),
    )
    second = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-18"),
    )
    stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=ActivateRuntimeRevisionRequest(
            mutation_id="mut-activate-rev-18",
            runtime_revision_id="rev-18",
            artifact_locator=second.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
            expected_serving_pointer_revision=1,
            expected_prior_traffic_revision_id="rev-17",
        ),
    )
    rolled = stack.service.rollback_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=RollbackRuntimeRevisionRequest(
            mutation_id="mut-rollback-rev-17",
            expected_current_traffic_revision_id="rev-18",
            expected_serving_pointer_revision=2,
            target_runtime_revision_id="rev-17",
        ),
    )
    assert rolled.restored_revision_id == "rev-17"
    serving = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving.traffic_serving_revision_id == "rev-17"


def test_stale_binding_conflict_propagated() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(expected_revision=0),
    )
    with pytest.raises(BindingRevisionConflict):
        stack.service.enable_binding(
            application_id=_APP,
            application_environment_id=_ENV,
            application_binding_id="bind-search",
            request=SetAgentEnablementRequest(expected_revision=0),
        )


def test_stale_activation_conflict_propagated() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(expected_revision=0),
    )
    first = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-17"),
    )
    stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_activate_request(
            runtime_revision_id="rev-17",
            artifact_locator=first.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
        ),
    )
    second = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-18"),
    )
    stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=ActivateRuntimeRevisionRequest(
            mutation_id="mut-activate-rev-18",
            runtime_revision_id="rev-18",
            artifact_locator=second.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
            expected_serving_pointer_revision=1,
            expected_prior_traffic_revision_id="rev-17",
        ),
    )
    with pytest.raises(RuntimeActivationConflict):
        stack.service.rollback_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=_rollback_request(
                expected_current_traffic_revision_id="rev-missing",
                expected_serving_pointer_revision=2,
            ),
        )


def test_catalog_list_blocked_without_provider() -> None:
    stack = build_admin_stack(with_catalog=False)
    with pytest.raises(AgentPlatformAdminBlockedError) as exc:
        stack.service.list_catalog()
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_MISSING_CATALOG_PROVIDER"


def test_build_revision_blocked_without_dependency_resolver() -> None:
    stack = build_admin_stack()
    stack.service._lock_service = None  # type: ignore[attr-defined]
    with pytest.raises(AgentPlatformAdminBlockedError) as exc:
        stack.service.build_application_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_build_request("rev-no-resolver"),
        )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_MISSING_DEPENDENCY_RESOLVER"


def test_catalog_list_delegates_to_provider() -> None:
    stack = build_admin_stack()
    listed = stack.service.list_catalog()
    assert listed.entries[0].display_name == "Researcher"


def test_raw_secret_config_rejected_by_request_model() -> None:
    with pytest.raises(ValidationError):
        BindAgentRequest(
            application_binding_id="bind-search",
            logical_agent_id="researcher",
            installation_slot_id="slot-search",
            config={"api_key": "sk-secret"},
        )
    with pytest.raises(ValidationError):
        UpdateAgentBindingRequest(expected_revision=0, config={"password": "hunter2"})


def test_ap9_activation_service_still_commits_on_shared_stores() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(expected_revision=0),
    )
    built = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-ap9"),
    )
    result = stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_activate_request(
            runtime_revision_id="rev-ap9",
            artifact_locator=built.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
        ),
    )
    assert result.traffic_serving_revision_id == "rev-ap9"
    history = stack.service.inspect_revision_history(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert history.traffic_serving_revision_id == "rev-ap9"
    assert any(item.runtime_revision_id == "rev-ap9" for item in history.revisions)


def test_cross_application_resource_isolation() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    assert stack.service.list_bindings(application_id=_APP_B, application_environment_id=_ENV).bindings == ()
    with pytest.raises(AgentDistributionNotFoundError):
        stack.service.inspect_installation(
            application_id=_APP_B,
            application_environment_id=_ENV,
            installation_id="inst-1",
        )
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(expected_revision=0),
    )
    built = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-scope"),
    )
    with pytest.raises(AgentDistributionNotFoundError):
        stack.service.inspect_revision(
            application_id=_APP_B,
            application_environment_id=_ENV,
            runtime_revision_id="rev-scope",
        )
    activated = stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_activate_request(
            runtime_revision_id="rev-scope",
            artifact_locator=built.artifact_locator or "test://artifact",
            expected_artifact_digest=_ARTIFACT,
        ),
    )
    assert activated.traffic_serving_revision_id == "rev-scope"
    app_b_serving = stack.service.inspect_serving(
        application_id=_APP_B,
        application_environment_id=_ENV,
    )
    assert app_b_serving.traffic_serving_revision_id is None
    assert stack.service.inspect_serving(
        application_id=_APP,
        application_environment_id=_ENV,
    ).traffic_serving_revision_id == "rev-scope"
    app_b_status = stack.service.inspect_agent_status(
        application_id=_APP_B,
        application_environment_id=_ENV,
        logical_agent_id="researcher",
    )
    assert app_b_status.bound is False
    assert app_b_status.included_in_active_revision is False
