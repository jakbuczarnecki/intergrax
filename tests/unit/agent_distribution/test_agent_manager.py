# © Artur Czarnecki. All rights reserved.

"""Agent Manager read model, mutation boundary, and route tests (Stage 14)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    AgentPlatformAdminGovernanceBlockedError,
    BindAgentRequest,
    InstallAgentRequest,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.agent_manager_command_facade import (
    AgentManagerCommandFacade,
)
from intergrax.agent_distribution.agent_manager_models import (
    AgentManagerDerivedStatus,
    AgentManagerListFilters,
    LifecycleMatchResolution,
)
from intergrax.agent_distribution.agent_manager_query_service import (
    AgentManagerQueryService,
)
from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
)
from intergrax.agent_distribution.in_memory_stores import (
    InMemoryAgentInstallationStore,
    InMemoryApplicationAgentBindingStore,
    InMemoryApplicationEnvironmentServingStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.effective_roster import EffectiveRosterBuilder
from intergrax.agent_distribution.federated_catalog import FederatedCatalogSourceProvider
from intergrax.applications._shared.agent_manager_routes import mount_agent_manager_routes
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdge,
    CapabilityEdgeType,
    CapabilityGraph,
    CapabilityNode,
    CapabilityNodeType,
)
from intergrax.runtime.architecture.capability_graph_applications import (
    application_capability_node_id,
)
from intergrax.runtime.architecture.capability_graph_query import CapabilityGraphQuery
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    AdminStack,
    _ARTIFACT,
    _APP,
    _DIGEST,
    _ENV,
    _PACKAGE_ID,
    _activate_request,
    _bind_request,
    _build_revision,
    _install_request,
    admin_test_principal,
    build_admin_stack,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _catalog_entry(
    *,
    entry_id: str,
    source_id: str,
    provider_kind: CatalogProviderKind,
    package_id_line: str = _PACKAGE_ID,
    display_name: str = "Researcher",
    categories: tuple[str, ...] = (),
) -> AgentCatalogEntry:
    return AgentCatalogEntry(
        catalog_entry_id=entry_id,
        catalog_source=CatalogSourceIdentity(
            catalog_source_id=source_id,
            provider_kind=provider_kind,
        ),
        display_name=display_name,
        package_id_line=package_id_line,
        categories=categories,
    )


class _StaticCatalog:
    def __init__(self, entries: list[AgentCatalogEntry]) -> None:
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


def _query_service(
    stack: AdminStack,
    *,
    catalog: object | None = None,
    capability_graph: CapabilityGraph | None = None,
) -> AgentManagerQueryService:
    provider = catalog if catalog is not None else stack.catalog
    state = stack.state
    return AgentManagerQueryService(
        catalog_provider=provider,
        installation_store=InMemoryAgentInstallationStore(state),
        binding_store=InMemoryApplicationAgentBindingStore(state),
        revision_store=InMemoryRuntimeRevisionStore(state),
        serving_store=InMemoryApplicationEnvironmentServingStore(state),
        roster_builder=EffectiveRosterBuilder(InMemoryAgentInstallationStore(state)),
        capability_graph_query=(
            CapabilityGraphQuery(capability_graph) if capability_graph else None
        ),
    )


def _manager_client(stack: AdminStack) -> TestClient:
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(require_api_key=False)
    mount_agent_manager_routes(app, query_service=_query_service(stack))
    return TestClient(app)


def _install_only(stack: AdminStack) -> None:
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(),
        principal=admin_test_principal(),
    )


def _install_bind(stack: AdminStack) -> None:
    _install_only(stack)
    stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_bind_request(),
        principal=admin_test_principal(),
    )


def _application_capability_graph(
    *,
    application_id: str = _APP,
    agent_contract_ids: tuple[str, ...] = (),
    extra_nodes: tuple[CapabilityNode, ...] = (),
) -> CapabilityGraph:
    application_node = application_capability_node_id(application_id)
    nodes: list[CapabilityNode] = [
        CapabilityNode(
            node_id=application_node,
            node_type=CapabilityNodeType.APPLICATION,
        ),
        *extra_nodes,
    ]
    edges: list[CapabilityEdge] = []
    for contract_id in agent_contract_ids:
        nodes.append(
            CapabilityNode(
                node_id=f"agent:{contract_id}",
                node_type=CapabilityNodeType.AGENT,
            )
        )
        edges.append(
            CapabilityEdge(
                source_node_id=application_node,
                target_node_id=f"agent:{contract_id}",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            )
        )
    return CapabilityGraph(nodes=nodes, edges=edges)


def test_catalog_only_agent_is_discoverable() -> None:
    stack = build_admin_stack()
    result = _query_service(stack).list_agents(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert result.total == 1
    entry = result.items[0]
    assert entry.derived_status is AgentManagerDerivedStatus.DISCOVERABLE
    assert entry.lifecycle.installed is False
    assert entry.availability.installable is True


def test_installed_but_unbound() -> None:
    stack = build_admin_stack()
    _install_only(stack)
    entry = _query_service(stack).list_agents(
        application_id=_APP,
        application_environment_id=_ENV,
    ).items[0]
    assert entry.derived_status is AgentManagerDerivedStatus.INSTALLED
    assert entry.lifecycle.installed is True
    assert entry.lifecycle.bound is False
    assert entry.availability.bindable is True


def test_bound_disabled() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    entry = _query_service(stack).list_agents(
        application_id=_APP,
        application_environment_id=_ENV,
    ).items[0]
    assert entry.derived_status is AgentManagerDerivedStatus.BOUND
    assert entry.lifecycle.bound is True
    assert entry.lifecycle.enabled_in_desired_state is False


def test_bound_enabled() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(
            mutation_id="mut-enable",
            expected_revision=0,
        ),
        principal=admin_test_principal(),
    )
    entry = _query_service(stack).list_agents(
        application_id=_APP,
        application_environment_id=_ENV,
    ).items[0]
    assert entry.derived_status is AgentManagerDerivedStatus.ENABLED
    assert entry.lifecycle.enabled_in_desired_state is True
    assert entry.runtime.serving is False


def test_desired_but_not_serving() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(
            mutation_id="mut-enable",
            expected_revision=0,
        ),
        principal=admin_test_principal(),
    )
    _build_revision(stack, "rev-candidate")
    entry = _query_service(stack).list_agents(
        application_id=_APP,
        application_environment_id=_ENV,
    ).items[0]
    assert entry.derived_status is AgentManagerDerivedStatus.READY_FOR_REVISION
    assert entry.runtime.included_in_candidate_revision is True
    assert entry.runtime.serving is False


def test_serving() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(
            mutation_id="mut-enable",
            expected_revision=0,
        ),
        principal=admin_test_principal(),
    )
    built = _build_revision(stack, "rev-serving")
    stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=_activate_request(
            runtime_revision_id=built.runtime_revision_id,
            artifact_locator=built.artifact_locator or "test://artifact",
            expected_artifact_digest=built.materialization_artifact_digest or _ARTIFACT,
        ),
    )
    entry = _query_service(stack).list_agents(
        application_id=_APP,
        application_environment_id=_ENV,
    ).items[0]
    assert entry.derived_status is AgentManagerDerivedStatus.SERVING
    assert entry.runtime.serving is True


def test_private_catalog_source_same_lifecycle() -> None:
    private = _StaticCatalog(
        [
            _catalog_entry(
                entry_id="cat-private",
                source_id="enterprise-private-1",
                provider_kind=CatalogProviderKind.ENTERPRISE_PRIVATE,
                display_name="Private Researcher",
            )
        ]
    )
    stack = build_admin_stack()
    query = AgentManagerQueryService(
        catalog_provider=private,
        installation_store=InMemoryAgentInstallationStore(stack.state),
        binding_store=InMemoryApplicationAgentBindingStore(stack.state),
        revision_store=InMemoryRuntimeRevisionStore(stack.state),
        serving_store=InMemoryApplicationEnvironmentServingStore(stack.state),
        roster_builder=EffectiveRosterBuilder(
            InMemoryAgentInstallationStore(stack.state),
        ),
    )
    _install_bind(stack)
    entry = query.list_agents(
        application_id=_APP,
        application_environment_id=_ENV,
    ).items[0]
    assert entry.discovery.provider_kind is CatalogProviderKind.ENTERPRISE_PRIVATE
    assert entry.derived_status is AgentManagerDerivedStatus.BOUND


def test_same_package_name_from_two_sources_remains_distinct() -> None:
    builtin = _StaticCatalog(
        [
            _catalog_entry(
                entry_id="cat-builtin",
                source_id="builtin-1",
                provider_kind=CatalogProviderKind.BUILTIN,
                display_name="Builtin Copy",
            )
        ]
    )
    third_party = _StaticCatalog(
        [
            _catalog_entry(
                entry_id="cat-third-party",
                source_id="third-party-1",
                provider_kind=CatalogProviderKind.GOVERNED_THIRD_PARTY,
                display_name="Third Party Copy",
            )
        ]
    )
    federated = FederatedCatalogSourceProvider((builtin, third_party))
    stack = build_admin_stack()
    query = AgentManagerQueryService(
        catalog_provider=federated,
        installation_store=InMemoryAgentInstallationStore(stack.state),
        binding_store=InMemoryApplicationAgentBindingStore(stack.state),
        revision_store=InMemoryRuntimeRevisionStore(stack.state),
        serving_store=InMemoryApplicationEnvironmentServingStore(stack.state),
        roster_builder=EffectiveRosterBuilder(
            InMemoryAgentInstallationStore(stack.state),
        ),
    )
    result = query.list_agents(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert result.total == 2
    ids = {item.identity.manager_entry_id for item in result.items}
    assert len(ids) == 2


def test_capability_filter() -> None:
    graph = CapabilityGraph(
        nodes=[
            CapabilityNode(
                node_id="agent:researcher",
                node_type=CapabilityNodeType.AGENT,
                metadata={"capabilities": "web_search,analysis"},
            ),
            CapabilityNode(
                node_id="application:app-a_application",
                node_type=CapabilityNodeType.APPLICATION,
            ),
        ],
        edges=[],
    )
    stack = build_admin_stack()
    _install_bind(stack)
    query = _query_service(stack, capability_graph=graph)
    filtered = query.list_agents(
        application_id=_APP,
        application_environment_id=_ENV,
        filters=AgentManagerListFilters(capability="web_search"),
    )
    assert filtered.total == 1


def test_list_agents_for_application_returns_only_graph_members() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    graph = _application_capability_graph(agent_contract_ids=("researcher",))
    query = _query_service(stack, capability_graph=graph)
    result = query.list_agents_for_application(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert result.total == 1
    assert result.items[0].lifecycle.logical_agent_id == "researcher"


def test_list_agents_for_application_known_app_zero_agents_returns_empty() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    graph = _application_capability_graph()
    query = _query_service(stack, capability_graph=graph)
    global_result = query.list_agents(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    scoped = query.list_agents_for_application(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert global_result.total >= 1
    assert scoped.total == 0
    assert scoped.items == ()


def test_list_agents_for_application_unknown_application_returns_empty() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    graph = _application_capability_graph(application_id="other-app")
    query = _query_service(stack, capability_graph=graph)
    scoped = query.list_agents_for_application(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert scoped.total == 0
    assert scoped.items == ()


def test_list_agents_for_application_without_capability_graph_preserves_global_list() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    query = _query_service(stack)
    global_result = query.list_agents(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    scoped = query.list_agents_for_application(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert scoped.total == global_result.total
    assert [item.identity.manager_entry_id for item in scoped.items] == [
        item.identity.manager_entry_id for item in global_result.items
    ]


def test_list_agents_for_application_does_not_infer_membership_from_capability_labels() -> None:
    stack = build_admin_stack()
    _install_only(stack)
    stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=BindAgentRequest(
            mutation_id="mut-bind-decoy",
            application_binding_id="bind-decoy",
            logical_agent_id="decoy",
            installation_slot_id="slot-search",
        ),
        principal=admin_test_principal(),
    )
    graph = _application_capability_graph(
        agent_contract_ids=("researcher",),
        extra_nodes=(
            CapabilityNode(
                node_id="agent:decoy",
                node_type=CapabilityNodeType.AGENT,
                metadata={"capabilities": "researcher"},
            ),
        ),
    )
    query = _query_service(stack, capability_graph=graph)
    result = query.list_agents_for_application(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert result.total == 0
    assert result.items == ()


def test_list_agents_for_application_deterministic_ordering() -> None:
    builtin = _StaticCatalog(
        [
            _catalog_entry(
                entry_id="cat-z",
                source_id="builtin-z",
                provider_kind=CatalogProviderKind.BUILTIN,
                display_name="Z Agent",
            )
        ]
    )
    official = _StaticCatalog(
        [
            _catalog_entry(
                entry_id="cat-a",
                source_id="official-a",
                provider_kind=CatalogProviderKind.OFFICIAL_CATALOG,
                display_name="A Agent",
            )
        ]
    )
    federated = FederatedCatalogSourceProvider((builtin, official))
    stack = build_admin_stack()
    _install_bind(stack)
    graph = _application_capability_graph(agent_contract_ids=("researcher",))
    query = AgentManagerQueryService(
        catalog_provider=federated,
        installation_store=InMemoryAgentInstallationStore(stack.state),
        binding_store=InMemoryApplicationAgentBindingStore(stack.state),
        revision_store=InMemoryRuntimeRevisionStore(stack.state),
        serving_store=InMemoryApplicationEnvironmentServingStore(stack.state),
        roster_builder=EffectiveRosterBuilder(
            InMemoryAgentInstallationStore(stack.state),
        ),
        capability_graph_query=CapabilityGraphQuery(graph),
    )
    first = query.list_agents_for_application(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    second = query.list_agents_for_application(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert [item.identity.manager_entry_id for item in first.items] == [
        item.identity.manager_entry_id for item in second.items
    ]


def test_deterministic_ordering() -> None:
    builtin = _StaticCatalog(
        [
            _catalog_entry(
                entry_id="cat-z",
                source_id="builtin-z",
                provider_kind=CatalogProviderKind.BUILTIN,
                display_name="Z Agent",
            )
        ]
    )
    official = _StaticCatalog(
        [
            _catalog_entry(
                entry_id="cat-a",
                source_id="official-a",
                provider_kind=CatalogProviderKind.OFFICIAL_CATALOG,
                display_name="A Agent",
            )
        ]
    )
    federated = FederatedCatalogSourceProvider((builtin, official))
    stack = build_admin_stack()
    query = AgentManagerQueryService(
        catalog_provider=federated,
        installation_store=InMemoryAgentInstallationStore(stack.state),
        binding_store=InMemoryApplicationAgentBindingStore(stack.state),
        revision_store=InMemoryRuntimeRevisionStore(stack.state),
        serving_store=InMemoryApplicationEnvironmentServingStore(stack.state),
        roster_builder=EffectiveRosterBuilder(
            InMemoryAgentInstallationStore(stack.state),
        ),
    )
    first = query.list_agents(application_id=_APP, application_environment_id=_ENV)
    second = query.list_agents(application_id=_APP, application_environment_id=_ENV)
    assert [item.identity.manager_entry_id for item in first.items] == [
        item.identity.manager_entry_id for item in second.items
    ]


def test_ambiguous_lifecycle_match_is_degraded() -> None:
    stack = build_admin_stack()
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(mutation_id="mut-install-1"),
        principal=admin_test_principal(),
    )
    base = _install_request(mutation_id="mut-install-2")
    duplicate = InstallAgentRequest(
        mutation_id="mut-install-2",
        installation_id="inst-2",
        installation_slot_id="slot-search-2",
        package_identity=base.package_identity,
        artifact_store_ref=base.artifact_store_ref,
        trust_record=base.trust_record,
        agent_project_metadata_ref=base.agent_project_metadata_ref,
    )
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=duplicate,
        principal=admin_test_principal(),
    )
    entry = next(
        item
        for item in _query_service(stack).list_agents(
            application_id=_APP,
            application_environment_id=_ENV,
        ).items
        if item.identity.manager_entry_id.startswith("catalog:")
    )
    assert entry.lifecycle.match_resolution is LifecycleMatchResolution.AMBIGUOUS
    assert entry.derived_status is AgentManagerDerivedStatus.DEGRADED



def test_manager_http_list_and_inspect() -> None:
    stack = build_admin_stack()
    client = _manager_client(stack)
    listed = client.get(
        f"/v1/agent-platform/manager/applications/{_APP}/environments/{_ENV}/agents"
    )
    assert listed.status_code == 200
    payload = listed.json()
    assert payload["total"] == 1
    entry_id = payload["items"][0]["identity"]["manager_entry_id"]
    inspected = client.get(
        f"/v1/agent-platform/manager/applications/{_APP}/environments/{_ENV}/agents/{entry_id}"
    )
    assert inspected.status_code == 200
    assert inspected.json()["identity"]["manager_entry_id"] == entry_id


def test_install_delegates_to_admin_service() -> None:
    stack = build_admin_stack()
    facade = AgentManagerCommandFacade(stack.service)
    admin_spy = MagicMock(wraps=stack.service)
    facade._admin = admin_spy  # type: ignore[method-assign]
    facade.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(),
        principal=admin_test_principal(),
    )
    admin_spy.install_agent.assert_called_once()


def test_bind_delegates_to_admin_service() -> None:
    stack = build_admin_stack()
    facade = AgentManagerCommandFacade(stack.service)
    admin_spy = MagicMock(wraps=stack.service)
    facade._admin = admin_spy  # type: ignore[method-assign]
    facade.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_bind_request(),
        principal=admin_test_principal(),
    )
    admin_spy.bind_agent.assert_called_once()


def test_enable_disable_delegate_to_admin_service() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    facade = AgentManagerCommandFacade(stack.service)
    admin_spy = MagicMock(wraps=stack.service)
    facade._admin = admin_spy  # type: ignore[method-assign]
    request = SetAgentEnablementRequest(
        mutation_id="mut-enable",
        expected_revision=0,
    )
    facade.enable_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=request,
        principal=admin_test_principal(),
    )
    facade.disable_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(
            mutation_id="mut-disable",
            expected_revision=1,
        ),
        principal=admin_test_principal(),
    )
    admin_spy.enable_binding.assert_called_once()
    admin_spy.disable_binding.assert_called_once()



def test_build_revision_delegates_to_admin_service() -> None:
    from tests.unit.agent_distribution.test_agent_platform_admin_service import (
        _build_request,
    )

    stack = build_admin_stack()
    _install_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(
            mutation_id="mut-enable",
            expected_revision=0,
        ),
        principal=admin_test_principal(),
    )
    facade = AgentManagerCommandFacade(stack.service)
    admin_spy = MagicMock(wraps=stack.service)
    facade._admin = admin_spy  # type: ignore[method-assign]
    facade.build_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-delegate"),
        principal=admin_test_principal(),
    )
    admin_spy.build_application_revision.assert_called_once()


def test_activate_delegates_to_admin_service() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(
            mutation_id="mut-enable",
            expected_revision=0,
        ),
        principal=admin_test_principal(),
    )
    built = _build_revision(stack, "rev-activate-delegate")
    facade = AgentManagerCommandFacade(stack.service)
    admin_spy = MagicMock(wraps=stack.service)
    facade._admin = admin_spy  # type: ignore[method-assign]
    facade.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=ActivateRuntimeRevisionRequest(
            mutation_id="mut-activate-delegate",
            runtime_revision_id=built.runtime_revision_id,
            artifact_locator=built.artifact_locator or "test://artifact",
            expected_artifact_digest=built.materialization_artifact_digest or _ARTIFACT,
            expected_serving_pointer_revision=0,
        ),
        principal=admin_test_principal(),
    )
    admin_spy.activate_revision.assert_called_once()


def test_governance_error_propagates_from_facade() -> None:
    stack = build_admin_stack()
    stack.service._environment_tenant_resolver = (  # type: ignore[attr-defined]
        StaticApplicationEnvironmentTenantResolver("other-tenant")
    )
    facade = AgentManagerCommandFacade(stack.service)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        facade.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_install_request(),
            principal=admin_test_principal(),
        )


def test_tenant_scope_preserved_in_delegation() -> None:
    stack = build_admin_stack()
    facade = AgentManagerCommandFacade(stack.service)
    principal = admin_test_principal()
    facade.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(),
        principal=principal,
    )
    installed = stack.service.list_installed(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert len(installed.installations) == 1
