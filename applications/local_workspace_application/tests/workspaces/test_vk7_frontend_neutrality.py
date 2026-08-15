from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationAudienceMode,
    ConversationExecutionContextV1,
    ConversationProductCapability,
    ConversationThreadContextPolicy,
)
from local_workspace_application.workspaces.knowledge_plugin_configuration_service import (
    KnowledgePluginConfigurationService,
)

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.remote_resource_discovery import (
    RemoteResourceAvailabilityV1,
    RemoteResourceDescriptorV1,
    RemoteResourceDiscoveryPageV1,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnectionAdministrativeStatus,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).parents[4]
_NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
_TENANT = "vk7-fixture-tenant"
_PROVIDER = "vk7_fixture_provider"
_CONNECTION = "vk7-fixture-connection"

_GENERIC_APPLICATION_SURFACES = (
    "applications/local_workspace_application/serving/knowledge_connected_source_routes.py",
    "applications/local_workspace_application/serving/knowledge_live_access_routes.py",
    "applications/local_workspace_application/serving/knowledge_query_policy_routes.py",
    "applications/local_workspace_application/workspaces/hybrid_ask_models.py",
    "applications/local_workspace_application/workspaces/hybrid_ask_service.py",
    "applications/local_workspace_application/workspaces/knowledge_access_service.py",
    "applications/local_workspace_application/workspaces/knowledge_administration_service.py",
    "applications/local_workspace_application/workspaces/knowledge_inspection_operations_service.py",
    "applications/local_workspace_application/workspaces/knowledge_plugin_configuration_service.py",
    "applications/local_workspace_application/workspaces/connected_source_discovery.py",
)

_FORBIDDEN_PROVIDER_IMPORTS = (
    "intergrax.integrations.providers.",
    "intergrax.runtime.vendor_knowledge.live.slack",
    "local_workspace_application.workspaces.slack_ask_orchestration",
)


def _context() -> ConversationExecutionContextV1:
    return ConversationExecutionContextV1(
        tenant_id=_TENANT,
        conversation_context_binding_id="vk7-context",
        audience_mode=ConversationAudienceMode.PERSONAL,
        workspace_id="vk7-workspace",
        principal_ref="vk7-principal",
        canonical_thread_ref="vk7-thread",
        activation_policy=ConversationActivationPolicy.ALWAYS,
        thread_context_policy=ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
        allowed_product_capabilities=frozenset(
            {ConversationProductCapability.READ_ONLY_ASK}
        ),
    )


class _FixtureConnections:
    def __init__(self) -> None:
        self.connection = SafeTenantConnectionV1(
            connection_ref=_CONNECTION,
            tenant_id=_TENANT,
            provider_id=_PROVIDER,
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            safe_display_name="Fixture Provider",
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            configuration_version=1,
            connected_principal_ref=None,
            created_at=_NOW,
            updated_at=_NOW,
        )

    def list_connections(self, **_: object) -> tuple[SafeTenantConnectionV1, ...]:
        return (self.connection,)

    def get_connection(self, connection_ref: str) -> SafeTenantConnectionV1:
        if connection_ref != _CONNECTION:
            raise LookupError(connection_ref)
        return self.connection


class _FixtureDiscovery:
    def list_source_kinds(self, *, connection_ref: str) -> tuple[str, ...]:
        return ("fixture_resource",) if connection_ref == _CONNECTION else ()

    async def list_remote_resources(
        self,
        *,
        connection_ref: str,
        **_: object,
    ) -> RemoteResourceDiscoveryPageV1:
        return RemoteResourceDiscoveryPageV1(
            resources=(
                RemoteResourceDescriptorV1(
                    connection_ref=connection_ref,
                    remote_resource_id="fixture-resource-1",
                    provider_id=_PROVIDER,
                    integration_kind=IntegrationCategory.ISSUE_TRACKER,
                    source_kind="fixture_resource",
                    resource_type="fixture_resource",
                    safe_display_label="Fixture resource",
                    safe_description="Provider-neutral test resource",
                    availability=RemoteResourceAvailabilityV1.AVAILABLE,
                    supported_capability_ids=(
                        "vendor.vk7_fixture_provider.fixture_resource.read",
                    ),
                    discovered_at=_NOW,
                    snapshot_version="fixture-snapshot-1",
                ),
            ),
            snapshot_version="fixture-snapshot-1",
        )


class _FixtureCapabilities:
    def list_capabilities(
        self,
        *,
        connection_ref: str,
        **_: object,
    ) -> tuple[LiveCapabilityDescriptorV1, ...]:
        if connection_ref != _CONNECTION:
            return ()
        return (
            LiveCapabilityDescriptorV1(
                capability_id="vendor.vk7_fixture_provider.fixture_resource.read",
                provider_id=_PROVIDER,
                integration_kind=IntegrationCategory.ISSUE_TRACKER,
                source_kind="fixture_resource",
                contract_version="1",
                effect=CapabilityEffectV1.READ,
                read_only=True,
                resource_scope_required=True,
                request_schema_ref="schema://vk7-fixture/request/v1",
                result_schema_ref="schema://vk7-fixture/result/v1",
                max_result_items=10,
                max_result_bytes=4096,
            ),
        )


def _fixture_service() -> KnowledgePluginConfigurationService:
    connections = _FixtureConnections()
    return KnowledgePluginConfigurationService(
        connection_service_factory=lambda _: connections,
        capability_catalog_factory=lambda _: _FixtureCapabilities(),
        resource_discovery_service_factory=lambda _: _FixtureDiscovery(),
        workspace_authorization=SimpleNamespace(
            get_workspace=lambda **_: SimpleNamespace(tenant_id=_TENANT)
        ),
    )


@pytest.mark.asyncio
async def test_new_provider_uses_generic_configuration_discovery_and_capability_contracts() -> None:
    snapshot = await _fixture_service().get_configuration_snapshot(
        tenant_id=_TENANT,
        execution_context=_context(),
    )

    assert [item.provider_id for item in snapshot.available_connections] == [_PROVIDER]
    assert [item.source_kind for item in snapshot.available_remote_resources] == [
        "fixture_resource"
    ]
    assert [
        item.capability_id for item in snapshot.available_resource_capabilities
    ] == ["vendor.vk7_fixture_provider.fixture_resource.read"]
    assert snapshot.available_remote_resources[0].configuration_modes


def test_generic_application_surfaces_have_no_concrete_provider_imports() -> None:
    for relative_path in _GENERIC_APPLICATION_SURFACES:
        tree = ast.parse((_ROOT / relative_path).read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported_modules.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            module.startswith(prefix)
            for module in imported_modules
            for prefix in _FORBIDDEN_PROVIDER_IMPORTS
        ), relative_path


def test_generic_application_surfaces_have_no_provider_switch_literals() -> None:
    provider_literals = {"slack", "ms365_graph", "google_workspace"}
    for relative_path in _GENERIC_APPLICATION_SURFACES:
        tree = ast.parse((_ROOT / relative_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            expression = ast.unparse(node.test)
            if not any(token in expression for token in ("provider", "source_kind")):
                continue
            assert not any(
                literal in expression.casefold() for literal in provider_literals
            ), f"{relative_path}: {expression}"
