from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

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
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationPolicy,
    ConversationAudienceMode,
    ConversationExecutionContextV1,
    ConversationProductCapability,
    ConversationThreadContextPolicy,
)
from local_workspace_application.workspaces.knowledge_plugin_configuration_service import (
    KnowledgeConnectionSummaryV1,
    KnowledgeConfigurationModeV1,
    KnowledgePluginConfigurationService,
    KnowledgePluginConfigurationSnapshotV1,
)
from local_workspace_application.conversation.interaction_draft_models import (
    ConversationInteractionDraft,
    KnowledgeResourcesListDraftAction,
)
from local_workspace_application.conversation.interaction_execution_models import (
    ConversationActionExecutionStatus,
    ConversationInteractionExecutionCommand,
)
from local_workspace_application.conversation.interaction_executor import ConversationInteractionExecutor
from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
    ConversationPlanningRequest,
    KnowledgeConnectionsListPlannedAction,
)
from local_workspace_application.conversation.interaction_plan_compiler import compile_interaction_draft
from local_workspace_application.conversation.interaction_planner import (
    PlanRequestValidationError,
    validate_plan_against_request,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _context() -> ConversationExecutionContextV1:
    return ConversationExecutionContextV1(
        tenant_id="tenant-a",
        conversation_context_binding_id="binding-1",
        audience_mode=ConversationAudienceMode.PERSONAL,
        workspace_id="workspace-1",
        principal_ref="principal-1",
        canonical_thread_ref="thread-1",
        activation_policy=ConversationActivationPolicy.ALWAYS,
        thread_context_policy=ConversationThreadContextPolicy.CURRENT_THREAD_BOUNDED,
        allowed_product_capabilities=frozenset(
            {ConversationProductCapability.READ_ONLY_ASK}
        ),
    )


def _connection(ref: str, provider: str, status: TenantConnectionAdministrativeStatus) -> SafeTenantConnectionV1:
    return SafeTenantConnectionV1(
        connection_ref=ref,
        tenant_id="tenant-a",
        provider_id=provider,
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        safe_display_name=ref,
        administrative_status=status,
        configuration_version=1,
        connected_principal_ref=None,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _capability(provider: str) -> LiveCapabilityDescriptorV1:
    return LiveCapabilityDescriptorV1(
        capability_id=f"vendor.{provider}.issues.read",
        provider_id=provider,
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
        contract_version="1",
        effect=CapabilityEffectV1.READ,
        read_only=True,
        resource_scope_required=True,
        request_schema_ref=f"schema://{provider}/request/v1",
        result_schema_ref=f"schema://{provider}/result/v1",
    )


class _Connections:
    def __init__(self, connections: tuple[SafeTenantConnectionV1, ...]) -> None:
        self.connections = connections

    def list_connections(self, **_: object) -> tuple[SafeTenantConnectionV1, ...]:
        return self.connections

    def get_connection(self, connection_ref: str) -> SafeTenantConnectionV1:
        return next(item for item in self.connections if item.connection_ref == connection_ref)


class _Catalog:
    def __init__(self, capabilities: dict[str, tuple[LiveCapabilityDescriptorV1, ...]]) -> None:
        self.capabilities = capabilities

    def list_capabilities(self, *, connection_ref: str, **_: object) -> tuple[LiveCapabilityDescriptorV1, ...]:
        return self.capabilities.get(connection_ref, ())


class _Discovery:
    def __init__(self, pages: dict[str, RemoteResourceDiscoveryPageV1]) -> None:
        self.pages = pages

    def list_source_kinds(self, *, connection_ref: str) -> tuple[str, ...]:
        return ("issues",) if connection_ref in self.pages else ()

    async def list_remote_resources(self, *, connection_ref: str, **_: object) -> RemoteResourceDiscoveryPageV1:
        return self.pages[connection_ref]


def _service() -> KnowledgePluginConfigurationService:
    first = _connection("connection-a", "provider_a", TenantConnectionAdministrativeStatus.ACTIVE)
    second = _connection("connection-b", "provider_b", TenantConnectionAdministrativeStatus.ACTIVE)
    resources = {
        ref: RemoteResourceDiscoveryPageV1(
            resources=(
                RemoteResourceDescriptorV1(
                    connection_ref=ref,
                    remote_resource_id=f"{ref}-resource",
                    provider_id=provider,
                    integration_kind=IntegrationCategory.ISSUE_TRACKER,
                    source_kind="issues",
                    resource_type="project",
                    safe_display_label=f"{ref} project",
                    safe_description="Safe project",
                    availability=RemoteResourceAvailabilityV1.AVAILABLE,
                    supported_capability_ids=(f"vendor.{provider}.issues.read",),
                    discovered_at=_NOW,
                    snapshot_version="snapshot-1",
                ),
            ),
            snapshot_version="snapshot-1",
        )
        for ref, provider in (("connection-a", "provider_a"), ("connection-b", "provider_b"))
    }
    connections = _Connections((second, first))
    catalog = _Catalog(
        {
            "connection-a": (_capability("provider_a"),),
            "connection-b": (_capability("provider_b"),),
        }
    )
    discovery = _Discovery(resources)
    return KnowledgePluginConfigurationService(
        connection_service_factory=lambda _: connections,
        capability_catalog_factory=lambda _: catalog,
        resource_discovery_service_factory=lambda _: discovery,
        workspace_authorization=SimpleNamespace(
            get_workspace=lambda **_: SimpleNamespace(tenant_id="tenant-a")
        ),
    )


@pytest.mark.asyncio
async def test_snapshot_is_dynamic_safe_and_provider_neutral() -> None:
    snapshot = await _service().get_configuration_snapshot(
        tenant_id="tenant-a",
        execution_context=_context(),
    )

    assert [item.connection_ref for item in snapshot.available_connections] == [
        "connection-a",
        "connection-b",
    ]
    assert len(snapshot.available_remote_resources) == 2
    assert {
        item.capability_id for item in snapshot.available_resource_capabilities
    } == {
        "vendor.provider_a.issues.read",
        "vendor.provider_b.issues.read",
    }
    assert all(
        item.configuration_mode is KnowledgeConfigurationModeV1.LIVE_ACCESS_ELIGIBLE
        for item in snapshot.available_resource_capabilities
    )
    assert all(
        item.indexed_source_eligibility is KnowledgeConfigurationModeV1.UNKNOWN
        for item in snapshot.available_resource_capabilities
    )
    assert "credential_ref" not in snapshot.model_dump()


def test_empty_catalog_is_valid() -> None:
    service = KnowledgePluginConfigurationService(
        connection_service_factory=lambda _: _Connections(()),
        capability_catalog_factory=lambda _: _Catalog({}),
        resource_discovery_service_factory=lambda _: _Discovery({}),
        workspace_authorization=SimpleNamespace(
            get_workspace=lambda **_: SimpleNamespace(tenant_id="tenant-a")
        ),
    )

    assert service.list_connections(
        tenant_id="tenant-a",
        execution_context=_context(),
    ) == ()


def test_planner_rejects_discovery_selector_not_in_snapshot() -> None:
    connection = _connection(
        "connection-a",
        "provider_a",
        TenantConnectionAdministrativeStatus.ACTIVE,
    )
    snapshot = KnowledgePluginConfigurationSnapshotV1(
        available_connections=(
            KnowledgeConnectionSummaryV1(
                connection_ref=connection.connection_ref,
                safe_display_label=connection.safe_display_name,
                provider_id=connection.provider_id,
                integration_kind=connection.integration_kind,
                administrative_status=connection.administrative_status,
                available_source_kinds=("issues",),
            ),
        )
    )
    request = ConversationPlanningRequest(
        message_text="show resources",
        knowledge_plugin_configuration=snapshot,
    )
    draft = ConversationInteractionDraft(
        actions=(
            KnowledgeResourcesListDraftAction(
                action_type="knowledge.resources.list",
                connection_ref="connection-a",
                source_kind="invented",
            ),
        )
    )
    plan = compile_interaction_draft(draft, request)

    with pytest.raises(PlanRequestValidationError):
        validate_plan_against_request(plan, request)


@pytest.mark.asyncio
async def test_executor_dispatches_one_authorized_product_service_call() -> None:
    class _PluginService:
        calls = 0

        def list_connections(self, **_: object) -> tuple[object, ...]:
            self.calls += 1
            return ()

    plugin = _PluginService()
    context = _context().model_copy(
        update={
            "allowed_product_capabilities": frozenset(
                {
                    ConversationProductCapability.KNOWLEDGE_CONFIGURATION_DISCOVERY,
                }
            )
        }
    )
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            KnowledgeConnectionsListPlannedAction(
                action_id="connections",
                action_type="knowledge.connections.list",
            ),
        ),
        response_mode="aggregate",
    )
    snapshot = KnowledgePluginConfigurationSnapshotV1(
        available_connections=(
            KnowledgeConnectionSummaryV1(
                connection_ref="connection-a",
                safe_display_label="Connection A",
                provider_id="provider_a",
                integration_kind=IntegrationCategory.ISSUE_TRACKER,
                administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            ),
        )
    )
    executor = ConversationInteractionExecutor(
        workspace_service=SimpleNamespace(),
        workspace_selection_service=SimpleNamespace(),
        knowledge_plugin_configuration_service=plugin,  # type: ignore[arg-type]
    )

    result = await executor.execute(
        ConversationInteractionExecutionCommand(
            tenant_id="tenant-a",
            planning_request=ConversationPlanningRequest(
                message_text="show connections",
                knowledge_plugin_configuration=snapshot,
            ),
            interaction_plan=plan,
            execution_context=context,
        )
    )

    assert result.action_results[0].status is ConversationActionExecutionStatus.COMPLETED
    assert plugin.calls == 1
