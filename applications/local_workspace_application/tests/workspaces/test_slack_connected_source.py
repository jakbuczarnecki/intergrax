# © Artur Czarnecki. All rights reserved.

"""Focused Slack connected source service tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SLACK_CONVERSATION_SOURCE_KIND,
    SlackConversationKind,
)
from intergrax.runtime.vendor_knowledge.adapters.slack_conversation import (
    SLACK_CONVERSATION_SCOPE_TYPE,
    encode_slack_conversation_scope_id,
    register_slack_conversation_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.binding_document_store import (
    DocumentStoreKnowledgeSourceBindingRepository,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingService,
    KnowledgeSourceBindingStatus,
)
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError
from intergrax.runtime.vendor_knowledge.models import KnowledgeSourceScope
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceBindingError,
    SlackConversationKindV1,
)
from local_workspace_application.workspaces.connected_source_tenant_binding import (
    SlackConversationTenantBindingRequest,
    WorkspaceConnectedSourceTenantBindingService,
)
from local_workspace_application.workspaces.knowledge_access_service import (
    CreateConnectedIndexedSourceRequest,
    WorkspaceKnowledgeAccessService,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    IndexedSourceAudienceEligibilityV1,
    IndexedSourceSyncModeV1,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_CONNECTION = "conn.slack"
_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


class _Resolver:
    def resolve(self, *, source):
        return object()


def _binding_service(store, tenant_id: str) -> KnowledgeSourceBindingService:
    registry = KnowledgeAdapterRegistry()
    register_slack_conversation_knowledge_adapter(registry)
    return KnowledgeSourceBindingService(
        tenant_id=tenant_id,
        repository=DocumentStoreKnowledgeSourceBindingRepository(store),
        integration_resolver=_Resolver(),
        adapter_registry=registry,
    )


def _request(*, label: str = "#project-orion") -> SlackConversationTenantBindingRequest:
    return SlackConversationTenantBindingRequest(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        conversation_id="C01234567",
        conversation_kind=SlackConversationKindV1.PUBLIC_CHANNEL,
        safe_display_name=label,
        root_oldest="1704067200.000001",
        root_latest="1706745600.000001",
    )


def test_tenant_binding_rename_reuses_existing() -> None:
    store = InMemoryDocumentStore()
    service = WorkspaceConnectedSourceTenantBindingService(
        lambda tenant_id: _binding_service(store, tenant_id)
    )
    first = service.create_or_get_equivalent_for_slack_conversation(_request(label="#old-name"))
    second = service.create_or_get_equivalent_for_slack_conversation(_request(label="#new-name"))
    assert first.binding_id == second.binding_id
    assert second.safe_display_name == "#old-name"


def test_tenant_binding_version_reuse_with_higher_existing_version() -> None:
    store = InMemoryDocumentStore()
    binding_service = _binding_service(store, _TENANT)
    encoded_scope = encode_slack_conversation_scope_id(
        conversation_id="C01234567",
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        oldest="1704067200.000001",
        latest="1706745600.000001",
    )
    from local_workspace_application.workspaces.connected_source_ids import tenant_binding_id

    binding_id = tenant_binding_id(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL.value,
        source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        encoded_scope=encoded_scope,
    )
    existing = KnowledgeSourceBinding(
        binding_id=binding_id,
        tenant_id=_TENANT,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        connection_ref=_CONNECTION,
        safe_display_name="#project-orion",
        scope=KnowledgeSourceScope(
            remote_scope_id=encoded_scope,
            remote_scope_type=SLACK_CONVERSATION_SCOPE_TYPE,
            safe_display_name="#project-orion",
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=3,
    )
    binding_service._repository.create(existing)
    service = WorkspaceConnectedSourceTenantBindingService(
        lambda tenant_id: _binding_service(store, tenant_id)
    )
    reused = service.create_or_get_equivalent_for_slack_conversation(_request())
    assert reused.configuration_version == 3


def test_inactive_tenant_binding_rejected() -> None:
    store = InMemoryDocumentStore()
    binding_service = _binding_service(store, _TENANT)
    encoded_scope = encode_slack_conversation_scope_id(
        conversation_id="C01234567",
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        oldest="1704067200.000001",
        latest="1706745600.000001",
    )
    from local_workspace_application.workspaces.connected_source_ids import tenant_binding_id

    binding_id = tenant_binding_id(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL.value,
        source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        encoded_scope=encoded_scope,
    )
    disabled = KnowledgeSourceBinding(
        binding_id=binding_id,
        tenant_id=_TENANT,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        connection_ref=_CONNECTION,
        safe_display_name="#project-orion",
        scope=KnowledgeSourceScope(
            remote_scope_id=encoded_scope,
            remote_scope_type=SLACK_CONVERSATION_SCOPE_TYPE,
            safe_display_name="#project-orion",
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.DISABLED,
        configuration_version=1,
    )
    binding_service._repository.create(disabled)
    service = WorkspaceConnectedSourceTenantBindingService(
        lambda tenant_id: _binding_service(store, tenant_id)
    )
    with pytest.raises(ConnectedSourceBindingError):
        service.create_or_get_equivalent_for_slack_conversation(_request())


def test_forced_full_personal_only_request_shape() -> None:
    request = CreateConnectedIndexedSourceRequest(
        tenant_id=_TENANT,
        workspace_id="workspace-1",
        connection_ref=_CONNECTION,
        opaque_candidate_ref="opaque",
        expected_revision=0,
        idempotency_key_hash="a" * 64,
        root_oldest="1704067200.000001",
        root_latest="1706745600.000001",
    )
    assert not hasattr(request, "sync_mode")
    assert not hasattr(request, "audience_eligibility")
    assert IndexedSourceSyncModeV1.FULL.value == "full"
    assert IndexedSourceAudienceEligibilityV1.PERSONAL_ONLY.value == "personal_only"
