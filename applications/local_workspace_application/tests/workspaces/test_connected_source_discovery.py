# © Artur Czarnecki. All rights reserved.

"""Unit tests for connected Slack source discovery, binding and sink."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest
from local_workspace_application.workspaces.connected_source_candidate import (
    decode_slack_conversation_candidate_ref,
    encode_slack_conversation_candidate_ref,
)
from local_workspace_application.workspaces.connected_source_discovery import (
    ConnectedSourceRevalidationLimits,
    WorkspaceRemoteResourceDiscoveryService,
)
from local_workspace_application.workspaces.connected_source_discovery_slack import (
    SlackRemoteResourceDiscoveryStrategy,
)
from local_workspace_application.workspaces.connected_source_discovery_strategy import (
    RemoteResourceDiscoveryStrategyRegistry,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDiscoveryError,
    RemoteResourceTypeV1,
    SlackConversationKindV1,
)
from local_workspace_application.workspaces.connected_source_opaque_ref_codec import (
    RemoteResourceOpaqueRefCodec,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.backend import (
    SlackConversationChannelBackend,
)
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SlackConversationInventoryPage,
    SlackConversationKind,
    SlackConversationSummary,
)
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_TENANT_B = "tenant-b"
_WORKSPACE = "workspace-1"
_CONNECTION = "conn.slack"
_SIGNING_KEY = "connected-source-test-signing-key"


class _FakeBackend(SlackConversationChannelBackend):
    def __init__(self, pages: list[SlackConversationInventoryPage] | None = None) -> None:
        super().__init__(
            config=SlackConversationChannelIntegrationConfig(
                enabled=True,
                app_token="xapp-test",
                bot_token="xoxb-test",
            )
        )
        self._pages = list(
            pages
            or [
                SlackConversationInventoryPage(
                    items=(
                        SlackConversationSummary(
                            conversation_id="C01234567",
                            kind=SlackConversationKind.PUBLIC_CHANNEL,
                            safe_name="#project-orion",
                            is_archived=False,
                            is_private=False,
                        ),
                    ),
                    next_cursor=None,
                )
            ]
        )
        self.provider_cursors_seen: list[str | None] = []

    async def list_accessible_conversations_page(self, *, cursor, limit):
        self.provider_cursors_seen.append(cursor)
        if not self._pages:
            return SlackConversationInventoryPage(items=(), next_cursor=None)
        return self._pages.pop(0)

    async def read_conversation_history_page(self, **kwargs):
        raise NotImplementedError

    async def read_thread_replies_page(self, **kwargs):
        raise NotImplementedError

    async def read_exact_message(self, **kwargs):
        raise NotImplementedError

    async def read_file_info(self, **kwargs):
        raise NotImplementedError


class _FakeWorkspaceLookup:
    def require_workspace(self, *, tenant_id: str, workspace_id: str):
        if tenant_id != _TENANT or workspace_id != _WORKSPACE:
            return None
        return Workspace(
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            name="ws",
            status=WorkspaceStatus.ACTIVE,
            created_at=_NOW,
            updated_at=_NOW,
        )


@pytest.fixture
def codec() -> RemoteResourceOpaqueRefCodec:
    return RemoteResourceOpaqueRefCodec.from_signing_key_material(_SIGNING_KEY)


@pytest.fixture
def discovery_env(codec: RemoteResourceOpaqueRefCodec):
    store = InMemoryDocumentStore()
    repo = __import__(
        "local_workspace_application.workspaces.repository",
        fromlist=["ManagedWorkspaceRepository"],
    ).ManagedWorkspaceRepository(store)
    workspace = Workspace(
        workspace_id=_WORKSPACE,
        tenant_id=_TENANT,
        name="ws",
        status=WorkspaceStatus.ACTIVE,
        created_at=_NOW,
        updated_at=_NOW,
    )
    repo.put_workspace(workspace)
    repo.put_knowledge_connection_attachment_version_if_absent(
        WorkspaceConnectionAttachment(
            attachment_id="att-1",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            safe_display_label="Slack",
            status=WorkspaceConnectionAttachmentStatusV1.ATTACHED,
            mutation_id="mut-1",
            effective_revision=1,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    head_mod = __import__(
        "local_workspace_application.workspaces.knowledge_configuration_models",
        fromlist=["WorkspaceKnowledgeConfigurationHead"],
    )
    repo.put_knowledge_configuration_head_if_absent(
        head_mod.WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=1,
            updated_at=_NOW,
        )
    )
    registry = KnowledgeConnectionRegistry()
    backend = _FakeBackend()
    integration = SlackConversationChannelIntegration.from_backend(
        backend,  # type: ignore[arg-type]
        enabled=True,
    )
    registry.register(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        integration=integration,
    )
    lookup = _FakeWorkspaceLookup()
    config = WorkspaceKnowledgeConfigurationService(repo, lookup)
    service = WorkspaceRemoteResourceDiscoveryService(
        workspace_lookup=lookup,
        configuration_reader=config,
        opaque_ref_codec=codec,
        strategy_registry=RemoteResourceDiscoveryStrategyRegistry(
            (
                SlackRemoteResourceDiscoveryStrategy(
                    connection_registry=registry,
                    opaque_ref_codec=codec,
                ),
            )
        ),
    )
    return service, registry, integration, backend, codec


@pytest.mark.asyncio
async def test_attached_slack_connection_lists_safe_candidates(discovery_env) -> None:
    service, _, _, _, _ = discovery_env
    page = await service.list_remote_resources(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        resource_type=RemoteResourceTypeV1.SLACK_CONVERSATION,
        cursor=None,
        limit=10,
    )
    assert len(page.items) == 1
    item = page.items[0]
    assert item.safe_display_label == "#project-orion"
    assert item.resource_type is RemoteResourceTypeV1.SLACK_CONVERSATION
    assert "token" not in item.opaque_candidate_ref


@pytest.mark.asyncio
async def test_unattached_connection_rejected(discovery_env) -> None:
    service, _, _, _, _ = discovery_env
    with pytest.raises(ConnectedSourceDiscoveryError) as exc:
        await service.list_remote_resources(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref="missing",
            resource_type=RemoteResourceTypeV1.SLACK_CONVERSATION,
            cursor=None,
            limit=10,
        )
    assert exc.value.error_code == "connection_not_attached"


def test_signed_candidate_mutation_rejected(codec: RemoteResourceOpaqueRefCodec) -> None:
    ref = encode_slack_conversation_candidate_ref(
        codec=codec,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        conversation_id="C01234567",
        conversation_kind=SlackConversationKindV1.PUBLIC_CHANNEL,
        safe_display_label="#project-orion",
    )
    tampered = ref[:-1] + ("A" if ref[-1] != "A" else "B")
    with pytest.raises(ConnectedSourceDiscoveryError):
        decode_slack_conversation_candidate_ref(codec, tampered)


def test_cross_tenant_candidate_rejected(codec: RemoteResourceOpaqueRefCodec) -> None:
    ref = encode_slack_conversation_candidate_ref(
        codec=codec,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        conversation_id="C01234567",
        conversation_kind=SlackConversationKindV1.PUBLIC_CHANNEL,
        safe_display_label="#project-orion",
    )
    payload = decode_slack_conversation_candidate_ref(codec, ref)
    with pytest.raises(ConnectedSourceDiscoveryError) as exc:
        __import__(
            "local_workspace_application.workspaces.connected_source_candidate",
            fromlist=["validate_candidate_scope"],
        ).validate_candidate_scope(
            payload,
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
        )
    assert exc.value.error_code == "workspace_not_found"


@pytest.mark.asyncio
async def test_signed_pagination_cursor_and_no_raw_provider_cursor(discovery_env) -> None:
    service, _, _, backend, _ = discovery_env
    backend._pages = [
        SlackConversationInventoryPage(
            items=(
                SlackConversationSummary(
                    conversation_id="C01234567",
                    kind=SlackConversationKind.PUBLIC_CHANNEL,
                    safe_name="#project-orion",
                    is_archived=False,
                    is_private=False,
                ),
            ),
            next_cursor="provider-page-2",
        ),
        SlackConversationInventoryPage(items=(), next_cursor=None),
    ]
    first = await service.list_remote_resources(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        resource_type=RemoteResourceTypeV1.SLACK_CONVERSATION,
        cursor=None,
        limit=10,
    )
    assert first.next_cursor is not None
    assert "provider-page-2" not in first.next_cursor
    await service.list_remote_resources(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        resource_type=RemoteResourceTypeV1.SLACK_CONVERSATION,
        cursor=first.next_cursor,
        limit=10,
    )
    assert backend.provider_cursors_seen == [None, "provider-page-2"]


@pytest.mark.asyncio
async def test_bounded_revalidation_limit_exceeded(discovery_env) -> None:
    service, _, _, backend, codec = discovery_env
    looping = SlackConversationInventoryPage(
        items=(),
        next_cursor="same",
    )
    backend._pages = [looping, looping, looping, looping]
    service._revalidation_limits = ConnectedSourceRevalidationLimits(
        max_pages=2,
        max_total_candidates=10,
        max_duration_seconds=1.0,
        page_size=5,
    )
    candidate_ref = encode_slack_conversation_candidate_ref(
        codec=codec,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        conversation_id="C999",
        conversation_kind=SlackConversationKindV1.PUBLIC_CHANNEL,
        safe_display_label="#missing",
    )
    with pytest.raises(ConnectedSourceDiscoveryError) as exc:
        await service.revalidate_candidate_label(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            resource_type=RemoteResourceTypeV1.SLACK_CONVERSATION,
            opaque_candidate_ref=candidate_ref,
        )
    assert exc.value.error_code == "candidate_revalidation_limit_exceeded"


def test_lkw_contains_no_slack_sdk_import() -> None:
    import local_workspace_application.workspaces.connected_source_discovery as mod

    source = inspect.getsource(mod)
    assert "slack_sdk" not in source
    assert "AsyncWebClient" not in source


def test_one_integration_instance_reused(discovery_env) -> None:
    _, registry, integration, _, _ = discovery_env
    resolved = registry.resolve(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
    )
    assert resolved is integration
