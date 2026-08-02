# © Artur Czarnecki. All rights reserved.

"""End-to-end Slack connected source proof through Vendor Knowledge and LKW sink."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from typing import Any

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SLACK_CONVERSATION_SOURCE_KIND,
    SlackConversationExactMessageResult,
    SlackConversationKind,
    SlackConversationMessage,
    SlackConversationMessagePage,
    compute_slack_conversation_message_revision,
)
from intergrax.integrations.providers.conversation_channel.slack.mapping import parse_slack_ts
from intergrax.runtime.vendor_knowledge.adapters.slack_conversation import (
    SLACK_CONVERSATION_SCOPE_TYPE,
    encode_slack_conversation_scope_id,
    register_slack_conversation_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingService,
    KnowledgeSourceBindingStatus,
)
from intergrax.runtime.vendor_knowledge.binding_document_store import (
    DocumentStoreKnowledgeSourceBindingRepository,
)
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
from intergrax.runtime.vendor_knowledge.models import KnowledgeSourceScope
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from intergrax.runtime.vendor_knowledge.sync_coordinator import VendorKnowledgeSyncCoordinator
from intergrax.runtime.vendor_knowledge.sync_document_store import (
    DocumentStoreKnowledgeRemoteItemStateRepository,
    DocumentStoreKnowledgeSourceLeaseRepository,
    DocumentStoreKnowledgeSyncCheckpointRepository,
)
from intergrax.runtime.vendor_knowledge.sync_models import KnowledgeSyncRunStatus
from local_workspace_application.workspaces.connected_source_sync_sink import (
    ConnectedSourceSyncSinkContext,
    WorkspaceConnectedSourceKnowledgeSyncSink,
)
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingResult,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    IndexedSourceAudienceEligibilityV1,
    IndexedSourceSyncModeV1,
    WorkspaceIndexedSourceBinding,
    WorkspaceIndexedSourceBindingStatusV1,
)
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_MARKER = "SLACK-ORION-DEPLOYMENT-BLOCKER-7319"
_CONVERSATION_ID = "C01234567"
_CONNECTION = "conn.slack"
_TENANT = "tenant-a"
_WORKSPACE = "workspace-1"
_OLDEST = "1704067200.000001"
_LATEST = "1706745600.000001"
_ROOT_TS = "1704153600.000001"
_REPLY_TS = "1704153601.000001"
_EDITED_TS = "1704153602.000001"
_TS = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


def _message(
    *,
    message_ts: str,
    text: str,
    reply_count: int = 0,
    root_thread_ts: str | None = None,
    edited_at: datetime | None = None,
) -> SlackConversationMessage:
    created_at = parse_slack_ts(message_ts) or _TS
    return SlackConversationMessage(
        conversation_id=_CONVERSATION_ID,
        message_ts=message_ts,
        root_thread_ts=root_thread_ts,
        actor_provider_id="U111",
        text=text,
        subtype=None,
        created_at=created_at,
        edited_at=edited_at,
        reply_count=reply_count,
        files=(),
        provider_metadata={},
    )


class _SlackFakeBackend:
    def __init__(self) -> None:
        self._history_pages = [
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(
                    _message(
                        message_ts=_ROOT_TS,
                        text=f"root {_MARKER}",
                        reply_count=1,
                    ),
                ),
                next_cursor="history-2",
            ),
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(
                    _message(
                        message_ts=_EDITED_TS,
                        text=f"edited {_MARKER}",
                        edited_at=datetime(2024, 1, 3, 12, 0, tzinfo=timezone.utc),
                    ),
                ),
            ),
        ]
        self._reply_pages = [
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(
                    _message(
                        message_ts=_REPLY_TS,
                        text=f"reply {_MARKER}",
                        root_thread_ts=_ROOT_TS,
                    ),
                ),
            )
        ]
        self._content: dict[str, SlackConversationMessage] = {}
        self._content: dict[str, SlackConversationMessage] = {}

    async def list_accessible_conversations_page(self, *, cursor, limit):
        from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
            SlackConversationInventoryPage,
            SlackConversationSummary,
        )

        return SlackConversationInventoryPage(
            items=(
                SlackConversationSummary(
                    conversation_id=_CONVERSATION_ID,
                    kind=SlackConversationKind.PUBLIC_CHANNEL,
                    safe_name="#project-orion",
                    is_archived=False,
                    is_private=False,
                ),
            )
        )

    async def read_conversation_history_page(self, **kwargs: Any) -> SlackConversationMessagePage:
        if kwargs.get("cursor") is None and self._history_pages:
            self._history_pages = list(self._history_pages)
        page = self._history_pages.pop(0)
        for item in page.items:
            self._content[item.message_ts] = item
        return page

    async def read_thread_replies_page(self, **kwargs: Any) -> SlackConversationMessagePage:
        page = self._reply_pages.pop(0)
        for item in page.items:
            self._content[item.message_ts] = item
        return page

    async def read_exact_message(self, **kwargs: Any) -> SlackConversationExactMessageResult:
        message_ts = kwargs["message_ts"]
        message = self._content.get(message_ts)
        if message is None:
            message = _message(message_ts=message_ts, text="exact")
        revision = kwargs.get("expected_revision")
        if revision is not None and revision != compute_slack_conversation_message_revision(message):
            from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
                SlackConversationMessageChanged,
            )

            raise SlackConversationMessageChanged()
        return SlackConversationExactMessageResult(found=True, message=message)

    async def read_file_info(self, **kwargs: Any):
        raise NotImplementedError


@dataclass
class _Resolver:
    integration: SlackConversationChannelIntegration

    def resolve(self, *, source) -> SlackConversationChannelIntegration:
        return self.integration


class _RecordingIndexingService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._indexed_paths: set[str] = set()

    async def index_one(self, **kwargs: Any) -> WorkspaceDocumentIndexingResult:
        physical_path = kwargs["physical_path"]
        content = physical_path.read_text(encoding="utf-8")
        self.calls.append({**kwargs, "content": content})
        path = kwargs["logical_source_path"]
        unchanged = path in self._indexed_paths
        if not unchanged:
            self._indexed_paths.add(path)
        assert _MARKER in content
        return WorkspaceDocumentIndexingResult(
            indexed=not unchanged,
            unchanged=unchanged,
            document_id=f"doc-{len(self._indexed_paths)}",
            documents_indexed=0 if unchanged else 1,
        )


class _WorkspaceLookup:
    def require_workspace(self, *, tenant_id: str, workspace_id: str):
        return Workspace(
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            name="ws",
            status=WorkspaceStatus.ACTIVE,
            created_at=_NOW,
            updated_at=_NOW,
        )


@pytest.mark.asyncio
async def test_slack_connected_source_end_to_end_indexing_proof() -> None:
    document_store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(document_store)
    repo.put_workspace(
        Workspace(
            workspace_id=_WORKSPACE,
            tenant_id=_TENANT,
            name="ws",
            status=WorkspaceStatus.ACTIVE,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    binding_id = "ksb-slack-orion"
    source_id = "src:connected:orion-proof"
    indexed_binding_id = "idx-orion-proof"
    repo.put_knowledge_indexed_source_version_if_absent(
        WorkspaceIndexedSourceBinding(
            indexed_source_binding_id=indexed_binding_id,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            knowledge_source_binding_ref=binding_id,
            source_id=source_id,
            sync_mode=IndexedSourceSyncModeV1.FULL,
            status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE,
            audience_eligibility=IndexedSourceAudienceEligibilityV1.PERSONAL_ONLY,
            mutation_id="mut-1",
            effective_revision=1,
            semantic_identity_hash="a" * 64,
            created_at=_NOW,
            updated_at=_NOW,
            cached_safe_display_label="#project-orion",
        )
    )
    repo.put_knowledge_configuration_head_if_absent(
        __import__(
            "local_workspace_application.workspaces.knowledge_configuration_models",
            fromlist=["WorkspaceKnowledgeConfigurationHead"],
        ).WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=1,
            updated_at=_NOW,
        )
    )
    repo.put_source(
        WorkspaceSource(
            source_id=source_id,
            workspace_id=_WORKSPACE,
            tenant_id=_TENANT,
            source_type=WorkspaceSourceType.CONNECTED_SOURCE,
            path="",
            recursive=False,
            status=WorkspaceSourceStatus.REGISTERED,
            created_at=_NOW,
            knowledge_configuration_creation_mutation_id="mut-1",
            knowledge_configuration_visibility_revision=1,
        )
    )

    backend = _SlackFakeBackend()
    config = SlackConversationChannelIntegrationConfig(
        enabled=True,
        app_token="xapp-test",
        bot_token="xoxb-test",
    )
    integration = SlackConversationChannelIntegration.from_backend(
        backend,  # type: ignore[arg-type]
        enabled=True,
        config=config,
    )
    registry = KnowledgeAdapterRegistry()
    register_slack_conversation_knowledge_adapter(registry)
    resolver = _Resolver(integration=integration)
    facade = VendorKnowledgeFacadeService(
        tenant_id=_TENANT,
        resolver=resolver,
        adapter_registry=registry,
    )
    tenant_binding = KnowledgeSourceBinding(
        binding_id=binding_id,
        tenant_id=_TENANT,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        connection_ref=_CONNECTION,
        safe_display_name="#project-orion",
        scope=KnowledgeSourceScope(
            remote_scope_id=encode_slack_conversation_scope_id(
                conversation_id=_CONVERSATION_ID,
                conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
                oldest=_OLDEST,
                latest=_LATEST,
            ),
            remote_scope_type=SLACK_CONVERSATION_SCOPE_TYPE,
            safe_display_name="#project-orion",
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
    )
    binding_repo = DocumentStoreKnowledgeSourceBindingRepository(document_store)
    binding_repo.create(tenant_binding)
    binding_service = KnowledgeSourceBindingService(
        tenant_id=_TENANT,
        repository=binding_repo,
        integration_resolver=resolver,
        adapter_registry=registry,
    )
    indexing = _RecordingIndexingService()
    sink = WorkspaceConnectedSourceKnowledgeSyncSink(
        repository=repo,
        indexing_service=indexing,
        context=ConnectedSourceSyncSinkContext(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=source_id,
            indexed_source_binding_id=indexed_binding_id,
            knowledge_source_binding_ref=binding_id,
            operation_id="op-1",
        ),
    )
    coordinator = VendorKnowledgeSyncCoordinator(
        tenant_id=_TENANT,
        owner_id="proof",
        binding_service=binding_service,
        facade=facade,
        lease_repository=DocumentStoreKnowledgeSourceLeaseRepository(document_store),
        checkpoint_repository=DocumentStoreKnowledgeSyncCheckpointRepository(document_store),
        item_state_repository=DocumentStoreKnowledgeRemoteItemStateRepository(document_store),
        sink=sink,
        lease_ttl_seconds=30,
    )
    restart = True
    while True:
        result = await coordinator.reconcile_once(
            binding_id=binding_id,
            restart=restart,
        )
        assert result.status is KnowledgeSyncRunStatus.COMPLETED
        restart = False
        if not result.has_more:
            break
    assert len(indexing.calls) >= 2
    for call in indexing.calls:
        assert _MARKER in call["content"]
