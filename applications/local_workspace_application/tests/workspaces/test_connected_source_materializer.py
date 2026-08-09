# © Artur Czarnecki. All rights reserved.

"""Provider-neutral indexed materialization contract proofs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from local_workspace_application.workspaces.connected_source_materializer import (
    ConnectedSourceContentMaterializerRegistry,
    MsGraphCalendarStructuredRecordMaterializer,
    MsGraphMailStructuredRecordMaterializer,
    MsGraphTeamsChannelStructuredRecordMaterializer,
    MsGraphTeamsChatStructuredRecordMaterializer,
    SlackConversationStructuredRecordMaterializer,
    default_connected_source_materializer_registry,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceSyncSinkError,
)
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingService,
)
from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationOwnershipV1,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_inventory import (
    MSGRAPH_CALENDAR_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (
    MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
    MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_folders import (
    MSGRAPH_MAIL_SOURCE_KIND,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SLACK_CONVERSATION_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.live.slack.registration import (
    build_slack_vendor_knowledge_source_plugin,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeItemRevision,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.indexed_materialization import (
    VendorKnowledgeMaterializationError,
)
from intergrax.runtime.vendor_knowledge.plugin import (
    VendorKnowledgeSourcePluginRegistry,
)

pytestmark = pytest.mark.unit


def _source(
    *,
    provider_id: str,
    integration_kind: IntegrationCategory,
    source_kind: str,
) -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        connection_ref="connection-1",
        scope=KnowledgeSourceScope(
            remote_scope_id="scope-1",
            remote_scope_type=source_kind,
            safe_display_name="Knowledge source",
        ),
    )


def _slack_content(text: str) -> KnowledgeContent:
    return KnowledgeContent(
        mode=KnowledgeContentMode.STRUCTURED_RECORD,
        structured_record={
            "schema": "slack.conversation.message.knowledge.v1",
            "provider": "slack",
            "source_kind": "slack_conversation",
            "conversation": {"safe_display_name": "#project"},
            "message": {"message_ts": "1704153600.000001"},
            "thread": {"root_thread_ts": None, "reply_count": 0},
            "actor": {"provider_id": "U111"},
            "text": text,
            "timestamps": {"created_at": "2024-01-02T12:00:00+00:00", "edited_at": None},
            "edit_state": {"edited": False},
            "safe_file_inventory": [],
        },
    )


def _graph_content() -> KnowledgeContent:
    return KnowledgeContent(
        mode=KnowledgeContentMode.STRUCTURED_RECORD,
        structured_record={
            "schema": "msgraph.teams-chat.message.knowledge.v1",
            "state": "active",
            "subject": "Project update",
            "body": {"kind": "text", "content": "Graph indexed content"},
            "sender": {"display_name": "Alex"},
            "created_at": "2024-01-02T12:00:00+00:00",
            "last_modified_at": "2024-01-02T12:00:00+00:00",
            "last_edited_at": None,
            "message_type": "message",
            "importance": "normal",
            "locale": "en-US",
            "attachments": {},
        },
    )


def _graph_mail_content(*, body_text: str = "Mail body") -> KnowledgeContent:
    return KnowledgeContent(
        mode=KnowledgeContentMode.STRUCTURED_RECORD,
        structured_record={
            "schema": "msgraph.mail.message.knowledge.v1",
            "subject": "Quarterly report",
            "conversation_id": "conversation-1",
            "internet_message_id": "<message-1@example.test>",
            "body_text": body_text,
            "unique_body_text": "Mail",
            "from": {"display_name": "Sender", "address": "sender@example.test"},
            "sender": {"display_name": "Sender", "address": "sender@example.test"},
            "reply_to": [],
            "to_recipients": [{"display_name": "Recipient", "address": "recipient@example.test"}],
            "cc_recipients": [],
            "bcc_recipients": [],
            "created_at": "2026-01-02T12:00:00+00:00",
            "last_modified_at": "2026-01-02T12:00:00+00:00",
            "received_at": "2026-01-02T12:00:00+00:00",
            "sent_at": "2026-01-02T12:00:00+00:00",
            "is_read": True,
            "is_draft": False,
            "importance": "normal",
            "attachments": {
                "has_attachments": True,
                "inventory_included": False,
                "binary_content_included": False,
            },
        },
    )


def _graph_teams_channel_content(
    *,
    body: str = "Channel root post",
    message_kind: str = "root",
) -> KnowledgeContent:
    return KnowledgeContent(
        mode=KnowledgeContentMode.STRUCTURED_RECORD,
        structured_record={
            "schema": "msgraph.teams-channel.message.knowledge.v1",
            "message_kind": message_kind,
            "state": "active",
            "subject": "Channel update",
            "body": {"kind": "text", "content": body},
            "sender": {"display_name": "Alex", "address": "alex@example.test"},
            "created_at": "2026-01-02T12:00:00+00:00",
            "last_modified_at": "2026-01-02T12:05:00+00:00",
            "last_edited_at": None,
            "message_type": "message",
            "importance": "normal",
            "locale": "en-US",
            "event_detail_type": None,
            "mentions": [],
            "reactions": [],
            "attachments": {
                "inventory_included": True,
                "binary_content_included": False,
                "hosted_content_included": False,
                "reference_urls_included": False,
                "items": [],
            },
        },
    )


def _graph_calendar_content(*, body: str = "Calendar event body") -> KnowledgeContent:
    return KnowledgeContent(
        mode=KnowledgeContentMode.STRUCTURED_RECORD,
        structured_record={
            "schema": "msgraph.calendar.event.knowledge.v1",
            "event_type": "single_instance",
            "subject": "Planning meeting",
            "body": {"kind": "text", "content": body, "preview": body},
            "start_at": "2026-01-02T12:00:00+00:00",
            "end_at": "2026-01-02T13:00:00+00:00",
            "original_start_at": None,
            "original_start_time_zone": None,
            "original_end_time_zone": None,
            "created_at": "2026-01-01T12:00:00+00:00",
            "last_modified_at": "2026-01-02T11:00:00+00:00",
            "organizer": {"display_name": "Alex", "address": "alex@example.test"},
            "attendees": [],
            "location": {"display_name": "Room 1"},
            "locations": [],
            "recurrence": None,
            "series_master_id": None,
            "cancelled_occurrence_ids": [],
            "categories": [],
            "i_cal_uid": "event-1@example.test",
            "importance": "normal",
            "sensitivity": "normal",
            "show_as": "busy",
            "response_status": {"response": "organizer", "responded_at": None},
            "is_all_day": False,
            "is_cancelled": False,
            "is_draft": False,
            "is_organizer": True,
            "is_online_meeting": False,
            "online_meeting_provider": "unknown",
            "has_attachments": False,
            "hide_attendees": False,
            "allow_new_time_proposals": True,
            "response_requested": True,
            "is_reminder_on": True,
            "reminder_minutes_before_start": 15,
            "attachments": {
                "attachment_inventory_included": False,
                "attachment_binary_content_included": False,
                "items": [],
            },
        },
    )


def test_registry_resolves_indexed_materializer_by_source_identity() -> None:
    registry = default_connected_source_materializer_registry()
    materializer = registry.resolve(
        _source(
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
            source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        ),
        schema_name="slack.conversation.message.knowledge.v1",
    )
    assert isinstance(materializer, SlackConversationStructuredRecordMaterializer)

    graph = registry.resolve(
        _source(
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
        ),
        schema_name="msgraph.teams-chat.message.knowledge.v1",
    )
    assert isinstance(graph, MsGraphTeamsChatStructuredRecordMaterializer)

    mail = registry.resolve(
        _source(
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            source_kind=MSGRAPH_MAIL_SOURCE_KIND,
        ),
        schema_name="msgraph.mail.message.knowledge.v1",
    )
    assert isinstance(mail, MsGraphMailStructuredRecordMaterializer)

    channel = registry.resolve(
        _source(
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            source_kind=MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
        ),
        schema_name="msgraph.teams-channel.message.knowledge.v1",
    )
    assert isinstance(channel, MsGraphTeamsChannelStructuredRecordMaterializer)

    calendar = registry.resolve(
        _source(
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            source_kind=MSGRAPH_CALENDAR_SOURCE_KIND,
        ),
        schema_name="msgraph.calendar.event.knowledge.v1",
    )
    assert isinstance(calendar, MsGraphCalendarStructuredRecordMaterializer)


def test_missing_indexed_runtime_registration_fails_closed() -> None:
    plugins = VendorKnowledgeSourcePluginRegistry()
    plugins.register(build_slack_vendor_knowledge_source_plugin())
    registry = ConnectedSourceContentMaterializerRegistry(
        materializers=(),
        plugin_registry=plugins,
    )
    with pytest.raises(
        ConnectedSourceSyncSinkError,
        match="connected_source_indexed_materializer_unregistered",
    ):
        registry.resolve(
            _source(
                provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
                integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
                source_kind=SLACK_CONVERSATION_SOURCE_KIND,
            ),
            schema_name="slack.conversation.message.knowledge.v1",
        )


def test_materializer_identity_mismatch_is_rejected() -> None:
    materializer = SlackConversationStructuredRecordMaterializer()
    with pytest.raises(
        ConnectedSourceSyncSinkError,
        match="connected_source_materializer_identity_mismatch",
    ):
        materializer.materialize(
            source=_source(
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
            ),
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            binding_id="binding-1",
            source_id="source-1",
            remote_id="remote-1",
            content=_slack_content("wrong source"),
            revision=KnowledgeItemRevision(version="1"),
            permissions=None,
        )


def test_stable_document_identity_survives_newer_revision() -> None:
    source = _source(
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        source_kind=SLACK_CONVERSATION_SOURCE_KIND,
    )
    materializer = SlackConversationStructuredRecordMaterializer()
    first = materializer.materialize(
        source=source,
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        binding_id="binding-1",
        source_id="source-1",
        remote_id="remote-1",
        content=_slack_content("version one"),
        revision=KnowledgeItemRevision(version="1"),
        permissions=None,
    )
    newer = materializer.materialize(
        source=source,
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        binding_id="binding-1",
        source_id="source-1",
        remote_id="remote-1",
        content=_slack_content("version two"),
        revision=KnowledgeItemRevision(version="2"),
        permissions=None,
    )
    assert first.document_id == newer.document_id
    assert first.knowledge_document.identity.document_id == newer.document_id
    assert first.content_hash != newer.content_hash
    assert first.source_revision != newer.source_revision


def test_graph_uses_same_canonical_bridge_and_preserves_provenance() -> None:
    source = _source(
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
    )
    materializer = default_connected_source_materializer_registry().resolve(
        source,
        schema_name="msgraph.teams-chat.message.knowledge.v1",
    )
    document = materializer.materialize(
        source=source,
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        binding_id="binding-1",
        source_id="source-1",
        remote_id="graph-remote-1",
        content=_graph_content(),
        revision=KnowledgeItemRevision(version="graph-1"),
        permissions=None,
    ).knowledge_document
    assert "Graph indexed content" in document.content
    assert document.scope.tenant_id == "tenant-1"
    assert document.scope.workspace_id == "workspace-1"
    assert document.provenance.provider_id == MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    assert document.provenance.source_kind == MSGRAPH_TEAMS_CHAT_SOURCE_KIND


def test_graph_mail_materializer_preserves_mail_scope_and_exclusions() -> None:
    source = _source(
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_MAIL_SOURCE_KIND,
    )
    document = MsGraphMailStructuredRecordMaterializer().materialize(
        source=source,
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        binding_id="binding-1",
        source_id="source-1",
        remote_id="graph-mail-remote-1",
        content=_graph_mail_content(),
        revision=KnowledgeItemRevision(version="mail-1"),
        permissions=None,
    ).knowledge_document

    assert "Mail body" in document.content
    assert "Thread: conversation membership metadata only" in document.content
    assert "attachment inventory and binary content are not included" in document.content
    assert document.scope.tenant_id == "tenant-1"
    assert document.provenance.provider_id == MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    assert document.provenance.source_kind == MSGRAPH_MAIL_SOURCE_KIND
    assert document.provenance.source_id == "graph-mail-remote-1"


def test_graph_teams_channel_materializer_keeps_root_projection_bounded() -> None:
    source = _source(
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
    )
    materializer = MsGraphTeamsChannelStructuredRecordMaterializer()
    document = materializer.materialize(
        source=source,
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        binding_id="binding-1",
        source_id="source-1",
        remote_id="channel-root-1",
        content=_graph_teams_channel_content(),
        revision=KnowledgeItemRevision(version="channel-1"),
        permissions=None,
    ).knowledge_document

    assert "Channel root post" in document.content
    assert "deletedDateTime" in document.content
    assert "hosted content" in document.content
    assert document.provenance.source_id == "channel-root-1"

    with pytest.raises(VendorKnowledgeMaterializationError):
        materializer.materialize(
            source=source,
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            binding_id="binding-1",
            source_id="source-1",
            remote_id="channel-reply-1",
            content=_graph_teams_channel_content(message_kind="reply"),
            revision=KnowledgeItemRevision(version="reply-1"),
            permissions=None,
        )


def test_graph_calendar_materializer_preserves_identity_across_revision() -> None:
    source = _source(
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_CALENDAR_SOURCE_KIND,
    )
    materializer = MsGraphCalendarStructuredRecordMaterializer()
    first = materializer.materialize(
        source=source,
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        binding_id="binding-1",
        source_id="source-1",
        remote_id="calendar-event-1",
        content=_graph_calendar_content(),
        revision=KnowledgeItemRevision(version="calendar-1"),
        permissions=None,
    )
    newer = materializer.materialize(
        source=source,
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        binding_id="binding-1",
        source_id="source-1",
        remote_id="calendar-event-1",
        content=_graph_calendar_content(body="Updated event body"),
        revision=KnowledgeItemRevision(version="calendar-2"),
        permissions=None,
    )

    assert first.document_id == newer.document_id
    assert first.content_hash != newer.content_hash
    assert "Planning meeting" in newer.markdown
    assert "binary content is not included" in newer.markdown


@pytest.mark.asyncio
async def test_graph_document_enters_existing_generic_index_service(tmp_path: Path) -> None:
    source = _source(
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
    )
    materializer = default_connected_source_materializer_registry().resolve(
        source,
        schema_name="msgraph.teams-chat.message.knowledge.v1",
    )
    document = materializer.materialize(
        source=source,
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        binding_id="binding-1",
        source_id="source-1",
        remote_id="graph-remote-1",
        content=_graph_content(),
        revision=KnowledgeItemRevision(version="graph-1"),
        permissions=None,
    ).knowledge_document
    physical_path = tmp_path / "graph-message.md"
    physical_path.write_text(document.content, encoding="utf-8")

    class _Executor:
        async def execute(self, _task):
            return SimpleNamespace(
                metadata={"ingest_summary": {"used": True, "num_chunks": 1}}
            )

    indexing = WorkspaceDocumentIndexingService(
        ManagedWorkspaceRepository(InMemoryDocumentStore()),
        _Executor(),
    )
    result = await indexing.index_connected_source_one(
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        source_id="source-1",
        operation_id="operation-1",
        physical_path=physical_path,
        logical_source_path="connected/msgraph_teams_chat-message/graph.md",
        safe_file_name="graph-message.md",
        content_hash=document.provenance.content_hash or "missing",
        document_id=document.identity.document_id,
        materialization_ownership=KnowledgeMaterializationOwnershipV1.connected(
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            source_id="source-1",
            indexed_source_binding_id="indexed-binding-1",
            knowledge_source_binding_ref="binding-1",
            delivery_id="delivery-1",
            remote_id="graph-remote-1",
            materialization_sequence=1,
        ),
    )
    assert result.indexed
    assert result.document_id == document.identity.document_id
