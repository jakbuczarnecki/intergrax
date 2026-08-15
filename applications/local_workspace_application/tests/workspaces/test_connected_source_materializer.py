# © Artur Czarnecki. All rights reserved.

"""Provider-neutral indexed materialization contract proofs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from local_workspace_application.workspaces.connected_source_materializer import (
    ConnectedSourceContentMaterializerRegistry,
    GoogleCalendarStructuredRecordMaterializer,
    GoogleDocsStructuredRecordMaterializer,
    GoogleSheetsStructuredRecordMaterializer,
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
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.knowledge_read import (
    CONFLUENCE_PAGES_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.calendar import (
    GOOGLE_CALENDAR_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.docs import (
    GOOGLE_DOCS_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.sheets import (
    GOOGLE_SHEETS_SOURCE_KIND,
)
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
from intergrax.runtime.vendor_knowledge.confluence_indexed_materializers import (
    CONFLUENCE_PAGES_RICH_TEXT_SCHEMA,
    ConfluencePageRichTextMaterializer,
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


def _confluence_source() -> KnowledgeSourceRef:
    source = _source(
        provider_id=CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
        integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
        source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
    )
    return source.model_copy(
        update={
            "scope": source.scope.model_copy(
                update={
                    "remote_scope_id": "10000",
                    "remote_scope_type": "confluence_space",
                }
            )
        }
    )


def _confluence_content(*, body: str = "<p>Confluence indexed content</p>") -> KnowledgeContent:
    return KnowledgeContent(
        mode=KnowledgeContentMode.RICH_TEXT,
        rich_text=f"<h1>Confluence page</h1>{body}",
        mime_type=CONFLUENCE_PAGES_RICH_TEXT_SCHEMA,
        encoding="utf-8",
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


def _google_calendar_content(*, summary: str = "Planning meeting") -> KnowledgeContent:
    return KnowledgeContent(
        mode=KnowledgeContentMode.STRUCTURED_RECORD,
        structured_record={
            "schema": "google_workspace.calendar.event.knowledge.v1",
            "calendar_id": "team@example.com",
            "event": {
                "id": "google-event-1",
                "etag": '"google-etag-1"',
                "status": "confirmed",
                "event_type": "default",
                "summary": summary,
                "description": "Calendar event body",
                "location": "Room 1",
                "updated": "2026-01-02T11:00:00Z",
                "sequence": 1,
                "start": {
                    "date_time": "2026-01-02T12:00:00Z",
                    "time_zone": "Europe/Warsaw",
                },
                "end": {
                    "date_time": "2026-01-02T13:00:00Z",
                    "time_zone": "Europe/Warsaw",
                },
                "organizer": {
                    "email": "organizer@example.test",
                    "display_name": "Organizer",
                },
                "attendees": [
                    {
                        "email": "attendee@example.test",
                        "response_status": "accepted",
                    }
                ],
                "recurrence": [],
            },
        },
    )


def _google_calendar_source() -> KnowledgeSourceRef:
    source = _source(
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_CALENDAR_SOURCE_KIND,
    )
    return source.model_copy(
        update={
            "scope": source.scope.model_copy(
                update={
                    "remote_scope_id": "team@example.com",
                    "remote_scope_type": "google_workspace_calendar",
                }
            )
        }
    )


def _google_docs_source() -> KnowledgeSourceRef:
    source = _source(
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_DOCS_SOURCE_KIND,
    )
    return source.model_copy(
        update={
            "scope": source.scope.model_copy(
                update={
                    "remote_scope_id": "google-doc-1",
                    "remote_scope_type": "google_workspace_docs_document",
                }
            )
        }
    )


def _google_sheets_source() -> KnowledgeSourceRef:
    source = _source(
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_SHEETS_SOURCE_KIND,
    )
    return source.model_copy(
        update={
            "scope": source.scope.model_copy(
                update={
                    "remote_scope_id": "google-sheet-1",
                    "remote_scope_type": "google_workspace_sheets_spreadsheet",
                }
            )
        }
    )


def _google_docs_content() -> KnowledgeContent:
    return KnowledgeContent(
        mode=KnowledgeContentMode.STRUCTURED_RECORD,
        structured_record={
            "schema_version": "google_workspace.docs.document.knowledge.v1",
            "document_id": "google-doc-1",
            "title": "Docs indexed proof",
            "suggestions_view_mode": "PREVIEW_WITHOUT_SUGGESTIONS",
            "tabs": [
                {
                    "tab_id": "tab-1",
                    "title": "Main",
                    "parent_tab_id": None,
                    "index": 0,
                    "nesting_level": 0,
                    "list_ids": [],
                    "inline_object_ids": [],
                    "positioned_object_ids": [],
                    "segments": [
                        {
                            "kind": "BODY",
                            "segment_id": None,
                            "blocks": [
                                {
                                    "kind": "PARAGRAPH",
                                    "start_index": 1,
                                    "end_index": 24,
                                    "paragraph": {
                                        "elements": [
                                            {
                                                "kind": "TEXT_RUN",
                                                "start_index": 1,
                                                "end_index": 24,
                                                "text": "Docs indexed content",
                                                "reference_id": None,
                                                "auxiliary_text": None,
                                                "mime_type": None,
                                            }
                                        ],
                                        "named_style_type": "NORMAL_TEXT",
                                        "heading_id": None,
                                        "bullet": None,
                                        "positioned_object_ids": [],
                                    },
                                    "table": None,
                                    "children": [],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    )


def _google_sheets_content() -> KnowledgeContent:
    string_value = {
        "kind": "STRING",
        "text": "Sheets indexed content",
        "number": None,
        "boolean": None,
        "error": None,
    }
    return KnowledgeContent(
        mode=KnowledgeContentMode.STRUCTURED_RECORD,
        structured_record={
            "schema_version": "google_workspace.sheets.spreadsheet.knowledge.v1",
            "spreadsheet_id": "google-sheet-1",
            "title": "Sheets indexed proof",
            "locale": "en_US",
            "time_zone": "UTC",
            "recalculation_interval": "ON_CHANGE",
            "sheets": [
                {
                    "sheet_id": 1,
                    "title": "Sheet1",
                    "index": 0,
                    "sheet_type": "GRID",
                    "hidden": False,
                    "right_to_left": False,
                    "row_count": 10,
                    "column_count": 4,
                    "frozen_row_count": 0,
                    "frozen_column_count": 0,
                    "grid_data": [
                        {
                            "start_row_index": 0,
                            "start_column_index": 0,
                            "rows": [
                                {
                                    "row_index": 0,
                                    "cells": [
                                        {
                                            "row_index": 0,
                                            "column_index": 0,
                                            "user_entered_value": string_value,
                                            "effective_value": string_value,
                                            "formatted_value": None,
                                            "note": None,
                                            "effective_number_format": None,
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                    "merged_ranges": [],
                }
            ],
            "named_ranges": [],
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

    google_calendar = registry.resolve(
        _source(
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            source_kind=GOOGLE_CALENDAR_SOURCE_KIND,
        ),
        schema_name="google_workspace.calendar.event.knowledge.v1",
    )
    assert isinstance(google_calendar, GoogleCalendarStructuredRecordMaterializer)

    google_docs = registry.resolve(
        _google_docs_source(),
        schema_name="google_workspace.docs.document.knowledge.v1",
    )
    assert isinstance(google_docs, GoogleDocsStructuredRecordMaterializer)

    google_sheets = registry.resolve(
        _google_sheets_source(),
        schema_name="google_workspace.sheets.spreadsheet.knowledge.v1",
    )
    assert isinstance(google_sheets, GoogleSheetsStructuredRecordMaterializer)


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


def test_google_calendar_materializer_projects_bounded_content_and_stable_identity() -> None:
    source = _google_calendar_source()
    materializer = GoogleCalendarStructuredRecordMaterializer()
    first = materializer.materialize(
        source=source,
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        binding_id="binding-1",
        source_id="source-1",
        remote_id="google-event-1",
        content=_google_calendar_content(),
        revision=KnowledgeItemRevision(version="1", etag='"google-etag-1"'),
        permissions=None,
    )
    newer = materializer.materialize(
        source=source,
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        binding_id="binding-1",
        source_id="source-1",
        remote_id="google-event-1",
        content=_google_calendar_content(summary="Updated planning meeting"),
        revision=KnowledgeItemRevision(version="2", etag='"google-etag-2"'),
        permissions=None,
    )

    assert first.document_id == newer.document_id
    assert first.content_hash != newer.content_hash
    assert "Updated planning meeting" in newer.markdown
    assert "Organizer <organizer@example.test>" in newer.markdown
    assert "attendee@example.test (accepted)" in newer.markdown
    assert "Attachment bytes" in newer.markdown
    assert "complete recurrence expansion" not in newer.markdown.lower()
    assert "absence from an ordinary snapshot is not authoritative deletion" in newer.markdown
    assert newer.knowledge_document.provenance.provider_id == (
        GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID
    )
    assert newer.knowledge_document.provenance.source_id == "google-event-1"


def test_google_calendar_materializer_fails_closed_for_mode_and_identity() -> None:
    source = _google_calendar_source()
    materializer = GoogleCalendarStructuredRecordMaterializer()
    with pytest.raises(VendorKnowledgeMaterializationError):
        materializer.materialize(
            source=source,
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            binding_id="binding-1",
            source_id="source-1",
            remote_id="other-event",
            content=_google_calendar_content(),
            revision=None,
            permissions=None,
        )
    with pytest.raises(VendorKnowledgeMaterializationError):
        materializer.materialize(
            source=source,
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            binding_id="binding-1",
            source_id="source-1",
            remote_id="google-event-1",
            content=_google_calendar_content().model_copy(
                update={"mode": KnowledgeContentMode.BINARY}
            ),
            revision=None,
            permissions=None,
        )


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


def test_confluence_materializer_normalizes_storage_and_preserves_page_identity() -> None:
    source = _confluence_source()
    materializer = ConfluencePageRichTextMaterializer()
    first = materializer.materialize(
        source=source,
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        binding_id="binding-1",
        source_id="source-1",
        remote_id="20001",
        content=_confluence_content(
            body=(
                "<p>Page body marker</p><table><tr><td>Cell A</td>"
                "<td>Cell B</td></tr></table><script>do_not_index()</script>"
            )
        ),
        revision=KnowledgeItemRevision(version="3"),
        permissions=None,
    )
    newer = materializer.materialize(
        source=source,
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        binding_id="binding-1",
        source_id="source-1",
        remote_id="20001",
        content=_confluence_content(body="<p>Updated page body marker</p>"),
        revision=KnowledgeItemRevision(version="4"),
        permissions=None,
    )

    assert first.document_id == newer.document_id
    assert first.content_hash != newer.content_hash
    assert first.source_revision != newer.source_revision
    assert "# Confluence page" in first.markdown
    assert "Page body marker" in first.markdown
    assert "Cell A | Cell B" in first.markdown
    assert "do_not_index" not in first.markdown
    assert first.knowledge_document.provenance.source_id == "20001"


def test_confluence_materializer_registry_and_fail_closed_contract() -> None:
    source = _confluence_source()
    registry = default_connected_source_materializer_registry()
    materializer = registry.resolve(
        source,
        schema_name=CONFLUENCE_PAGES_RICH_TEXT_SCHEMA,
    )
    assert isinstance(materializer, ConfluencePageRichTextMaterializer)

    kwargs = {
        "source": source,
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "binding_id": "binding-1",
        "source_id": "source-1",
        "remote_id": "20001",
        "content": _confluence_content(),
        "revision": KnowledgeItemRevision(version="3"),
        "permissions": None,
    }
    for invalid in (
        {"source": source.model_copy(update={"provider_id": "other"})},
        {
            "content": _confluence_content().model_copy(
                update={"mode": KnowledgeContentMode.BINARY}
            )
        },
        {
            "content": _confluence_content().model_copy(
                update={"mime_type": "text/plain"}
            )
        },
        {"remote_id": "not-a-page-id"},
        {
            "content": _confluence_content(
                body="<script>only unhelpful content</script>"
            )
        },
    ):
        with pytest.raises(VendorKnowledgeMaterializationError):
            materializer.materialize(**{**kwargs, **invalid})


@pytest.mark.asyncio
async def test_confluence_materializer_reaches_generic_index_service(tmp_path: Path) -> None:
    materialized = ConfluencePageRichTextMaterializer().materialize(
        source=_confluence_source(),
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        binding_id="binding-1",
        source_id="source-1",
        remote_id="20001",
        content=_confluence_content(body="<p>Search body marker</p>"),
        revision=KnowledgeItemRevision(version="3"),
        permissions=None,
    )
    physical_path = tmp_path / materialized.safe_file_name
    physical_path.write_text(materialized.markdown, encoding="utf-8")

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
        operation_id="operation-confluence-1",
        physical_path=physical_path,
        logical_source_path=materialized.logical_source_path,
        safe_file_name=materialized.safe_file_name,
        content_hash=materialized.content_hash,
        document_id=materialized.document_id,
        materialization_ownership=KnowledgeMaterializationOwnershipV1.connected(
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            source_id="source-1",
            indexed_source_binding_id="indexed-binding-1",
            knowledge_source_binding_ref="binding-1",
            delivery_id="delivery-confluence-1",
            remote_id="20001",
            materialization_sequence=1,
        ),
    )

    assert result.indexed
    assert result.document_id == materialized.document_id
    assert "Search body marker" in physical_path.read_text(encoding="utf-8")


async def _assert_google_materialized_document_indexed(
    *,
    tmp_path: Path,
    materialized,
    remote_id: str,
    operation_id: str,
) -> None:
    physical_path = tmp_path / materialized.safe_file_name
    physical_path.write_text(materialized.markdown, encoding="utf-8")

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
        operation_id=operation_id,
        physical_path=physical_path,
        logical_source_path=materialized.logical_source_path,
        safe_file_name=materialized.safe_file_name,
        content_hash=materialized.content_hash,
        document_id=materialized.document_id,
        materialization_ownership=KnowledgeMaterializationOwnershipV1.connected(
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            source_id="source-1",
            indexed_source_binding_id="indexed-binding-1",
            knowledge_source_binding_ref="binding-1",
            delivery_id=operation_id,
            remote_id=remote_id,
            materialization_sequence=1,
        ),
    )
    assert result.indexed
    assert result.document_id == materialized.knowledge_document.identity.document_id


@pytest.mark.asyncio
async def test_google_docs_materializer_reaches_generic_index_boundary(tmp_path: Path) -> None:
    materialized = GoogleDocsStructuredRecordMaterializer().materialize(
        source=_google_docs_source(),
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        binding_id="binding-1",
        source_id="source-1",
        remote_id="google-doc-1",
        content=_google_docs_content(),
        revision=KnowledgeItemRevision(version=None),
        permissions=None,
    )
    assert "Docs indexed content" in materialized.markdown
    await _assert_google_materialized_document_indexed(
        tmp_path=tmp_path,
        materialized=materialized,
        remote_id="google-doc-1",
        operation_id="operation-google-docs-1",
    )


@pytest.mark.asyncio
async def test_google_sheets_materializer_reaches_generic_index_boundary(
    tmp_path: Path,
) -> None:
    materialized = GoogleSheetsStructuredRecordMaterializer().materialize(
        source=_google_sheets_source(),
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        binding_id="binding-1",
        source_id="source-1",
        remote_id="google-sheet-1",
        content=_google_sheets_content(),
        revision=KnowledgeItemRevision(version=None),
        permissions=None,
    )
    assert "Sheets indexed content" in materialized.markdown
    await _assert_google_materialized_document_indexed(
        tmp_path=tmp_path,
        materialized=materialized,
        remote_id="google-sheet-1",
        operation_id="operation-google-sheets-1",
    )


@pytest.mark.parametrize(
    ("materializer", "source", "content", "remote_id"),
    [
        pytest.param(
            GoogleDocsStructuredRecordMaterializer(),
            _google_docs_source(),
            _google_docs_content(),
            "google-doc-1",
            id="docs",
        ),
        pytest.param(
            GoogleSheetsStructuredRecordMaterializer(),
            _google_sheets_source(),
            _google_sheets_content(),
            "google-sheet-1",
            id="sheets",
        ),
    ],
)
def test_google_materializers_fail_closed_for_identity_mode_schema_and_content(
    materializer,
    source: KnowledgeSourceRef,
    content: KnowledgeContent,
    remote_id: str,
) -> None:
    kwargs = {
        "source": source,
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "binding_id": "binding-1",
        "source_id": "source-1",
        "remote_id": remote_id,
        "content": content,
        "revision": None,
        "permissions": None,
    }
    with pytest.raises(VendorKnowledgeMaterializationError):
        materializer.materialize(
            **{**kwargs, "source": source.model_copy(update={"provider_id": "other"})}
        )
    with pytest.raises(VendorKnowledgeMaterializationError):
        materializer.materialize(
            **{
                **kwargs,
                "content": content.model_copy(
                    update={"mode": KnowledgeContentMode.BINARY}
                ),
            }
        )
    malformed = content.model_copy(update={"structured_record": {}})
    with pytest.raises(VendorKnowledgeMaterializationError):
        materializer.materialize(**{**kwargs, "content": malformed})
    with pytest.raises(VendorKnowledgeMaterializationError):
        materializer.materialize(
            **{
                **kwargs,
                "remote_id": f"{remote_id}-other",
            }
        )


@pytest.mark.asyncio
async def test_google_calendar_document_enters_existing_generic_index_service(
    tmp_path: Path,
) -> None:
    materializer = default_connected_source_materializer_registry().resolve(
        _google_calendar_source(),
        schema_name="google_workspace.calendar.event.knowledge.v1",
    )
    materialized = materializer.materialize(
        source=_google_calendar_source(),
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        binding_id="binding-1",
        source_id="source-1",
        remote_id="google-event-1",
        content=_google_calendar_content(),
        revision=KnowledgeItemRevision(version="google-1"),
        permissions=None,
    )
    document = materialized.knowledge_document
    physical_path = tmp_path / materialized.safe_file_name
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
        operation_id="operation-google-1",
        physical_path=physical_path,
        logical_source_path=materialized.logical_source_path,
        safe_file_name=materialized.safe_file_name,
        content_hash=materialized.content_hash,
        document_id=materialized.document_id,
        materialization_ownership=KnowledgeMaterializationOwnershipV1.connected(
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            source_id="source-1",
            indexed_source_binding_id="indexed-binding-1",
            knowledge_source_binding_ref="binding-1",
            delivery_id="delivery-google-1",
            remote_id="google-event-1",
            materialization_sequence=1,
        ),
    )
    assert result.indexed
    assert result.document_id == document.identity.document_id
