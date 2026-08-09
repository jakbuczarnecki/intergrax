# © Artur Czarnecki. All rights reserved.

"""Materialize vendor knowledge structured records into LKW indexable documents."""

from __future__ import annotations

import hashlib
from typing import Literal, Protocol, runtime_checkable

from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceSyncSinkError,
)
from pydantic import BaseModel, ConfigDict, Field

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
    MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SLACK_CONVERSATION_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.indexed_materialization import (
    MaterializedConnectedSourceDocument,
    build_materialized_connected_source_document,
)
from intergrax.runtime.vendor_knowledge.confluence_indexed_materializers import (
    ConfluencePageRichTextMaterializer,
)
from intergrax.runtime.vendor_knowledge.ms365_graph_indexed_materializers import (
    MsGraphCalendarStructuredRecordMaterializer,
    MsGraphMailStructuredRecordMaterializer,
    MsGraphTeamsChannelStructuredRecordMaterializer,
)
from intergrax.runtime.vendor_knowledge.google_workspace_indexed_materializers import (
    GoogleCalendarStructuredRecordMaterializer,
    GoogleDocsStructuredRecordMaterializer,
    GoogleSheetsStructuredRecordMaterializer,
)
from intergrax.runtime.vendor_knowledge.jira_indexed_materializers import (
    JiraIssueStructuredRecordMaterializer,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeItemRevision,
    KnowledgePermissions,
    KnowledgeSourceRef,
)
from intergrax.runtime.vendor_knowledge.plugin import (
    VendorKnowledgeMode,
    VendorKnowledgeSourceIdentity,
    VendorKnowledgeSourcePluginRegistry,
)
from intergrax.runtime.vendor_knowledge.plugin_composition import (
    build_default_vendor_knowledge_source_plugin_registry,
)

_SLACK_CONVERSATION_MESSAGE_SCHEMA = "slack.conversation.message.knowledge.v1"
_MSGRAPH_TEAMS_CHAT_MESSAGE_SCHEMA = "msgraph.teams-chat.message.knowledge.v1"
_REMOTE_HASH_PREFIX_LEN = 16

_SLACK_IDENTITY = VendorKnowledgeSourceIdentity(
    provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    integration_category=IntegrationCategory.CONVERSATION_CHANNEL,
    source_kind=SLACK_CONVERSATION_SOURCE_KIND,
)
_MSGRAPH_TEAMS_CHAT_IDENTITY = VendorKnowledgeSourceIdentity(
    provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    integration_category=IntegrationCategory.COLLABORATION_SUITE,
    source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
)
class _SlackConversationMessageKnowledgeRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    schema_: Literal["slack.conversation.message.knowledge.v1"] = Field(
        alias="schema"
    )
    provider: Literal["slack"]
    source_kind: Literal["slack_conversation"]
    conversation: dict[str, object]
    message: dict[str, object]
    thread: dict[str, object]
    actor: dict[str, object]
    text: str
    timestamps: dict[str, object]
    edit_state: dict[str, object]
    safe_file_inventory: list[dict[str, object]] = Field(default_factory=list)


class _MsGraphTeamsChatMessageKnowledgeRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    schema_: Literal["msgraph.teams-chat.message.knowledge.v1"] = Field(
        alias="schema"
    )
    state: Literal["active"]
    subject: str | None = None
    body: dict[str, object]
    sender: dict[str, object]
    created_at: str
    last_modified_at: str
    last_edited_at: str | None = None
    message_type: str
    importance: str
    locale: str | None = None
    attachments: dict[str, object] = Field(default_factory=dict)


@runtime_checkable
class ConnectedSourceContentMaterializer(Protocol):
    identity: VendorKnowledgeSourceIdentity
    runtime_ref: str
    schema_name: str

    def materialize(
        self,
        *,
        source: KnowledgeSourceRef,
        tenant_id: str,
        workspace_id: str,
        binding_id: str,
        source_id: str,
        remote_id: str,
        content: KnowledgeContent,
        revision: KnowledgeItemRevision | None,
        permissions: KnowledgePermissions | None,
    ) -> MaterializedConnectedSourceDocument:
        ...


class ConnectedSourceContentMaterializerRegistry:
    """Provider-neutral indexed materializer registry resolved through VK-2."""

    def __init__(
        self,
        materializers: tuple[ConnectedSourceContentMaterializer, ...],
        *,
        plugin_registry: VendorKnowledgeSourcePluginRegistry | None = None,
    ) -> None:
        self._by_schema: dict[str, ConnectedSourceContentMaterializer] = {}
        self._by_identity: dict[
            tuple[str, IntegrationCategory, str],
            ConnectedSourceContentMaterializer,
        ] = {}
        self._by_runtime_ref: dict[str, ConnectedSourceContentMaterializer] = {}
        self._plugins = plugin_registry or _default_plugin_registry()
        for item in materializers:
            if item.schema_name in self._by_schema:
                raise ConnectedSourceSyncSinkError("connected_source_materializer_duplicate")
            if item.identity.key in self._by_identity:
                raise ConnectedSourceSyncSinkError("connected_source_materializer_identity_duplicate")
            if item.runtime_ref in self._by_runtime_ref:
                raise ConnectedSourceSyncSinkError("connected_source_materializer_runtime_duplicate")
            self._by_schema[item.schema_name] = item
            self._by_identity[item.identity.key] = item
            self._by_runtime_ref[item.runtime_ref] = item

    def resolve(
        self,
        source: KnowledgeSourceRef | str,
        *,
        schema_name: str | None = None,
    ) -> ConnectedSourceContentMaterializer:
        if isinstance(source, str):
            materializer = self._by_schema.get(source)
            if materializer is None:
                raise ConnectedSourceSyncSinkError("connected_source_materializer_unsupported")
            return materializer
        identity = VendorKnowledgeSourceIdentity(
            provider_id=source.provider_id,
            integration_category=source.integration_kind,
            source_kind=source.source_kind,
        )
        plugin = self._plugins.lookup(identity)
        if plugin is None or not plugin.supports(VendorKnowledgeMode.INDEXED):
            raise ConnectedSourceSyncSinkError("connected_source_indexed_capability_unregistered")
        capability = plugin.capability(VendorKnowledgeMode.INDEXED)
        assert capability is not None
        materializer = self._by_runtime_ref.get(capability.runtime_ref)
        if materializer is None:
            raise ConnectedSourceSyncSinkError("connected_source_indexed_materializer_unregistered")
        if materializer.identity != identity:
            raise ConnectedSourceSyncSinkError("connected_source_materializer_identity_mismatch")
        if schema_name is not None and materializer.schema_name != schema_name:
            raise ConnectedSourceSyncSinkError("connected_source_materializer_schema_mismatch")
        return materializer


class SlackConversationStructuredRecordMaterializer:
    identity = _SLACK_IDENTITY
    runtime_ref = "indexed-source:slack:slack_conversation"
    schema_name = _SLACK_CONVERSATION_MESSAGE_SCHEMA

    def materialize(
        self,
        *,
        source: KnowledgeSourceRef,
        tenant_id: str,
        workspace_id: str,
        binding_id: str,
        source_id: str,
        remote_id: str,
        content: KnowledgeContent,
        revision: KnowledgeItemRevision | None,
        permissions: KnowledgePermissions | None,
    ) -> MaterializedConnectedSourceDocument:
        _validate_materializer_source(self.identity, source)
        if content.mode is not KnowledgeContentMode.STRUCTURED_RECORD:
            raise ConnectedSourceSyncSinkError("connected_source_content_mode_invalid")
        record = content.structured_record
        if not isinstance(record, dict):
            raise ConnectedSourceSyncSinkError("connected_source_structured_record_invalid")
        try:
            validated = _SlackConversationMessageKnowledgeRecord.model_validate(record)
        except ValueError:
            raise ConnectedSourceSyncSinkError("connected_source_structured_record_invalid") from None

        safe_conversation_label = _safe_conversation_label(validated)
        markdown = _render_slack_message_markdown(
            safe_conversation_label=safe_conversation_label,
            record=validated,
        )
        remote_hash_prefix = hashlib.sha256(remote_id.encode("utf-8")).hexdigest()[
            :_REMOTE_HASH_PREFIX_LEN
        ]
        safe_file_name = f"slack-message-{remote_hash_prefix}.md"
        return _build_materialized_document(
            identity=self.identity,
            source=source,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            remote_id=remote_id,
            markdown=markdown,
            safe_file_name=safe_file_name,
            revision=revision,
            permissions=permissions,
        )


class MsGraphTeamsChatStructuredRecordMaterializer:
    identity = _MSGRAPH_TEAMS_CHAT_IDENTITY
    runtime_ref = "indexed-source:ms365_graph:teams_chat"
    schema_name = _MSGRAPH_TEAMS_CHAT_MESSAGE_SCHEMA

    def materialize(
        self,
        *,
        source: KnowledgeSourceRef,
        tenant_id: str,
        workspace_id: str,
        binding_id: str,
        source_id: str,
        remote_id: str,
        content: KnowledgeContent,
        revision: KnowledgeItemRevision | None,
        permissions: KnowledgePermissions | None,
    ) -> MaterializedConnectedSourceDocument:
        _validate_materializer_source(self.identity, source)
        if content.mode is not KnowledgeContentMode.STRUCTURED_RECORD:
            raise ConnectedSourceSyncSinkError("connected_source_content_mode_invalid")
        record = content.structured_record
        if not isinstance(record, dict):
            raise ConnectedSourceSyncSinkError("connected_source_structured_record_invalid")
        try:
            validated = _MsGraphTeamsChatMessageKnowledgeRecord.model_validate(record)
        except ValueError:
            raise ConnectedSourceSyncSinkError("connected_source_structured_record_invalid") from None
        body = validated.body.get("content")
        if not isinstance(body, str) or not body.strip():
            raise ConnectedSourceSyncSinkError("connected_source_structured_record_invalid")
        sender = validated.sender.get("display_name") or validated.sender.get("provider_id")
        sender_text = sender.strip() if isinstance(sender, str) else ""
        lines = [
            f"# {(validated.subject or 'Teams chat message').strip() or 'Teams chat message'}",
            "",
            body.strip(),
        ]
        if sender_text:
            lines.extend(["", f"Sender: {sender_text}"])
        lines.extend(
            [
                f"Created at: {validated.created_at}",
                f"Last modified at: {validated.last_modified_at}",
                "",
                f"Provider: {self.identity.provider_id}",
                f"Source kind: {self.identity.source_kind}",
                "",
            ]
        )
        remote_hash_prefix = hashlib.sha256(remote_id.encode("utf-8")).hexdigest()[
            :_REMOTE_HASH_PREFIX_LEN
        ]
        return _build_materialized_document(
            identity=self.identity,
            source=source,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            remote_id=remote_id,
            markdown="\n".join(lines),
            safe_file_name=f"teams-chat-message-{remote_hash_prefix}.md",
            revision=revision,
            permissions=permissions,
        )


def _validate_materializer_source(
    identity: VendorKnowledgeSourceIdentity,
    source: KnowledgeSourceRef,
) -> None:
    actual = VendorKnowledgeSourceIdentity(
        provider_id=source.provider_id,
        integration_category=source.integration_kind,
        source_kind=source.source_kind,
    )
    if actual != identity:
        raise ConnectedSourceSyncSinkError("connected_source_materializer_identity_mismatch")


def _build_materialized_document(
    *,
    identity: VendorKnowledgeSourceIdentity,
    source: KnowledgeSourceRef,
    tenant_id: str,
    workspace_id: str,
    binding_id: str,
    source_id: str,
    remote_id: str,
    markdown: str,
    safe_file_name: str,
    revision: KnowledgeItemRevision | None,
    permissions: KnowledgePermissions | None,
) -> MaterializedConnectedSourceDocument:
    return build_materialized_connected_source_document(
        identity=identity,
        source=source,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        binding_id=binding_id,
        source_id=source_id,
        remote_id=remote_id,
        markdown=markdown,
        safe_file_name=safe_file_name,
        revision=revision,
        permissions=permissions,
    )


def _default_plugin_registry() -> VendorKnowledgeSourcePluginRegistry:
    return build_default_vendor_knowledge_source_plugin_registry()


def _safe_conversation_label(record: _SlackConversationMessageKnowledgeRecord) -> str:
    conversation = record.conversation
    for key in ("safe_display_name", "safe_name", "safe_label"):
        value = conversation.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Slack conversation"


def _render_slack_message_markdown(
    *,
    safe_conversation_label: str,
    record: _SlackConversationMessageKnowledgeRecord,
) -> str:
    conversation_id = record.conversation.get("conversation_id")
    message_ts = record.message.get("message_ts")
    created_at = record.timestamps.get("created_at")
    edited_at = record.timestamps.get("edited_at")
    actor_id = record.actor.get("provider_id")
    root_thread_ts = record.thread.get("root_thread_ts")
    reply_count = record.thread.get("reply_count")
    lines = [
        f"# {safe_conversation_label}",
        "",
        f"Conversation: {safe_conversation_label}",
        f"Message: {record.text.strip()}",
    ]
    if isinstance(conversation_id, str) and conversation_id:
        lines.append(f"Conversation ID: {conversation_id}")
    if isinstance(message_ts, str) and message_ts:
        lines.append(f"Message timestamp: {message_ts}")
        if isinstance(conversation_id, str) and conversation_id:
            lines.append(f"Safe locator: slack://{conversation_id}/{message_ts}")
    if isinstance(created_at, str) and created_at:
        lines.append(f"Created at: {created_at}")
    if isinstance(edited_at, str) and edited_at:
        lines.append(f"Edited at: {edited_at}")
    if isinstance(actor_id, str) and actor_id:
        lines.append(f"Actor: {actor_id}")
    if isinstance(root_thread_ts, str) and root_thread_ts:
        lines.append("Thread root: reply")
        lines.append(f"Thread root timestamp: {root_thread_ts}")
    elif isinstance(reply_count, int) and reply_count > 0:
        lines.append("Thread root: root")
        if isinstance(message_ts, str) and message_ts:
            lines.append(f"Thread root timestamp: {message_ts}")
    if record.safe_file_inventory:
        lines.append("")
        lines.append("Files:")
        for item in record.safe_file_inventory:
            safe_name = item.get("safe_file_name")
            mime = item.get("mimetype")
            if isinstance(safe_name, str) and safe_name:
                if isinstance(mime, str) and mime:
                    lines.append(f"- {safe_name} ({mime})")
                else:
                    lines.append(f"- {safe_name}")
    lines.extend(
        [
            "",
            f"Provider: {record.provider}",
            f"Source kind: {record.source_kind}",
            "",
        ]
    )
    return "\n".join(lines)


def default_connected_source_materializer_registry() -> ConnectedSourceContentMaterializerRegistry:
    return ConnectedSourceContentMaterializerRegistry(
        materializers=(
            ConfluencePageRichTextMaterializer(),
            SlackConversationStructuredRecordMaterializer(),
            MsGraphTeamsChatStructuredRecordMaterializer(),
            MsGraphMailStructuredRecordMaterializer(),
            MsGraphTeamsChannelStructuredRecordMaterializer(),
            MsGraphCalendarStructuredRecordMaterializer(),
            GoogleCalendarStructuredRecordMaterializer(),
            GoogleDocsStructuredRecordMaterializer(),
            GoogleSheetsStructuredRecordMaterializer(),
            JiraIssueStructuredRecordMaterializer(),
        )
    )


IndexedSourceMaterializationRegistry = ConnectedSourceContentMaterializerRegistry
