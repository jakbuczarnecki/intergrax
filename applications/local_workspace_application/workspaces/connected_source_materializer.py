# © Artur Czarnecki. All rights reserved.

"""Materialize vendor knowledge structured records into LKW indexable documents."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.vendor_knowledge.models import KnowledgeContent, KnowledgeContentMode
from local_workspace_application.workspaces.connected_source_ids import connected_logical_path
from local_workspace_application.workspaces.connected_source_models import ConnectedSourceSyncSinkError

_SLACK_CONVERSATION_MESSAGE_SCHEMA = "slack.conversation.message.knowledge.v1"
_REMOTE_HASH_PREFIX_LEN = 16


class _SlackConversationMessageKnowledgeRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    schema: Literal["slack.conversation.message.knowledge.v1"]
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


@dataclass(frozen=True, slots=True)
class MaterializedConnectedSourceDocument:
    logical_source_path: str
    safe_file_name: str
    markdown: str
    content_hash: str


@runtime_checkable
class ConnectedSourceContentMaterializer(Protocol):
    schema_name: str

    def materialize(
        self,
        *,
        source_id: str,
        remote_id: str,
        content: KnowledgeContent,
    ) -> MaterializedConnectedSourceDocument:
        ...


class ConnectedSourceContentMaterializerRegistry:
    def __init__(self, materializers: tuple[ConnectedSourceContentMaterializer, ...]) -> None:
        self._by_schema: dict[str, ConnectedSourceContentMaterializer] = {}
        for item in materializers:
            if item.schema_name in self._by_schema:
                raise ConnectedSourceSyncSinkError("connected_source_materializer_duplicate")
            self._by_schema[item.schema_name] = item

    def resolve(self, schema_name: str) -> ConnectedSourceContentMaterializer:
        materializer = self._by_schema.get(schema_name)
        if materializer is None:
            raise ConnectedSourceSyncSinkError("connected_source_materializer_unsupported")
        return materializer


class SlackConversationStructuredRecordMaterializer:
    schema_name = _SLACK_CONVERSATION_MESSAGE_SCHEMA

    def materialize(
        self,
        *,
        source_id: str,
        remote_id: str,
        content: KnowledgeContent,
    ) -> MaterializedConnectedSourceDocument:
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
        markdown_bytes = markdown.encode("utf-8")
        content_hash = hashlib.sha256(markdown_bytes).hexdigest()
        logical_source_path = connected_logical_path(source_id=source_id, remote_id=remote_id)
        remote_hash_prefix = hashlib.sha256(remote_id.encode("utf-8")).hexdigest()[
            :_REMOTE_HASH_PREFIX_LEN
        ]
        safe_file_name = f"slack-message-{remote_hash_prefix}.md"
        return MaterializedConnectedSourceDocument(
            logical_source_path=logical_source_path,
            safe_file_name=safe_file_name,
            markdown=markdown,
            content_hash=content_hash,
        )


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
    if isinstance(created_at, str) and created_at:
        lines.append(f"Created at: {created_at}")
    if isinstance(edited_at, str) and edited_at:
        lines.append(f"Edited at: {edited_at}")
    if isinstance(actor_id, str) and actor_id:
        lines.append(f"Actor: {actor_id}")
    if isinstance(root_thread_ts, str) and root_thread_ts:
        lines.append("Thread root: reply")
    elif isinstance(reply_count, int) and reply_count > 0:
        lines.append("Thread root: root")
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
        materializers=(SlackConversationStructuredRecordMaterializer(),)
    )
