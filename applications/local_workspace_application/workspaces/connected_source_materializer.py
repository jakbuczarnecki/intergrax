# © Artur Czarnecki. All rights reserved.

"""Materialize vendor knowledge structured records into LKW indexable documents."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.runtime.vendor_knowledge.models import KnowledgeContent, KnowledgeContentMode
from local_workspace_application.workspaces.connected_source_ids import connected_logical_path
from local_workspace_application.workspaces.connected_source_models import ConnectedSourceSyncSinkError

_SLACK_CONVERSATION_MESSAGE_SCHEMA = "slack.conversation.message.knowledge.v1"


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
        self._by_schema = {item.schema_name: item for item in materializers}

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
        schema = record.get("schema")
        if schema != self.schema_name:
            raise ConnectedSourceSyncSinkError("connected_source_schema_unsupported")

        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            text = _title_from_record(record)

        message = record.get("message")
        message_ts = ""
        if isinstance(message, dict):
            raw_ts = message.get("message_ts")
            if isinstance(raw_ts, str):
                message_ts = raw_ts.strip()

        conversation = record.get("conversation")
        conversation_id = ""
        if isinstance(conversation, dict):
            raw_id = conversation.get("conversation_id")
            if isinstance(raw_id, str):
                conversation_id = raw_id.strip()

        title = _title_from_record(record)
        markdown = _render_slack_message_markdown(
            title=title,
            text=text,
            conversation_id=conversation_id,
            message_ts=message_ts,
        )
        canonical = json.dumps(
            {"schema": schema, "remote_id": remote_id, "markdown": markdown},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        content_hash = hashlib.sha256(canonical).hexdigest()
        logical_source_path = connected_logical_path(source_id=source_id, remote_id=remote_id)
        safe_file_name = f"{title[:120] or 'slack-message'}.md"
        return MaterializedConnectedSourceDocument(
            logical_source_path=logical_source_path,
            safe_file_name=safe_file_name,
            markdown=markdown,
            content_hash=content_hash,
        )


def _title_from_record(record: dict[str, object]) -> str:
    text = record.get("text")
    if isinstance(text, str) and text.strip():
        collapsed = " ".join(text.strip().split())
        return collapsed[:120]
    return "Slack message"


def _render_slack_message_markdown(
    *,
    title: str,
    text: str,
    conversation_id: str,
    message_ts: str,
) -> str:
    lines = [f"# {title}", ""]
    if conversation_id:
        lines.append(f"Conversation: `{conversation_id}`")
    if message_ts:
        lines.append(f"Message timestamp: `{message_ts}`")
    if conversation_id or message_ts:
        lines.append("")
    lines.append(text.strip())
    lines.append("")
    return "\n".join(lines)


def default_connected_source_materializer_registry() -> ConnectedSourceContentMaterializerRegistry:
    return ConnectedSourceContentMaterializerRegistry(
        materializers=(SlackConversationStructuredRecordMaterializer(),)
    )
