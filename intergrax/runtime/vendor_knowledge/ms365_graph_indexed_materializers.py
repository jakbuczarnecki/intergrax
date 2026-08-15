# © Artur Czarnecki. All rights reserved.

"""Microsoft Graph provider-owned Indexed materialization strategies."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_folders import (
    MSGRAPH_MAIL_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_inventory import (
    MSGRAPH_CALENDAR_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (
    MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.indexed_materialization import (
    MaterializedConnectedSourceDocument,
    VendorKnowledgeMaterializationError,
    build_materialized_connected_source_document,
    validate_materializer_source,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeItemRevision,
    KnowledgePermissions,
    KnowledgeSourceRef,
)
from intergrax.runtime.vendor_knowledge.plugin import VendorKnowledgeSourceIdentity

_MSGRAPH_MAIL_MESSAGE_SCHEMA = "msgraph.mail.message.knowledge.v1"
_REMOTE_HASH_PREFIX_LEN = 16

_MSGRAPH_MAIL_IDENTITY = VendorKnowledgeSourceIdentity(
    provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    integration_category=IntegrationCategory.COLLABORATION_SUITE,
    source_kind=MSGRAPH_MAIL_SOURCE_KIND,
)
_MSGRAPH_TEAMS_CHANNEL_MESSAGE_SCHEMA = "msgraph.teams-channel.message.knowledge.v1"
_MSGRAPH_CALENDAR_EVENT_SCHEMA = "msgraph.calendar.event.knowledge.v1"

_MSGRAPH_TEAMS_CHANNEL_IDENTITY = VendorKnowledgeSourceIdentity(
    provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    integration_category=IntegrationCategory.COLLABORATION_SUITE,
    source_kind=MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
)
_MSGRAPH_CALENDAR_IDENTITY = VendorKnowledgeSourceIdentity(
    provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    integration_category=IntegrationCategory.COLLABORATION_SUITE,
    source_kind=MSGRAPH_CALENDAR_SOURCE_KIND,
)


class _MsGraphMailMessageKnowledgeRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["msgraph.mail.message.knowledge.v1"] = Field(alias="schema")
    subject: str | None = None
    conversation_id: str | None = None
    internet_message_id: str | None = None
    body_text: str
    unique_body_text: str | None = None
    from_participant: dict[str, object] | None = Field(default=None, alias="from")
    sender: dict[str, object] | None = None
    reply_to: list[dict[str, object]] = Field(default_factory=list)
    to_recipients: list[dict[str, object]] = Field(default_factory=list)
    cc_recipients: list[dict[str, object]] = Field(default_factory=list)
    bcc_recipients: list[dict[str, object]] = Field(default_factory=list)
    created_at: str | None = None
    last_modified_at: str | None = None
    received_at: str | None = None
    sent_at: str | None = None
    is_read: bool
    is_draft: bool
    importance: str
    attachments: dict[str, object] = Field(default_factory=dict)


class _MsGraphTeamsChannelMessageKnowledgeRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["msgraph.teams-channel.message.knowledge.v1"] = Field(alias="schema")
    message_kind: Literal["root"]
    state: Literal["active"]
    subject: str | None = None
    body: dict[str, object]
    sender: dict[str, object] | None = None
    created_at: str
    last_modified_at: str
    last_edited_at: str | None = None
    message_type: str
    importance: str
    locale: str | None = None
    event_detail_type: str | None = None
    mentions: list[dict[str, object]] = Field(default_factory=list)
    reactions: list[dict[str, object]] = Field(default_factory=list)
    attachments: dict[str, object] = Field(default_factory=dict)


class _MsGraphCalendarEventKnowledgeRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["msgraph.calendar.event.knowledge.v1"] = Field(alias="schema")
    event_type: str
    subject: str | None = None
    body: dict[str, object]
    start_at: str
    end_at: str
    original_start_at: str | None = None
    original_start_time_zone: str | None = None
    original_end_time_zone: str | None = None
    created_at: str
    last_modified_at: str
    organizer: dict[str, object] | None = None
    attendees: list[dict[str, object]] = Field(default_factory=list)
    location: dict[str, object] | None = None
    locations: list[dict[str, object] | None] = Field(default_factory=list)
    recurrence: dict[str, object] | None = None
    series_master_id: str | None = None
    cancelled_occurrence_ids: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    i_cal_uid: str | None = None
    importance: str
    sensitivity: str
    show_as: str
    response_status: dict[str, object]
    is_all_day: bool
    is_cancelled: bool
    is_draft: bool
    is_organizer: bool
    is_online_meeting: bool
    online_meeting_provider: str
    has_attachments: bool
    hide_attendees: bool
    allow_new_time_proposals: bool
    response_requested: bool
    is_reminder_on: bool
    reminder_minutes_before_start: int | None = None
    attachments: dict[str, object] = Field(default_factory=dict)


class MsGraphMailStructuredRecordMaterializer:
    """Materialize only the accepted Mail message body/metadata projection."""

    identity = _MSGRAPH_MAIL_IDENTITY
    runtime_ref = "indexed-source:ms365_graph:mail"
    schema_name = _MSGRAPH_MAIL_MESSAGE_SCHEMA

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
        validate_materializer_source(self.identity, source)
        if content.mode is not KnowledgeContentMode.STRUCTURED_RECORD:
            raise VendorKnowledgeMaterializationError(
                "connected_source_content_mode_invalid"
            )
        record = content.structured_record
        if not isinstance(record, dict):
            raise VendorKnowledgeMaterializationError(
                "connected_source_structured_record_invalid"
            )
        try:
            validated = _MsGraphMailMessageKnowledgeRecord.model_validate(record)
        except ValueError:
            raise VendorKnowledgeMaterializationError(
                "connected_source_structured_record_invalid"
            ) from None

        body = validated.body_text.strip() or (validated.unique_body_text or "").strip()
        if not body:
            raise VendorKnowledgeMaterializationError(
                "connected_source_structured_record_invalid"
            )
        markdown = _render_msgraph_mail_message_markdown(record=validated, body=body)
        remote_hash_prefix = hashlib.sha256(remote_id.encode("utf-8")).hexdigest()[
            :_REMOTE_HASH_PREFIX_LEN
        ]
        return build_materialized_connected_source_document(
            identity=self.identity,
            source=source,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            remote_id=remote_id,
            markdown=markdown,
            safe_file_name=f"mail-message-{remote_hash_prefix}.md",
            revision=revision,
            permissions=permissions,
        )


class MsGraphTeamsChannelStructuredRecordMaterializer:
    """Materialize one bounded active Teams Channel root post."""

    identity = _MSGRAPH_TEAMS_CHANNEL_IDENTITY
    runtime_ref = "indexed-source:ms365_graph:teams_channel"
    schema_name = _MSGRAPH_TEAMS_CHANNEL_MESSAGE_SCHEMA

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
        validate_materializer_source(self.identity, source)
        validated = _validate_structured_record(
            content,
            _MsGraphTeamsChannelMessageKnowledgeRecord,
        )
        body = validated.body.get("content")
        if not isinstance(body, str) or not body.strip():
            raise VendorKnowledgeMaterializationError(
                "connected_source_structured_record_invalid"
            )
        lines = [
            f"# {(validated.subject or 'Teams channel post').strip() or 'Teams channel post'}",
            "",
            body.strip(),
        ]
        sender = _safe_graph_participant_label(validated.sender)
        if sender:
            lines.extend(["", f"Sender: {sender}"])
        lines.extend(
            [
                "",
                f"Created at: {validated.created_at}",
                f"Last modified at: {validated.last_modified_at}",
                f"Message type: {validated.message_type}",
                f"Importance: {validated.importance}",
            ]
        )
        if validated.locale:
            lines.append(f"Locale: {validated.locale}")
        if validated.event_detail_type:
            lines.append(f"Event detail type: {validated.event_detail_type}")
        lines.extend(
            [
                "",
                "Attachments, hosted content, and reference URLs are not included.",
                "Deletion: provider deletedDateTime is represented by the source tombstone; "
                "deleted posts are not materialized.",
                "",
                f"Provider: {self.identity.provider_id}",
                f"Source kind: {self.identity.source_kind}",
                "",
            ]
        )
        return build_materialized_connected_source_document(
            identity=self.identity,
            source=source,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            remote_id=remote_id,
            markdown="\n".join(lines),
            safe_file_name=f"teams-channel-message-{_remote_hash_prefix(remote_id)}.md",
            revision=revision,
            permissions=permissions,
        )


class MsGraphCalendarStructuredRecordMaterializer:
    """Materialize the accepted bounded Calendar event projection."""

    identity = _MSGRAPH_CALENDAR_IDENTITY
    runtime_ref = "indexed-source:ms365_graph:calendar"
    schema_name = _MSGRAPH_CALENDAR_EVENT_SCHEMA

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
        validate_materializer_source(self.identity, source)
        validated = _validate_structured_record(
            content,
            _MsGraphCalendarEventKnowledgeRecord,
        )
        body = validated.body.get("content")
        if not isinstance(body, str) or not body.strip():
            raise VendorKnowledgeMaterializationError(
                "connected_source_structured_record_invalid"
            )
        title = (validated.subject or "Calendar event").strip() or "Calendar event"
        lines = [
            f"# {title}",
            "",
            body.strip(),
            "",
            f"Event type: {validated.event_type}",
            f"Starts at: {validated.start_at}",
            f"Ends at: {validated.end_at}",
            f"Created at: {validated.created_at}",
            f"Last modified at: {validated.last_modified_at}",
        ]
        organizer = _safe_graph_participant_label(validated.organizer)
        if organizer:
            lines.append(f"Organizer: {organizer}")
        location = _safe_graph_location_label(validated.location)
        if location:
            lines.append(f"Location: {location}")
        lines.extend(
            [
                f"All day: {'yes' if validated.is_all_day else 'no'}",
                f"Cancelled: {'yes' if validated.is_cancelled else 'no'}",
                f"Online meeting: {'yes' if validated.is_online_meeting else 'no'}",
                f"Attachments: {'inventory metadata present' if validated.has_attachments else 'none'}; "
                "binary content is not included.",
                "",
                "Removal semantics remain source-owned: primary delta tombstones and "
                "non-primary calendar window reconciliation are not collapsed.",
                "",
                f"Provider: {self.identity.provider_id}",
                f"Source kind: {self.identity.source_kind}",
                "",
            ]
        )
        return build_materialized_connected_source_document(
            identity=self.identity,
            source=source,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            remote_id=remote_id,
            markdown="\n".join(lines),
            safe_file_name=f"calendar-event-{_remote_hash_prefix(remote_id)}.md",
            revision=revision,
            permissions=permissions,
        )


def _validate_structured_record(
    content: KnowledgeContent,
    model: type[BaseModel],
) -> BaseModel:
    if content.mode is not KnowledgeContentMode.STRUCTURED_RECORD:
        raise VendorKnowledgeMaterializationError("connected_source_content_mode_invalid")
    record = content.structured_record
    if not isinstance(record, dict):
        raise VendorKnowledgeMaterializationError(
            "connected_source_structured_record_invalid"
        )
    try:
        return model.model_validate(record)
    except ValueError:
        raise VendorKnowledgeMaterializationError(
            "connected_source_structured_record_invalid"
        ) from None


def _remote_hash_prefix(remote_id: str) -> str:
    return hashlib.sha256(remote_id.encode("utf-8")).hexdigest()[:_REMOTE_HASH_PREFIX_LEN]


def _safe_graph_participant_label(participant: dict[str, object] | None) -> str:
    if participant is None:
        return ""
    display_name = participant.get("display_name")
    address = participant.get("address")
    safe_display = display_name.strip() if isinstance(display_name, str) else ""
    safe_address = address.strip() if isinstance(address, str) else ""
    if safe_display and safe_address:
        return f"{safe_display} <{safe_address}>"
    return safe_display or safe_address


def _safe_graph_location_label(location: dict[str, object] | None) -> str:
    if location is None:
        return ""
    display_name = location.get("display_name")
    return display_name.strip() if isinstance(display_name, str) else ""


def _render_msgraph_mail_message_markdown(
    *,
    record: _MsGraphMailMessageKnowledgeRecord,
    body: str,
) -> str:
    subject = (record.subject or "").strip() or "Mail message"
    lines = [f"# {subject}", "", body]
    sender = _safe_mail_participant_label(record.sender or record.from_participant)
    if sender:
        lines.extend(["", f"From: {sender}"])
    for label, participants in (
        ("To", record.to_recipients),
        ("Cc", record.cc_recipients),
        ("Reply-To", record.reply_to),
    ):
        rendered = _safe_mail_participant_list(participants)
        if rendered:
            lines.append(f"{label}: {rendered}")
    if record.conversation_id:
        lines.extend(
            [
                "",
                f"Conversation ID: {record.conversation_id.strip()}",
                "Thread: conversation membership metadata only; thread messages are not included.",
            ]
        )
    if record.received_at:
        lines.append(f"Received at: {record.received_at}")
    if record.sent_at:
        lines.append(f"Sent at: {record.sent_at}")
    if record.last_modified_at:
        lines.append(f"Last modified at: {record.last_modified_at}")
    if record.attachments.get("has_attachments") is True:
        lines.extend(
            [
                "",
                "Attachments: presence metadata only; attachment inventory and binary content are not included.",
            ]
        )
    lines.extend(
        [
            "",
            f"Read: {'yes' if record.is_read else 'no'}",
            f"Draft: {'yes' if record.is_draft else 'no'}",
            f"Importance: {record.importance}",
            "",
            "Provider: microsoft_graph",
            "Source kind: msgraph_mail_folder",
            "",
        ]
    )
    return "\n".join(lines)


def _safe_mail_participant_label(participant: dict[str, object] | None) -> str:
    if participant is None:
        return ""
    display_name = participant.get("display_name")
    address = participant.get("address")
    safe_display = display_name.strip() if isinstance(display_name, str) else ""
    safe_address = address.strip() if isinstance(address, str) else ""
    if safe_display and safe_address:
        return f"{safe_display} <{safe_address}>"
    return safe_display or safe_address


def _safe_mail_participant_list(participants: list[dict[str, object]]) -> str:
    return ", ".join(
        label
        for participant in participants
        if (label := _safe_mail_participant_label(participant))
    )
