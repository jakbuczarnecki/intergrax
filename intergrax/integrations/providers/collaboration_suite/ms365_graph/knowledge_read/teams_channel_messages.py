# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Teams Channel knowledge-read: root and reply message pages."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable
from urllib.parse import quote, unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.common import (
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeTransport,
    parse_msgraph_collection_page,
    validate_msgraph_continuation_url,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (
    MsGraphTeamsChannel,
    validate_msgraph_teams_channel,
    validate_msgraph_teams_channel_id,
    validate_msgraph_teams_channel_message_id,
    validate_msgraph_teams_team_id,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_messages import (
    MsGraphTeamsChatAttachmentKind,
    MsGraphTeamsChatAttachmentReference,
    MsGraphTeamsChatMention,
    MsGraphTeamsChatReaction,
    MsGraphTeamsForwardedMessageReference,
    MsGraphTeamsIdentity,
    MsGraphTeamsIdentityKind,
    validate_msgraph_teams_chat_attachment_reference,
    validate_msgraph_teams_chat_mention,
    validate_msgraph_teams_chat_reaction,
    validate_msgraph_teams_identity,
)

DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS = 2_000_000
ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS = 8_000_000

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_MESSAGES_RESPONSE = "unexpected Microsoft Graph Teams channel messages response"
_INVALID_MESSAGES_REQUEST = "invalid Microsoft Graph Teams channel messages request"
_INVALID_MESSAGES_CONTINUATION = "invalid Microsoft Graph Teams channel messages continuation"
_MAX_REVISION_LEN = 4096
_MAX_SUBJECT_LEN = 4096
_MAX_MENTION_TEXT_LEN = 4096
_MAX_REACTION_TYPE_LEN = 256
_MAX_DISPLAY_NAME_LEN = 1024
_MAX_IDENTITY_TYPE_LEN = 1024
_MAX_TENANT_ID_LEN = 2048
_MAX_CONTENT_TYPE_LEN = 2048
_MAX_ATTACHMENT_NAME_LEN = 4096
_MAX_EMBEDDED_CONTENT_LEN = 1_000_000
_MAX_CONTENT_URL_LEN = 8192
_MAX_EVENT_DETAIL_TYPE_LEN = 1024
_MAX_LOCALE_LEN = 128
_MAX_ENUM_STRING_LEN = 1024
_MIN_MESSAGE_LIMIT = 1
_MAX_MESSAGE_LIMIT = 50
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_PREFER_UNKNOWN_ENUM = {"Prefer": "include-unknown-enum-members"}


class MsGraphTeamsChannelMessageKind(StrEnum):
    ROOT = "root"
    REPLY = "reply"


class MsGraphTeamsChannelMessageState(StrEnum):
    ACTIVE = "active"
    DELETED = "deleted"


class MsGraphTeamsChannelMessageType(StrEnum):
    MESSAGE = "message"
    CHAT_EVENT = "chat_event"
    TYPING = "typing"
    SYSTEM_EVENT_MESSAGE = "system_event_message"
    UNKNOWN = "unknown"


class MsGraphTeamsChannelBodyKind(StrEnum):
    TEXT = "text"
    HTML = "html"


class MsGraphTeamsChannelImportance(StrEnum):
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    UNKNOWN = "unknown"


_MESSAGE_TYPE_MAP: dict[str, MsGraphTeamsChannelMessageType] = {
    "message": MsGraphTeamsChannelMessageType.MESSAGE,
    "chatevent": MsGraphTeamsChannelMessageType.CHAT_EVENT,
    "typing": MsGraphTeamsChannelMessageType.TYPING,
    "systemeventmessage": MsGraphTeamsChannelMessageType.SYSTEM_EVENT_MESSAGE,
}

_IMPORTANCE_MAP: dict[str, MsGraphTeamsChannelImportance] = {
    "normal": MsGraphTeamsChannelImportance.NORMAL,
    "high": MsGraphTeamsChannelImportance.HIGH,
    "urgent": MsGraphTeamsChannelImportance.URGENT,
}

_BODY_KIND_MAP: dict[str, MsGraphTeamsChannelBodyKind] = {
    "text": MsGraphTeamsChannelBodyKind.TEXT,
    "html": MsGraphTeamsChannelBodyKind.HTML,
}


def _validate_enum_string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if len(trimmed) > _MAX_ENUM_STRING_LEN:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    return trimmed


def _map_message_type(value: object) -> MsGraphTeamsChannelMessageType:
    trimmed = _validate_enum_string(value)
    normalized = trimmed.lower().replace("_", "")
    return _MESSAGE_TYPE_MAP.get(normalized, MsGraphTeamsChannelMessageType.UNKNOWN)


def _map_importance(value: object) -> MsGraphTeamsChannelImportance:
    trimmed = _validate_enum_string(value)
    normalized = trimmed.lower()
    return _IMPORTANCE_MAP.get(normalized, MsGraphTeamsChannelImportance.UNKNOWN)


def _parse_timezone_aware_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if trimmed.endswith("Z"):
        trimmed = f"{trimmed[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(trimmed)
    except ValueError:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    return parsed.astimezone(timezone.utc)


def _normalize_model_datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    return value.astimezone(timezone.utc)


def _validate_revision(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if len(trimmed) > _MAX_REVISION_LEN:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    return trimmed


def _validate_optional_trimmed_string(value: object, *, max_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        return None
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if len(trimmed) > max_length:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    return trimmed


def _validate_bounded_string(value: object, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if "\x00" in value:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if _ASCII_CONTROL.search(value):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if len(value) > max_length:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    return value


def _validate_https_reference_url(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if "\x00" in value:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if len(value) > _MAX_CONTENT_URL_LEN:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    parsed = urlparse(value)
    if parsed.scheme != "https":
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if not parsed.hostname:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if parsed.username or parsed.password:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    return value


def _validate_message_max_chars(value: object) -> int:
    if type(value) is not int:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if value < 1 or value > ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    return value


def _parse_channel_identity(
    payload: object,
    *,
    expected_team_id: str,
    expected_channel_id: str,
) -> tuple[str, str]:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if "teamId" not in payload or "channelId" not in payload:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    team_id = validate_msgraph_teams_team_id(payload.get("teamId"))
    channel_id = validate_msgraph_teams_channel_id(payload.get("channelId"))
    if team_id != expected_team_id or channel_id != expected_channel_id:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    return team_id, channel_id


def _parse_identity_user(payload: object) -> MsGraphTeamsIdentity | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    remote_id: str | None = None
    if "id" in payload and payload.get("id") is not None:
        remote_id = validate_msgraph_teams_channel_message_id(payload.get("id"))
    display_name: str | None = None
    if "displayName" in payload and payload.get("displayName") is not None:
        display_name = _validate_optional_trimmed_string(
            payload.get("displayName"),
            max_length=_MAX_DISPLAY_NAME_LEN,
        )
        if display_name is None:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    tenant_id: str | None = None
    if "tenantId" in payload and payload.get("tenantId") is not None:
        tenant_id = _validate_optional_trimmed_string(
            payload.get("tenantId"),
            max_length=_MAX_TENANT_ID_LEN,
        )
        if tenant_id is None:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    identity_type: str | None = None
    if "userIdentityType" in payload and payload.get("userIdentityType") is not None:
        identity_type = _validate_optional_trimmed_string(
            payload.get("userIdentityType"),
            max_length=_MAX_IDENTITY_TYPE_LEN,
        )
        if identity_type is None:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if "@odata.type" in payload and payload.get("@odata.type") is not None:
        validated_odata_type = _validate_optional_trimmed_string(
            payload.get("@odata.type"),
            max_length=_MAX_IDENTITY_TYPE_LEN,
        )
        if validated_odata_type is None:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if remote_id is None and display_name is None:
        return None
    return MsGraphTeamsIdentity(
        identity_kind=MsGraphTeamsIdentityKind.USER,
        remote_id=remote_id,
        display_name=display_name,
        tenant_id=tenant_id,
        identity_type=identity_type,
    )


def _parse_identity_application(payload: object) -> MsGraphTeamsIdentity | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    remote_id: str | None = None
    if "id" in payload and payload.get("id") is not None:
        remote_id = validate_msgraph_teams_channel_message_id(payload.get("id"))
    display_name: str | None = None
    if "displayName" in payload and payload.get("displayName") is not None:
        display_name = _validate_optional_trimmed_string(
            payload.get("displayName"),
            max_length=_MAX_DISPLAY_NAME_LEN,
        )
        if display_name is None:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    identity_type: str | None = None
    if "applicationIdentityType" in payload and payload.get("applicationIdentityType") is not None:
        identity_type = _validate_optional_trimmed_string(
            payload.get("applicationIdentityType"),
            max_length=_MAX_IDENTITY_TYPE_LEN,
        )
        if identity_type is None:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if "@odata.type" in payload and payload.get("@odata.type") is not None:
        validated_odata_type = _validate_optional_trimmed_string(
            payload.get("@odata.type"),
            max_length=_MAX_IDENTITY_TYPE_LEN,
        )
        if validated_odata_type is None:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if remote_id is None and display_name is None:
        return None
    return MsGraphTeamsIdentity(
        identity_kind=MsGraphTeamsIdentityKind.APPLICATION,
        remote_id=remote_id,
        display_name=display_name,
        identity_type=identity_type,
    )


def _parse_identity_conversation(payload: object) -> MsGraphTeamsIdentity | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    remote_id: str | None = None
    if "id" in payload and payload.get("id") is not None:
        remote_id = validate_msgraph_teams_channel_message_id(payload.get("id"))
    display_name: str | None = None
    if "displayName" in payload and payload.get("displayName") is not None:
        display_name = _validate_optional_trimmed_string(
            payload.get("displayName"),
            max_length=_MAX_DISPLAY_NAME_LEN,
        )
        if display_name is None:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    identity_type: str | None = None
    if "conversationIdentityType" in payload and payload.get("conversationIdentityType") is not None:
        identity_type = _validate_optional_trimmed_string(
            payload.get("conversationIdentityType"),
            max_length=_MAX_IDENTITY_TYPE_LEN,
        )
        if identity_type is None:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if remote_id is None and display_name is None:
        return None
    return MsGraphTeamsIdentity(
        identity_kind=MsGraphTeamsIdentityKind.CONVERSATION,
        remote_id=remote_id,
        display_name=display_name,
        identity_type=identity_type,
    )


def _parse_from_identity_set(payload: object) -> MsGraphTeamsIdentity | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if payload.get("device") is not None:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    user = _parse_identity_user(payload.get("user"))
    application = _parse_identity_application(payload.get("application"))
    if user is not None and application is not None:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if user is not None:
        return user
    if application is not None:
        return application
    return None


def _parse_mentioned_identity_set(payload: object) -> MsGraphTeamsIdentity:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    user = _parse_identity_user(payload.get("user"))
    application = _parse_identity_application(payload.get("application"))
    conversation = _parse_identity_conversation(payload.get("conversation"))
    present = [item for item in (user, application, conversation) if item is not None]
    if len(present) != 1:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    return present[0]


def _parse_reaction_identity_set(payload: object) -> MsGraphTeamsIdentity:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    user = _parse_identity_user(payload.get("user"))
    if user is None:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    return user


def _map_attachment_kind(content_type: str) -> MsGraphTeamsChatAttachmentKind:
    if content_type == "reference":
        return MsGraphTeamsChatAttachmentKind.REFERENCE
    if content_type == "forwardedMessageReference":
        return MsGraphTeamsChatAttachmentKind.FORWARDED_MESSAGE_REFERENCE
    if content_type == "application/vnd.microsoft.card.codesnippet":
        return MsGraphTeamsChatAttachmentKind.CODE_SNIPPET
    if content_type == "application/vnd.microsoft.card.announcement":
        return MsGraphTeamsChatAttachmentKind.ANNOUNCEMENT
    if content_type.startswith("application/vnd.microsoft.card."):
        return MsGraphTeamsChatAttachmentKind.CARD
    return MsGraphTeamsChatAttachmentKind.UNKNOWN


def _parse_forwarded_message_reference(content: str) -> MsGraphTeamsForwardedMessageReference:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
    if not isinstance(parsed, dict):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
    if "originalMessageId" not in parsed or "originalConversationId" not in parsed:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
    if "originalSentDateTime" not in parsed:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
    try:
        original_message_id = validate_msgraph_teams_channel_message_id(
            parsed.get("originalMessageId")
        )
        original_chat_id = validate_msgraph_teams_channel_id(parsed.get("originalConversationId"))
        original_sent_at = _parse_timezone_aware_datetime(parsed.get("originalSentDateTime"))
    except ValueError:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
    original_sender: MsGraphTeamsIdentity | None = None
    if "originalMessageSender" in parsed and parsed.get("originalMessageSender") is not None:
        try:
            original_sender = _parse_from_identity_set(parsed.get("originalMessageSender"))
        except ValueError:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
    return MsGraphTeamsForwardedMessageReference(
        original_message_id=original_message_id,
        original_chat_id=original_chat_id,
        original_sent_at=original_sent_at,
        original_sender=original_sender,
    )


def _parse_attachment_reference(payload: dict[str, Any]) -> MsGraphTeamsChatAttachmentReference:
    if "id" not in payload or "contentType" not in payload:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    remote_id = validate_msgraph_teams_channel_message_id(payload.get("id"))
    content_type = payload.get("contentType")
    if not isinstance(content_type, str):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    trimmed_type = content_type.strip()
    if not trimmed_type or _ASCII_CONTROL.search(trimmed_type):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    if len(trimmed_type) > _MAX_CONTENT_TYPE_LEN:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
    attachment_kind = _map_attachment_kind(trimmed_type)

    name = None
    if "name" in payload:
        name = _validate_optional_trimmed_string(payload.get("name"), max_length=_MAX_ATTACHMENT_NAME_LEN)

    teams_app_id = None
    if "teamsAppId" in payload:
        teams_app_id = _validate_optional_trimmed_string(
            payload.get("teamsAppId"),
            max_length=_MAX_ATTACHMENT_NAME_LEN,
        )

    has_content = "content" in payload and payload.get("content") is not None
    has_content_url = "contentUrl" in payload and payload.get("contentUrl") is not None
    if has_content and has_content_url:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE)

    has_thumbnail_url = "thumbnailUrl" in payload and payload.get("thumbnailUrl") is not None

    embedded_content: str | None = None
    content_url: str | None = None
    forwarded_message: MsGraphTeamsForwardedMessageReference | None = None

    if attachment_kind is MsGraphTeamsChatAttachmentKind.REFERENCE:
        if not has_content_url:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        content_url = _validate_https_reference_url(payload.get("contentUrl"))
    elif attachment_kind is MsGraphTeamsChatAttachmentKind.FORWARDED_MESSAGE_REFERENCE:
        if not has_content:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        raw_content = payload.get("content")
        if not isinstance(raw_content, str):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        if len(raw_content) > _MAX_EMBEDDED_CONTENT_LEN:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        forwarded_message = _parse_forwarded_message_reference(raw_content)
    elif has_content:
        raw_content = payload.get("content")
        if not isinstance(raw_content, str):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        if "\x00" in raw_content:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        if len(raw_content) > _MAX_EMBEDDED_CONTENT_LEN:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        embedded_content = raw_content

    return MsGraphTeamsChatAttachmentReference(
        remote_id=remote_id,
        attachment_kind=attachment_kind,
        content_type=trimmed_type,
        name=name,
        teams_app_id=teams_app_id,
        embedded_content=embedded_content,
        content_url=content_url,
        has_thumbnail_url=has_thumbnail_url,
        forwarded_message=forwarded_message,
    )


class MsGraphTeamsChannelMessage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    team_remote_id: str = Field(repr=False)
    channel_remote_id: str = Field(repr=False)
    thread_root_remote_id: str = Field(repr=False)
    message_kind: MsGraphTeamsChannelMessageKind
    remote_id: str = Field(repr=False)

    revision: str = Field(repr=False)

    state: MsGraphTeamsChannelMessageState
    message_type: MsGraphTeamsChannelMessageType
    importance: MsGraphTeamsChannelImportance

    created_at: datetime
    last_modified_at: datetime
    last_edited_at: datetime | None = None
    deleted_at: datetime | None = None

    subject: str | None = Field(default=None, repr=False)

    body_kind: MsGraphTeamsChannelBodyKind | None = None
    body_content: str | None = Field(default=None, repr=False)

    sender: MsGraphTeamsIdentity | None = Field(default=None, repr=False)

    attachments: tuple[MsGraphTeamsChatAttachmentReference, ...] = Field(default=(), repr=False)
    mentions: tuple[MsGraphTeamsChatMention, ...] = Field(default=(), repr=False)
    reactions: tuple[MsGraphTeamsChatReaction, ...] = Field(default=(), repr=False)

    event_detail_type: str | None = None
    locale: str | None = None

    @field_validator("team_remote_id", mode="before")
    @classmethod
    def _validate_team_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_team_id(value)

    @field_validator("channel_remote_id", mode="before")
    @classmethod
    def _validate_channel_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_id(value)

    @field_validator("thread_root_remote_id", "remote_id", mode="before")
    @classmethod
    def _validate_message_ids(cls, value: object) -> str:
        return validate_msgraph_teams_channel_message_id(value)

    @field_validator("revision", mode="before")
    @classmethod
    def _validate_revision(cls, value: object) -> str:
        return _validate_revision(value)

    @field_validator("created_at", "last_modified_at", mode="before")
    @classmethod
    def _validate_required_datetimes(cls, value: object) -> datetime:
        return _normalize_model_datetime(value)

    @field_validator("last_edited_at", "deleted_at", mode="before")
    @classmethod
    def _validate_optional_datetimes(cls, value: object) -> datetime | None:
        if value is None:
            return None
        return _normalize_model_datetime(value)

    @field_validator("subject", mode="before")
    @classmethod
    def _validate_subject(cls, value: object) -> str | None:
        return _validate_optional_trimmed_string(value, max_length=_MAX_SUBJECT_LEN)

    @field_validator("body_content", mode="before")
    @classmethod
    def _validate_body_content(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        if "\x00" in value:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        return value

    @field_validator("locale", mode="before")
    @classmethod
    def _validate_locale(cls, value: object) -> str | None:
        return _validate_optional_trimmed_string(value, max_length=_MAX_LOCALE_LEN)

    @field_validator("event_detail_type", mode="before")
    @classmethod
    def _validate_event_detail_type(cls, value: object) -> str | None:
        return _validate_optional_trimmed_string(value, max_length=_MAX_EVENT_DETAIL_TYPE_LEN)

    @field_validator("attachments", "mentions", "reactions", mode="before")
    @classmethod
    def _validate_collections(cls, value: object) -> tuple[Any, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_message_shape(self) -> Self:
        if self.last_modified_at < self.created_at:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        if self.last_edited_at is not None and self.last_edited_at < self.created_at:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        if self.deleted_at is not None and self.deleted_at < self.created_at:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        if self.message_kind is MsGraphTeamsChannelMessageKind.ROOT:
            if self.thread_root_remote_id != self.remote_id:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        elif self.thread_root_remote_id == self.remote_id:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        if self.state is MsGraphTeamsChannelMessageState.ACTIVE:
            if self.deleted_at is not None:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
            if self.body_kind is None or self.body_content is None:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        elif self.state is MsGraphTeamsChannelMessageState.DELETED:
            if self.deleted_at is None:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        return self


class MsGraphTeamsChannelRootMessagePage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    team_remote_id: str
    channel_remote_id: str

    items: tuple[MsGraphTeamsChannelMessage, ...]

    continuation: MsGraphKnowledgeContinuation | None = Field(default=None, repr=False)

    @field_validator("team_remote_id", mode="before")
    @classmethod
    def _validate_team_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_team_id(value)

    @field_validator("channel_remote_id", mode="before")
    @classmethod
    def _validate_channel_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_id(value)

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, value: object) -> tuple[MsGraphTeamsChannelMessage, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphTeamsChannelMessage):
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        return value

    @field_validator("continuation", mode="before")
    @classmethod
    def _validate_continuation(cls, value: object) -> MsGraphKnowledgeContinuation | None:
        if value is None:
            return None
        if not isinstance(value, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        if value.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_page_shape(self) -> MsGraphTeamsChannelRootMessagePage:
        remote_ids = [item.remote_id for item in self.items]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        for item in self.items:
            if item.team_remote_id != self.team_remote_id:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
            if item.channel_remote_id != self.channel_remote_id:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
            if item.message_kind is not MsGraphTeamsChannelMessageKind.ROOT:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
            if item.thread_root_remote_id != item.remote_id:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        return self

    @property
    def has_more(self) -> bool:
        return self.continuation is not None

    @property
    def is_complete(self) -> bool:
        return self.continuation is None


class MsGraphTeamsChannelReplyPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    team_remote_id: str
    channel_remote_id: str
    root_message_remote_id: str
    root_message_revision: str = Field(repr=False)

    items: tuple[MsGraphTeamsChannelMessage, ...]

    continuation: MsGraphKnowledgeContinuation | None = Field(default=None, repr=False)

    @field_validator("team_remote_id", mode="before")
    @classmethod
    def _validate_team_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_team_id(value)

    @field_validator("channel_remote_id", mode="before")
    @classmethod
    def _validate_channel_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_id(value)

    @field_validator("root_message_remote_id", mode="before")
    @classmethod
    def _validate_root_message_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_message_id(value)

    @field_validator("root_message_revision", mode="before")
    @classmethod
    def _validate_root_message_revision(cls, value: object) -> str:
        return _validate_revision(value)

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, value: object) -> tuple[MsGraphTeamsChannelMessage, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphTeamsChannelMessage):
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        return value

    @field_validator("continuation", mode="before")
    @classmethod
    def _validate_continuation(cls, value: object) -> MsGraphKnowledgeContinuation | None:
        if value is None:
            return None
        if not isinstance(value, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        if value.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_page_shape(self) -> MsGraphTeamsChannelReplyPage:
        remote_ids = [item.remote_id for item in self.items]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        for item in self.items:
            if item.team_remote_id != self.team_remote_id:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
            if item.channel_remote_id != self.channel_remote_id:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
            if item.thread_root_remote_id != self.root_message_remote_id:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
            if item.message_kind is not MsGraphTeamsChannelMessageKind.REPLY:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        return self

    @property
    def has_more(self) -> bool:
        return self.continuation is not None

    @property
    def is_complete(self) -> bool:
        return self.continuation is None


def _safe_construct_message(**kwargs: object) -> MsGraphTeamsChannelMessage:
    try:
        return MsGraphTeamsChannelMessage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None


def _safe_construct_root_page(**kwargs: object) -> MsGraphTeamsChannelRootMessagePage:
    try:
        return MsGraphTeamsChannelRootMessagePage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None


def _safe_construct_reply_page(**kwargs: object) -> MsGraphTeamsChannelReplyPage:
    try:
        return MsGraphTeamsChannelReplyPage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None


class MsGraphTeamsChannelMessageChanged(IntegrationDependencyError):
    def __init__(self) -> None:
        super().__init__("Microsoft Graph Teams channel message changed during read")


def parse_msgraph_teams_channel_message(
    payload: object,
    *,
    expected_team_id: str,
    expected_channel_id: str,
    message_kind: MsGraphTeamsChannelMessageKind,
    expected_thread_root_remote_id: str | None = None,
    max_chars: int,
) -> MsGraphTeamsChannelMessage:
    if type(max_chars) is not int or max_chars < 1 or max_chars > ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    required_keys = (
        "id",
        "etag",
        "messageType",
        "createdDateTime",
        "lastModifiedDateTime",
        "deletedDateTime",
        "importance",
        "channelIdentity",
    )
    for key in required_keys:
        if key not in payload:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    if payload.get("chatId") is not None:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    try:
        validated_team_id = validate_msgraph_teams_team_id(expected_team_id)
        validated_channel_id = validate_msgraph_teams_channel_id(expected_channel_id)
        remote_id = validate_msgraph_teams_channel_message_id(payload.get("id"))
        _parse_channel_identity(
            payload.get("channelIdentity"),
            expected_team_id=validated_team_id,
            expected_channel_id=validated_channel_id,
        )
        revision = _validate_revision(payload.get("etag"))
        message_type = _map_message_type(payload.get("messageType"))
        importance = _map_importance(payload.get("importance"))
        created_at = _parse_timezone_aware_datetime(payload.get("createdDateTime"))
        last_modified_at = _parse_timezone_aware_datetime(payload.get("lastModifiedDateTime"))
    except ValueError:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    reply_raw = payload.get("replyToId")
    if message_kind is MsGraphTeamsChannelMessageKind.ROOT:
        if reply_raw is not None:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        thread_root_remote_id = remote_id
    else:
        if reply_raw is None:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        if expected_thread_root_remote_id is None:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        try:
            validated_thread_root = validate_msgraph_teams_channel_message_id(
                expected_thread_root_remote_id
            )
            reply_to_id = validate_msgraph_teams_channel_message_id(reply_raw)
        except ValueError:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        if reply_to_id != validated_thread_root:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        thread_root_remote_id = validated_thread_root

    deleted_raw = payload.get("deletedDateTime")
    if deleted_raw is None:
        state = MsGraphTeamsChannelMessageState.ACTIVE
        deleted_at = None
    else:
        state = MsGraphTeamsChannelMessageState.DELETED
        try:
            deleted_at = _parse_timezone_aware_datetime(deleted_raw)
        except ValueError:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    last_edited_at: datetime | None = None
    if "lastEditedDateTime" in payload and payload.get("lastEditedDateTime") is not None:
        try:
            last_edited_at = _parse_timezone_aware_datetime(payload.get("lastEditedDateTime"))
        except ValueError:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    subject: str | None = None
    if "subject" in payload:
        try:
            subject = _validate_optional_trimmed_string(payload.get("subject"), max_length=_MAX_SUBJECT_LEN)
        except ValueError:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    locale: str | None = None
    if "locale" in payload:
        try:
            locale = _validate_optional_trimmed_string(payload.get("locale"), max_length=_MAX_LOCALE_LEN)
        except ValueError:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    event_detail_type: str | None = None
    if "eventDetail" in payload and payload.get("eventDetail") is not None:
        event_detail = payload.get("eventDetail")
        if not isinstance(event_detail, dict):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        odata_type = event_detail.get("@odata.type")
        if not isinstance(odata_type, str):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        try:
            event_detail_type = _validate_optional_trimmed_string(
                odata_type,
                max_length=_MAX_EVENT_DETAIL_TYPE_LEN,
            )
            if event_detail_type is None:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        except ValueError:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    body_kind: MsGraphTeamsChannelBodyKind | None = None
    body_content: str | None = None
    sender: MsGraphTeamsIdentity | None = None
    attachments: tuple[MsGraphTeamsChatAttachmentReference, ...] = ()
    mentions: tuple[MsGraphTeamsChatMention, ...] = ()
    reactions: tuple[MsGraphTeamsChatReaction, ...] = ()

    if state is MsGraphTeamsChannelMessageState.ACTIVE:
        for key in ("body", "from", "attachments", "mentions", "reactions"):
            if key not in payload:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

        body = payload.get("body")
        if not isinstance(body, dict):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        content_type = body.get("contentType")
        if not isinstance(content_type, str):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        normalized_type = content_type.strip().lower()
        body_kind = _BODY_KIND_MAP.get(normalized_type)
        if body_kind is None:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        raw_content = body.get("content")
        if not isinstance(raw_content, str):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        if "\x00" in raw_content:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        if len(raw_content) > max_chars:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        body_content = raw_content

        try:
            sender = _parse_from_identity_set(payload.get("from"))
        except ValueError:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

        raw_attachments = payload.get("attachments")
        if not isinstance(raw_attachments, list):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        parsed_attachments: list[MsGraphTeamsChatAttachmentReference] = []
        for item in raw_attachments:
            if not isinstance(item, dict):
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
            try:
                parsed_attachments.append(_parse_attachment_reference(item))
            except ValueError:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        attachments = tuple(parsed_attachments)

        raw_mentions = payload.get("mentions")
        if not isinstance(raw_mentions, list):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        parsed_mentions: list[MsGraphTeamsChatMention] = []
        for item in raw_mentions:
            if not isinstance(item, dict):
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
            try:
                mention_id = item.get("id")
                if type(mention_id) is not int or mention_id < 0:
                    raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
                mention_text = _validate_bounded_string(
                    item.get("mentionedText"),
                    max_length=_MAX_MENTION_TEXT_LEN,
                )
                mentioned = _parse_mentioned_identity_set(item.get("mentioned"))
                parsed_mentions.append(
                    MsGraphTeamsChatMention(
                        mention_id=mention_id,
                        mention_text=mention_text,
                        mentioned=mentioned,
                    )
                )
            except ValueError:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        mentions = tuple(parsed_mentions)

        raw_reactions = payload.get("reactions")
        if not isinstance(raw_reactions, list):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        parsed_reactions: list[MsGraphTeamsChatReaction] = []
        for item in raw_reactions:
            if not isinstance(item, dict):
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
            try:
                reaction_type = item.get("reactionType")
                created_raw = item.get("createdDateTime")
                user_identity = _parse_reaction_identity_set(item.get("user"))
                display_name = None
                if "displayName" in item:
                    display_name = _validate_optional_trimmed_string(
                        item.get("displayName"),
                        max_length=_MAX_DISPLAY_NAME_LEN,
                    )
                parsed_reactions.append(
                    MsGraphTeamsChatReaction(
                        reaction_type=reaction_type,
                        display_name=display_name,
                        created_at=_parse_timezone_aware_datetime(created_raw),
                        user=user_identity,
                    )
                )
            except ValueError:
                raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        reactions = tuple(parsed_reactions)

    return _safe_construct_message(
        team_remote_id=validated_team_id,
        channel_remote_id=validated_channel_id,
        thread_root_remote_id=thread_root_remote_id,
        message_kind=message_kind,
        remote_id=remote_id,
        revision=revision,
        state=state,
        message_type=message_type,
        importance=importance,
        created_at=created_at,
        last_modified_at=last_modified_at,
        last_edited_at=last_edited_at,
        deleted_at=deleted_at,
        subject=subject,
        body_kind=body_kind,
        body_content=body_content,
        sender=sender,
        attachments=attachments,
        mentions=mentions,
        reactions=reactions,
        event_detail_type=event_detail_type,
        locale=locale,
    )


def validate_msgraph_teams_channel_message(
    value: object,
    *,
    max_chars: int = ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
) -> MsGraphTeamsChannelMessage:
    try:
        validated_max_chars = _validate_message_max_chars(max_chars)
    except ValueError:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    if isinstance(value, MsGraphTeamsChannelMessage):
        source: object = value.model_dump(mode="python")
    elif isinstance(value, dict):
        source = value
    else:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
    if not isinstance(source, dict):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
    try:
        dumped = dict(source)
        if dumped.get("sender") is not None:
            dumped["sender"] = validate_msgraph_teams_identity(dumped["sender"])
        raw_attachments = dumped.get("attachments", ())
        if type(raw_attachments) is not tuple:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        dumped["attachments"] = tuple(
            validate_msgraph_teams_chat_attachment_reference(item)
            for item in raw_attachments
        )
        raw_mentions = dumped.get("mentions", ())
        if type(raw_mentions) is not tuple:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        dumped["mentions"] = tuple(
            validate_msgraph_teams_chat_mention(item) for item in raw_mentions
        )
        raw_reactions = dumped.get("reactions", ())
        if type(raw_reactions) is not tuple:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        dumped["reactions"] = tuple(
            validate_msgraph_teams_chat_reaction(item) for item in raw_reactions
        )
        validated = MsGraphTeamsChannelMessage.model_validate(dumped)
        if (
            validated.state is MsGraphTeamsChannelMessageState.ACTIVE
            and validated.body_content is not None
            and len(validated.body_content) > validated_max_chars
        ):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE)
        return validated
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None


def validate_msgraph_teams_channel_root_message_page(
    value: object,
    *,
    team_id: str,
    channel_id: str,
    graph_base_url: str,
    max_chars_per_message: int,
) -> MsGraphTeamsChannelRootMessagePage:
    try:
        validated_max_chars = _validate_message_max_chars(max_chars_per_message)
    except ValueError:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    if not isinstance(value, MsGraphTeamsChannelRootMessagePage):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    try:
        validated_team_id = validate_msgraph_teams_team_id(team_id)
        validated_channel_id = validate_msgraph_teams_channel_id(channel_id)
    except ValueError:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    try:
        raw_team_id = value.team_remote_id
        raw_channel_id = value.channel_remote_id
        raw_items = value.items
        raw_continuation = value.continuation
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    if raw_team_id != validated_team_id or raw_channel_id != validated_channel_id:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    if type(raw_items) is not tuple:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    validated_items: list[MsGraphTeamsChannelMessage] = []
    for item in raw_items:
        if not isinstance(item, MsGraphTeamsChannelMessage):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        validated_item = validate_msgraph_teams_channel_message(item, max_chars=validated_max_chars)
        if (
            validated_item.team_remote_id != validated_team_id
            or validated_item.channel_remote_id != validated_channel_id
            or validated_item.message_kind is not MsGraphTeamsChannelMessageKind.ROOT
        ):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        validated_items.append(validated_item)

    remote_ids = [item.remote_id for item in validated_items]
    if len(remote_ids) != len(set(remote_ids)):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    validated_continuation: MsGraphKnowledgeContinuation | None = None
    if raw_continuation is not None:
        if not isinstance(raw_continuation, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        try:
            validated_continuation = validate_msgraph_teams_channel_root_messages_continuation(
                raw_continuation,
                team_id=validated_team_id,
                channel_id=validated_channel_id,
                graph_base_url=graph_base_url,
            )
        except IntegrationConfigurationError:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    try:
        return MsGraphTeamsChannelRootMessagePage(
            team_remote_id=validated_team_id,
            channel_remote_id=validated_channel_id,
            items=tuple(validated_items),
            continuation=validated_continuation,
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None


def validate_msgraph_teams_channel_reply_page(
    value: object,
    *,
    team_id: str,
    channel_id: str,
    root_message_remote_id: str,
    root_message_revision: str,
    graph_base_url: str,
    max_chars_per_message: int,
) -> MsGraphTeamsChannelReplyPage:
    try:
        validated_max_chars = _validate_message_max_chars(max_chars_per_message)
    except ValueError:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    if not isinstance(value, MsGraphTeamsChannelReplyPage):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    try:
        validated_team_id = validate_msgraph_teams_team_id(team_id)
        validated_channel_id = validate_msgraph_teams_channel_id(channel_id)
        validated_root_message = validate_msgraph_teams_channel_message_id(
            root_message_remote_id
        )
        validated_root_revision = _validate_revision(root_message_revision)
    except ValueError:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    try:
        raw_team_id = value.team_remote_id
        raw_channel_id = value.channel_remote_id
        raw_root_message = value.root_message_remote_id
        raw_root_revision = value.root_message_revision
        raw_items = value.items
        raw_continuation = value.continuation
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    if (
        raw_team_id != validated_team_id
        or raw_channel_id != validated_channel_id
        or raw_root_message != validated_root_message
        or raw_root_revision != validated_root_revision
    ):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    if type(raw_items) is not tuple:
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    validated_items: list[MsGraphTeamsChannelMessage] = []
    for item in raw_items:
        if not isinstance(item, MsGraphTeamsChannelMessage):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        validated_item = validate_msgraph_teams_channel_message(item, max_chars=validated_max_chars)
        if (
            validated_item.team_remote_id != validated_team_id
            or validated_item.channel_remote_id != validated_channel_id
            or validated_item.thread_root_remote_id != validated_root_message
            or validated_item.message_kind is not MsGraphTeamsChannelMessageKind.REPLY
        ):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        validated_items.append(validated_item)

    remote_ids = [item.remote_id for item in validated_items]
    if len(remote_ids) != len(set(remote_ids)):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    validated_continuation: MsGraphKnowledgeContinuation | None = None
    if raw_continuation is not None:
        if not isinstance(raw_continuation, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None
        try:
            validated_continuation = validate_msgraph_teams_channel_replies_continuation(
                raw_continuation,
                team_id=validated_team_id,
                channel_id=validated_channel_id,
                root_message_remote_id=validated_root_message,
                graph_base_url=graph_base_url,
            )
        except IntegrationConfigurationError:
            raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None

    try:
        return MsGraphTeamsChannelReplyPage(
            team_remote_id=validated_team_id,
            channel_remote_id=validated_channel_id,
            root_message_remote_id=validated_root_message,
            root_message_revision=validated_root_revision,
            items=tuple(validated_items),
            continuation=validated_continuation,
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_MESSAGES_RESPONSE) from None


def _graph_base_path(graph_base_url: str) -> str:
    parsed_base = urlparse(graph_base_url)
    return parsed_base.path.rstrip("/") or "/"


def _decode_odata_literal(literal: str) -> str:
    return literal.replace("''", "'")


def _decode_path_segment(segment: str, *, odata_literal: bool) -> str:
    decoded = unquote(segment)
    if odata_literal:
        return _decode_odata_literal(decoded)
    return decoded


def _extract_channel_root_messages_path(
    path: str,
    *,
    graph_base_path: str,
) -> tuple[str, str] | None:
    normalized = path.rstrip("/") or "/"
    base = graph_base_path.rstrip("/") or "/"

    patterns: list[tuple[str, bool, bool]] = [
        (rf"^{re.escape(base)}/teams/([^/]+)/channels/([^/]+)/messages$", False, False),
        (
            rf"^{re.escape(base)}/teams\('((?:[^']|'')*)'\)/channels\('((?:[^']|'')*)'\)/messages$",
            True,
            True,
        ),
        (rf"^{re.escape(base)}/teams/([^/]+)/channels\('((?:[^']|'')*)'\)/messages$", False, True),
        (rf"^{re.escape(base)}/teams\('((?:[^']|'')*)'\)/channels/([^/]+)/messages$", True, False),
    ]

    for pattern, team_odata, channel_odata in patterns:
        match = re.fullmatch(pattern, normalized, re.IGNORECASE)
        if match is None:
            continue
        team_segment = match.group(1)
        channel_segment = match.group(2)
        if not team_segment or not channel_segment:
            return None
        return (
            _decode_path_segment(team_segment, odata_literal=team_odata),
            _decode_path_segment(channel_segment, odata_literal=channel_odata),
        )
    return None


def _extract_channel_reply_messages_path(
    path: str,
    *,
    graph_base_path: str,
) -> tuple[str, str, str] | None:
    normalized = path.rstrip("/") or "/"
    base = graph_base_path.rstrip("/") or "/"

    patterns: list[tuple[str, bool, bool, bool]] = [
        (
            rf"^{re.escape(base)}/teams/([^/]+)/channels/([^/]+)/messages/([^/]+)/replies$",
            False,
            False,
            False,
        ),
        (
            rf"^{re.escape(base)}/teams\('((?:[^']|'')*)'\)/channels\('((?:[^']|'')*)'\)/messages\('((?:[^']|'')*)'\)/replies$",
            True,
            True,
            True,
        ),
        (
            rf"^{re.escape(base)}/teams/([^/]+)/channels\('((?:[^']|'')*)'\)/messages/([^/]+)/replies$",
            False,
            True,
            False,
        ),
        (
            rf"^{re.escape(base)}/teams/([^/]+)/channels/([^/]+)/messages\('((?:[^']|'')*)'\)/replies$",
            False,
            False,
            True,
        ),
        (
            rf"^{re.escape(base)}/teams\('((?:[^']|'')*)'\)/channels/([^/]+)/messages/([^/]+)/replies$",
            True,
            False,
            False,
        ),
        (
            rf"^{re.escape(base)}/teams\('((?:[^']|'')*)'\)/channels/([^/]+)/messages\('((?:[^']|'')*)'\)/replies$",
            True,
            False,
            True,
        ),
        (
            rf"^{re.escape(base)}/teams/([^/]+)/channels\('((?:[^']|'')*)'\)/messages\('((?:[^']|'')*)'\)/replies$",
            False,
            True,
            True,
        ),
        (
            rf"^{re.escape(base)}/teams\('((?:[^']|'')*)'\)/channels\('((?:[^']|'')*)'\)/messages/([^/]+)/replies$",
            True,
            True,
            False,
        ),
    ]

    for pattern, team_odata, channel_odata, message_odata in patterns:
        match = re.fullmatch(pattern, normalized, re.IGNORECASE)
        if match is None:
            continue
        team_segment = match.group(1)
        channel_segment = match.group(2)
        message_segment = match.group(3)
        if not team_segment or not channel_segment or not message_segment:
            return None
        return (
            _decode_path_segment(team_segment, odata_literal=team_odata),
            _decode_path_segment(channel_segment, odata_literal=channel_odata),
            _decode_path_segment(message_segment, odata_literal=message_odata),
        )
    return None


def validate_msgraph_teams_channel_root_messages_continuation(
    continuation: object,
    *,
    team_id: str,
    channel_id: str,
    graph_base_url: str,
) -> MsGraphKnowledgeContinuation:
    if not isinstance(continuation, MsGraphKnowledgeContinuation):
        raise IntegrationConfigurationError(_INVALID_MESSAGES_CONTINUATION) from None

    try:
        revalidated = MsGraphKnowledgeContinuation.model_validate(
            continuation.model_dump(mode="python")
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise IntegrationConfigurationError(_INVALID_MESSAGES_CONTINUATION) from None

    if revalidated.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
        raise IntegrationConfigurationError(_INVALID_MESSAGES_CONTINUATION) from None

    try:
        validated_url = validate_msgraph_continuation_url(
            revalidated.url,
            graph_base_url=graph_base_url,
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MESSAGES_CONTINUATION) from None

    parsed = urlparse(validated_url)
    extracted = _extract_channel_root_messages_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted is None:
        raise IntegrationConfigurationError(_INVALID_MESSAGES_CONTINUATION) from None

    extracted_team, extracted_channel = extracted
    try:
        validated_team = validate_msgraph_teams_team_id(team_id)
        validated_channel = validate_msgraph_teams_channel_id(channel_id)
        validated_extracted_team = validate_msgraph_teams_team_id(extracted_team)
        validated_extracted_channel = validate_msgraph_teams_channel_id(extracted_channel)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MESSAGES_CONTINUATION) from None

    if (
        validated_extracted_team != validated_team
        or validated_extracted_channel != validated_channel
    ):
        raise IntegrationConfigurationError(_INVALID_MESSAGES_CONTINUATION) from None

    return revalidated


def validate_msgraph_teams_channel_replies_continuation(
    continuation: object,
    *,
    team_id: str,
    channel_id: str,
    root_message_remote_id: str,
    graph_base_url: str,
) -> MsGraphKnowledgeContinuation:
    if not isinstance(continuation, MsGraphKnowledgeContinuation):
        raise IntegrationConfigurationError(_INVALID_MESSAGES_CONTINUATION) from None

    try:
        revalidated = MsGraphKnowledgeContinuation.model_validate(
            continuation.model_dump(mode="python")
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise IntegrationConfigurationError(_INVALID_MESSAGES_CONTINUATION) from None

    if revalidated.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
        raise IntegrationConfigurationError(_INVALID_MESSAGES_CONTINUATION) from None

    try:
        validated_url = validate_msgraph_continuation_url(
            revalidated.url,
            graph_base_url=graph_base_url,
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MESSAGES_CONTINUATION) from None

    parsed = urlparse(validated_url)
    extracted = _extract_channel_reply_messages_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted is None:
        raise IntegrationConfigurationError(_INVALID_MESSAGES_CONTINUATION) from None

    extracted_team, extracted_channel, extracted_thread_root = extracted
    try:
        validated_team = validate_msgraph_teams_team_id(team_id)
        validated_channel = validate_msgraph_teams_channel_id(channel_id)
        validated_root_message = validate_msgraph_teams_channel_message_id(root_message_remote_id)
        validated_extracted_team = validate_msgraph_teams_team_id(extracted_team)
        validated_extracted_channel = validate_msgraph_teams_channel_id(extracted_channel)
        validated_extracted_root_message = validate_msgraph_teams_channel_message_id(
            extracted_thread_root
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MESSAGES_CONTINUATION) from None

    if (
        validated_extracted_team != validated_team
        or validated_extracted_channel != validated_channel
        or validated_extracted_root_message != validated_root_message
    ):
        raise IntegrationConfigurationError(_INVALID_MESSAGES_CONTINUATION) from None

    return revalidated


def _deduplicate_messages(
    items: tuple[MsGraphTeamsChannelMessage, ...],
) -> tuple[MsGraphTeamsChannelMessage, ...]:
    last_by_id: dict[str, MsGraphTeamsChannelMessage] = {}
    order: list[str] = []
    for item in items:
        if item.remote_id not in last_by_id:
            order.append(item.remote_id)
        else:
            order.remove(item.remote_id)
            order.append(item.remote_id)
        last_by_id[item.remote_id] = item
    return tuple(last_by_id[remote_id] for remote_id in order)


def _validate_message_limits(limit: object, max_chars_per_message: object) -> tuple[int, int]:
    if type(limit) is not int:
        raise IntegrationConfigurationError(_INVALID_MESSAGES_REQUEST)
    if limit < _MIN_MESSAGE_LIMIT or limit > _MAX_MESSAGE_LIMIT:
        raise IntegrationConfigurationError(_INVALID_MESSAGES_REQUEST)
    try:
        validated_max_chars = _validate_message_max_chars(max_chars_per_message)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MESSAGES_REQUEST) from None
    return limit, validated_max_chars


def _compare_message_observation(
    payload: dict[str, object],
    *,
    message: MsGraphTeamsChannelMessage,
) -> None:
    try:
        response_id = validate_msgraph_teams_channel_message_id(payload.get("id"))
        response_revision = _validate_revision(payload.get("etag"))
        deleted_raw = payload.get("deletedDateTime")
        is_deleted = deleted_raw is not None
        _parse_channel_identity(
            payload.get("channelIdentity"),
            expected_team_id=message.team_remote_id,
            expected_channel_id=message.channel_remote_id,
        )
    except ValueError:
        raise MsGraphTeamsChannelMessageChanged() from None

    if payload.get("chatId") is not None:
        raise MsGraphTeamsChannelMessageChanged() from None

    reply_raw = payload.get("replyToId")
    if message.message_kind is MsGraphTeamsChannelMessageKind.ROOT:
        if reply_raw is not None:
            raise MsGraphTeamsChannelMessageChanged() from None
    else:
        if reply_raw is None:
            raise MsGraphTeamsChannelMessageChanged() from None
        try:
            reply_to_id = validate_msgraph_teams_channel_message_id(reply_raw)
        except ValueError:
            raise MsGraphTeamsChannelMessageChanged() from None
        if reply_to_id != message.thread_root_remote_id:
            raise MsGraphTeamsChannelMessageChanged() from None

    expected_deleted = message.state is MsGraphTeamsChannelMessageState.DELETED
    if (
        response_id != message.remote_id
        or response_revision != message.revision
        or is_deleted != expected_deleted
    ):
        raise MsGraphTeamsChannelMessageChanged() from None


def read_and_validate_current_teams_channel_message_observation(
    *,
    message: MsGraphTeamsChannelMessage,
    transport: MsGraphKnowledgeTransport,
) -> None:
    validated_message = validate_msgraph_teams_channel_message(message)
    quoted_team = quote(validated_message.team_remote_id, safe="")
    quoted_channel = quote(validated_message.channel_remote_id, safe="")
    if validated_message.message_kind is MsGraphTeamsChannelMessageKind.ROOT:
        quoted_root = quote(validated_message.remote_id, safe="")
        path = f"/teams/{quoted_team}/channels/{quoted_channel}/messages/{quoted_root}"
    else:
        quoted_root = quote(validated_message.thread_root_remote_id, safe="")
        quoted_reply = quote(validated_message.remote_id, safe="")
        path = (
            f"/teams/{quoted_team}/channels/{quoted_channel}/messages/{quoted_root}"
            f"/replies/{quoted_reply}"
        )
    payload = transport.get_initial_json(
        path=path,
        headers=_PREFER_UNKNOWN_ENUM,
        not_found_is_dependency=True,
    )
    _compare_message_observation(payload, message=validated_message)


@runtime_checkable
class MsGraphTeamsChannelMessagesReadClient(Protocol):
    def read_teams_channel_root_messages_page(
        self,
        *,
        channel: MsGraphTeamsChannel,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
        max_chars_per_message: int,
    ) -> MsGraphTeamsChannelRootMessagePage:
        ...

    def read_teams_channel_replies_page(
        self,
        *,
        root_message: MsGraphTeamsChannelMessage,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
        max_chars_per_message: int,
    ) -> MsGraphTeamsChannelReplyPage:
        ...


class MsGraphTeamsChannelMessagesReader:
    """Paged Teams channel root and reply message inventory."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        transport: MsGraphKnowledgeTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    def read_teams_channel_root_messages_page(
        self,
        *,
        channel: MsGraphTeamsChannel,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
        max_chars_per_message: int,
    ) -> MsGraphTeamsChannelRootMessagePage:
        try:
            validated_channel = validate_msgraph_teams_channel(channel)
        except ValueError:
            raise IntegrationConfigurationError(_INVALID_MESSAGES_REQUEST) from None

        validated_limit, validated_max_chars = _validate_message_limits(
            limit,
            max_chars_per_message,
        )

        if continuation is None:
            quoted_team = quote(validated_channel.team_remote_id, safe="")
            quoted_channel = quote(validated_channel.remote_id, safe="")
            path = f"/teams/{quoted_team}/channels/{quoted_channel}/messages"
            payload = self._transport.get_initial_json(
                path=path,
                params={"$top": validated_limit},
                headers=_PREFER_UNKNOWN_ENUM,
            )
        else:
            validated_continuation = validate_msgraph_teams_channel_root_messages_continuation(
                continuation,
                team_id=validated_channel.team_remote_id,
                channel_id=validated_channel.remote_id,
                graph_base_url=self._config.graph_base_url,
            )
            payload = self._transport.get_continuation_json(
                continuation=validated_continuation,
                headers=_PREFER_UNKNOWN_ENUM,
            )

        collection_page = parse_msgraph_collection_page(
            payload,
            graph_base_url=self._config.graph_base_url,
            delta_mode=False,
        )

        parsed_items: list[MsGraphTeamsChannelMessage] = []
        for raw_item in collection_page.items:
            parsed_items.append(
                parse_msgraph_teams_channel_message(
                    raw_item,
                    expected_team_id=validated_channel.team_remote_id,
                    expected_channel_id=validated_channel.remote_id,
                    message_kind=MsGraphTeamsChannelMessageKind.ROOT,
                    max_chars=validated_max_chars,
                )
            )

        deduplicated = _deduplicate_messages(tuple(parsed_items))
        return validate_msgraph_teams_channel_root_message_page(
            _safe_construct_root_page(
                team_remote_id=validated_channel.team_remote_id,
                channel_remote_id=validated_channel.remote_id,
                items=deduplicated,
                continuation=collection_page.continuation,
            ),
            team_id=validated_channel.team_remote_id,
            channel_id=validated_channel.remote_id,
            graph_base_url=self._config.graph_base_url,
            max_chars_per_message=validated_max_chars,
        )

    def read_teams_channel_replies_page(
        self,
        *,
        root_message: MsGraphTeamsChannelMessage,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
        max_chars_per_message: int,
    ) -> MsGraphTeamsChannelReplyPage:
        try:
            validated_root = validate_msgraph_teams_channel_message(root_message)
        except ValueError:
            raise IntegrationConfigurationError(_INVALID_MESSAGES_REQUEST) from None

        if validated_root.message_kind is not MsGraphTeamsChannelMessageKind.ROOT:
            raise IntegrationConfigurationError(_INVALID_MESSAGES_REQUEST) from None

        validated_limit, validated_max_chars = _validate_message_limits(
            limit,
            max_chars_per_message,
        )

        read_and_validate_current_teams_channel_message_observation(
            message=validated_root,
            transport=self._transport,
        )

        if continuation is None:
            quoted_team = quote(validated_root.team_remote_id, safe="")
            quoted_channel = quote(validated_root.channel_remote_id, safe="")
            quoted_root = quote(validated_root.remote_id, safe="")
            path = (
                f"/teams/{quoted_team}/channels/{quoted_channel}/messages/"
                f"{quoted_root}/replies"
            )
            payload = self._transport.get_initial_json(
                path=path,
                params={"$top": validated_limit},
                headers=_PREFER_UNKNOWN_ENUM,
            )
        else:
            validated_continuation = validate_msgraph_teams_channel_replies_continuation(
                continuation,
                team_id=validated_root.team_remote_id,
                channel_id=validated_root.channel_remote_id,
                root_message_remote_id=validated_root.remote_id,
                graph_base_url=self._config.graph_base_url,
            )
            payload = self._transport.get_continuation_json(
                continuation=validated_continuation,
                headers=_PREFER_UNKNOWN_ENUM,
            )

        collection_page = parse_msgraph_collection_page(
            payload,
            graph_base_url=self._config.graph_base_url,
            delta_mode=False,
        )

        parsed_items: list[MsGraphTeamsChannelMessage] = []
        for raw_item in collection_page.items:
            parsed_items.append(
                parse_msgraph_teams_channel_message(
                    raw_item,
                    expected_team_id=validated_root.team_remote_id,
                    expected_channel_id=validated_root.channel_remote_id,
                    message_kind=MsGraphTeamsChannelMessageKind.REPLY,
                    expected_thread_root_remote_id=validated_root.remote_id,
                    max_chars=validated_max_chars,
                )
            )

        deduplicated = _deduplicate_messages(tuple(parsed_items))
        page = validate_msgraph_teams_channel_reply_page(
            _safe_construct_reply_page(
                team_remote_id=validated_root.team_remote_id,
                channel_remote_id=validated_root.channel_remote_id,
                root_message_remote_id=validated_root.remote_id,
                root_message_revision=validated_root.revision,
                items=deduplicated,
                continuation=collection_page.continuation,
            ),
            team_id=validated_root.team_remote_id,
            channel_id=validated_root.channel_remote_id,
            root_message_remote_id=validated_root.remote_id,
            root_message_revision=validated_root.revision,
            graph_base_url=self._config.graph_base_url,
            max_chars_per_message=validated_max_chars,
        )

        read_and_validate_current_teams_channel_message_observation(
            message=validated_root,
            transport=self._transport,
        )
        return page
