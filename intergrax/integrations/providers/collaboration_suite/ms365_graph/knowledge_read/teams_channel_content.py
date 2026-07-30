# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Teams Channel knowledge-read: exact message content."""

from __future__ import annotations

import re
from typing import Protocol, Self, runtime_checkable
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.common import (
    MsGraphKnowledgeTransport,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (
    validate_msgraph_teams_channel_id,
    validate_msgraph_teams_channel_message_id,
    validate_msgraph_teams_team_id,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (
    ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    MsGraphTeamsChannelMessage,
    MsGraphTeamsChannelMessageChanged,
    MsGraphTeamsChannelMessageKind,
    MsGraphTeamsChannelMessageState,
    parse_msgraph_teams_channel_message,
    validate_msgraph_teams_channel_message,
)

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_CONTENT_RESPONSE = "unexpected Microsoft Graph Teams channel message content response"
_INVALID_CONTENT_REQUEST = "invalid Microsoft Graph Teams channel message content request"
_MAX_REVISION_LEN = 4096
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_PREFER_UNKNOWN_ENUM = {"Prefer": "include-unknown-enum-members"}


class MsGraphTeamsChannelContentTooLarge(IntegrationConfigurationError):
    """Teams channel message body exceeds the configured character limit."""

    def __init__(self) -> None:
        super().__init__(
            "Microsoft Graph Teams channel message exceeds the configured content limit"
        )


def _validate_revision(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CONTENT_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CONTENT_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CONTENT_RESPONSE)
    if len(trimmed) > _MAX_REVISION_LEN:
        raise ValueError(_MALFORMED_CONTENT_RESPONSE)
    return trimmed


def _validate_max_chars(max_chars: object) -> int:
    if type(max_chars) is not int:
        raise IntegrationConfigurationError(_INVALID_CONTENT_REQUEST) from None
    if max_chars < 1 or max_chars > ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS:
        raise IntegrationConfigurationError(_INVALID_CONTENT_REQUEST) from None
    return max_chars


class MsGraphTeamsChannelMessageReference(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    team_remote_id: str = Field(repr=False)
    channel_remote_id: str = Field(repr=False)
    thread_root_remote_id: str = Field(repr=False)
    message_kind: MsGraphTeamsChannelMessageKind
    remote_id: str = Field(repr=False)
    revision: str = Field(repr=False)

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
    def _validate_revision_field(cls, value: object) -> str:
        return _validate_revision(value)

    @field_validator("message_kind", mode="before")
    @classmethod
    def _validate_message_kind_field(cls, value: object) -> MsGraphTeamsChannelMessageKind:
        if not isinstance(value, MsGraphTeamsChannelMessageKind):
            raise ValueError(_MALFORMED_CONTENT_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_reference_shape(self) -> Self:
        if self.message_kind is MsGraphTeamsChannelMessageKind.ROOT:
            if self.thread_root_remote_id != self.remote_id:
                raise ValueError(_MALFORMED_CONTENT_RESPONSE)
        elif self.thread_root_remote_id == self.remote_id:
            raise ValueError(_MALFORMED_CONTENT_RESPONSE)
        return self


def validate_msgraph_teams_channel_message_reference(
    value: object,
) -> MsGraphTeamsChannelMessageReference:
    if isinstance(value, MsGraphTeamsChannelMessageReference):
        source: object = value.model_dump(mode="python")
    elif isinstance(value, dict):
        source = value
    else:
        raise ValueError(_MALFORMED_CONTENT_RESPONSE) from None
    if not isinstance(source, dict):
        raise ValueError(_MALFORMED_CONTENT_RESPONSE) from None
    try:
        dumped = dict(source)
        dumped["team_remote_id"] = validate_msgraph_teams_team_id(dumped.get("team_remote_id"))
        dumped["channel_remote_id"] = validate_msgraph_teams_channel_id(dumped.get("channel_remote_id"))
        dumped["thread_root_remote_id"] = validate_msgraph_teams_channel_message_id(
            dumped.get("thread_root_remote_id")
        )
        dumped["remote_id"] = validate_msgraph_teams_channel_message_id(dumped.get("remote_id"))
        dumped["revision"] = _validate_revision(dumped.get("revision"))
        message_kind = dumped.get("message_kind")
        if not isinstance(message_kind, MsGraphTeamsChannelMessageKind):
            raise ValueError(_MALFORMED_CONTENT_RESPONSE)
        dumped["message_kind"] = message_kind
        return MsGraphTeamsChannelMessageReference.model_validate(dumped)
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CONTENT_RESPONSE) from None


def validate_msgraph_teams_channel_message_content(
    value: object,
    *,
    reference: MsGraphTeamsChannelMessageReference,
    max_chars: int,
) -> MsGraphTeamsChannelMessage:
    validated_max_chars = _validate_max_chars(max_chars)
    validated_reference = validate_msgraph_teams_channel_message_reference(reference)

    if not isinstance(value, MsGraphTeamsChannelMessage):
        raise ValueError(_MALFORMED_CONTENT_RESPONSE) from None

    try:
        source = value.model_dump(mode="python")
        if not isinstance(source, dict):
            raise ValueError(_MALFORMED_CONTENT_RESPONSE)
        revalidate_source = dict(source)
        revalidate_source["sender"] = value.sender
        revalidate_source["attachments"] = value.attachments
        revalidate_source["mentions"] = value.mentions
        revalidate_source["reactions"] = value.reactions
        validated_message = validate_msgraph_teams_channel_message(
            revalidate_source,
            max_chars=ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )
    except (ValueError, TypeError, AttributeError, ValidationError, KeyError):
        raise ValueError(_MALFORMED_CONTENT_RESPONSE) from None

    if (
        validated_message.team_remote_id != validated_reference.team_remote_id
        or validated_message.channel_remote_id != validated_reference.channel_remote_id
        or validated_message.thread_root_remote_id != validated_reference.thread_root_remote_id
        or validated_message.message_kind != validated_reference.message_kind
        or validated_message.remote_id != validated_reference.remote_id
        or validated_message.revision != validated_reference.revision
    ):
        raise MsGraphTeamsChannelMessageChanged() from None

    if validated_message.state is not MsGraphTeamsChannelMessageState.ACTIVE:
        raise MsGraphTeamsChannelMessageChanged() from None
    if validated_message.body_kind is None or validated_message.body_content is None:
        raise MsGraphTeamsChannelMessageChanged() from None

    if len(validated_message.body_content) > validated_max_chars:
        raise MsGraphTeamsChannelContentTooLarge() from None

    return validated_message


@runtime_checkable
class MsGraphTeamsChannelContentReadClient(Protocol):
    def read_teams_channel_message_content(
        self,
        *,
        message: MsGraphTeamsChannelMessageReference,
        max_chars: int,
    ) -> MsGraphTeamsChannelMessage:
        ...


class MsGraphTeamsChannelContentReader:
    """Exact Teams channel root and reply message content reader."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        transport: MsGraphKnowledgeTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    def read_message_content(
        self,
        *,
        message: MsGraphTeamsChannelMessageReference,
        max_chars: int,
    ) -> MsGraphTeamsChannelMessage:
        validated_reference = validate_msgraph_teams_channel_message_reference(message)
        validated_max_chars = _validate_max_chars(max_chars)

        quoted_team = quote(validated_reference.team_remote_id, safe="")
        quoted_channel = quote(validated_reference.channel_remote_id, safe="")
        if validated_reference.message_kind is MsGraphTeamsChannelMessageKind.ROOT:
            quoted_message = quote(validated_reference.remote_id, safe="")
            path = f"/teams/{quoted_team}/channels/{quoted_channel}/messages/{quoted_message}"
            message_kind = MsGraphTeamsChannelMessageKind.ROOT
            expected_thread_root: str | None = None
        else:
            quoted_root = quote(validated_reference.thread_root_remote_id, safe="")
            quoted_reply = quote(validated_reference.remote_id, safe="")
            path = (
                f"/teams/{quoted_team}/channels/{quoted_channel}/messages/{quoted_root}"
                f"/replies/{quoted_reply}"
            )
            message_kind = MsGraphTeamsChannelMessageKind.REPLY
            expected_thread_root = validated_reference.thread_root_remote_id

        payload = self._transport.get_initial_json(
            path=path,
            headers=_PREFER_UNKNOWN_ENUM,
            not_found_is_dependency=True,
        )

        parsed = parse_msgraph_teams_channel_message(
            payload,
            expected_team_id=validated_reference.team_remote_id,
            expected_channel_id=validated_reference.channel_remote_id,
            message_kind=message_kind,
            expected_thread_root_remote_id=expected_thread_root,
            max_chars=ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )

        return validate_msgraph_teams_channel_message_content(
            parsed,
            reference=validated_reference,
            max_chars=validated_max_chars,
        )
