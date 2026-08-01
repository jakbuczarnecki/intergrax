# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Teams Chat knowledge-read: exact message content."""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.common import (
    MsGraphKnowledgeTransport,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_folders import (
    validate_msgraph_mailbox_user_id,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
    _validate_exact_durable_opaque_reference_field,
    validate_msgraph_teams_chat_id,
    validate_msgraph_teams_chat_message_id,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_messages import (
    ABSOLUTE_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    MsGraphTeamsChatMessage,
    MsGraphTeamsChatMessageChanged,
    MsGraphTeamsChatMessageState,
    parse_msgraph_teams_chat_message,
    validate_msgraph_teams_chat_message,
)

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_CONTENT_RESPONSE = "unexpected Microsoft Graph Teams chat message content response"
_INVALID_CONTENT_REQUEST = "invalid Microsoft Graph Teams chat message content request"
_MAX_REVISION_LEN = 4096
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_PREFER_UNKNOWN_ENUM = {"Prefer": "include-unknown-enum-members"}


class MsGraphTeamsChatContentTooLarge(IntegrationConfigurationError):
    """Teams chat message body exceeds the configured character limit."""

    def __init__(self) -> None:
        super().__init__(
            "Microsoft Graph Teams chat message exceeds the configured content limit"
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


def _validate_exact_durable_revision(value: object) -> str:
    return _validate_exact_durable_opaque_reference_field(
        value,
        validator=_validate_revision,
        error=_MALFORMED_CONTENT_RESPONSE,
    )


def _validate_max_chars(max_chars: object) -> int:
    if type(max_chars) is not int:
        raise IntegrationConfigurationError(_INVALID_CONTENT_REQUEST) from None
    if max_chars < 1 or max_chars > ABSOLUTE_TEAMS_CHAT_MESSAGE_MAX_CHARS:
        raise IntegrationConfigurationError(_INVALID_CONTENT_REQUEST) from None
    return max_chars


class MsGraphTeamsChatMessageReference(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str = Field(repr=False)
    chat_remote_id: str = Field(repr=False)
    remote_id: str = Field(repr=False)
    revision: str = Field(repr=False)

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return _validate_exact_durable_opaque_reference_field(
            value,
            validator=validate_msgraph_mailbox_user_id,
            error=_MALFORMED_CONTENT_RESPONSE,
        )

    @field_validator("chat_remote_id", mode="before")
    @classmethod
    def _validate_chat_remote_id(cls, value: object) -> str:
        return _validate_exact_durable_opaque_reference_field(
            value,
            validator=validate_msgraph_teams_chat_id,
            error=_MALFORMED_CONTENT_RESPONSE,
        )

    @field_validator("remote_id", mode="before")
    @classmethod
    def _validate_remote_id(cls, value: object) -> str:
        return _validate_exact_durable_opaque_reference_field(
            value,
            validator=validate_msgraph_teams_chat_message_id,
            error=_MALFORMED_CONTENT_RESPONSE,
        )

    @field_validator("revision", mode="before")
    @classmethod
    def _validate_revision_field(cls, value: object) -> str:
        return _validate_exact_durable_revision(value)


def validate_msgraph_teams_chat_message_reference(
    value: object,
) -> MsGraphTeamsChatMessageReference:
    if isinstance(value, MsGraphTeamsChatMessageReference):
        source: object = value.model_dump(mode="python")
    elif isinstance(value, dict):
        source = value
    else:
        raise ValueError(_MALFORMED_CONTENT_RESPONSE) from None
    if not isinstance(source, dict):
        raise ValueError(_MALFORMED_CONTENT_RESPONSE) from None
    try:
        dumped = dict(source)
        dumped["mailbox_user_id"] = _validate_exact_durable_opaque_reference_field(
            dumped.get("mailbox_user_id"),
            validator=validate_msgraph_mailbox_user_id,
            error=_MALFORMED_CONTENT_RESPONSE,
        )
        dumped["chat_remote_id"] = _validate_exact_durable_opaque_reference_field(
            dumped.get("chat_remote_id"),
            validator=validate_msgraph_teams_chat_id,
            error=_MALFORMED_CONTENT_RESPONSE,
        )
        dumped["remote_id"] = _validate_exact_durable_opaque_reference_field(
            dumped.get("remote_id"),
            validator=validate_msgraph_teams_chat_message_id,
            error=_MALFORMED_CONTENT_RESPONSE,
        )
        dumped["revision"] = _validate_exact_durable_revision(dumped.get("revision"))
        return MsGraphTeamsChatMessageReference.model_validate(dumped)
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CONTENT_RESPONSE) from None


def validate_msgraph_teams_chat_message_content(
    value: object,
    *,
    reference: MsGraphTeamsChatMessageReference,
    max_chars: int,
) -> MsGraphTeamsChatMessage:
    validated_max_chars = _validate_max_chars(max_chars)
    validated_reference = validate_msgraph_teams_chat_message_reference(reference)

    if not isinstance(value, MsGraphTeamsChatMessage):
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
        validated_message = validate_msgraph_teams_chat_message(
            revalidate_source,
            max_chars=ABSOLUTE_TEAMS_CHAT_MESSAGE_MAX_CHARS,
        )
    except (ValueError, TypeError, AttributeError, ValidationError, KeyError):
        raise ValueError(_MALFORMED_CONTENT_RESPONSE) from None

    if (
        validated_message.mailbox_user_id != validated_reference.mailbox_user_id
        or validated_message.chat_remote_id != validated_reference.chat_remote_id
        or validated_message.remote_id != validated_reference.remote_id
        or validated_message.revision != validated_reference.revision
    ):
        raise MsGraphTeamsChatMessageChanged() from None

    if validated_message.state is not MsGraphTeamsChatMessageState.ACTIVE:
        raise MsGraphTeamsChatMessageChanged() from None
    if validated_message.body_kind is None or validated_message.body_content is None:
        raise MsGraphTeamsChatMessageChanged() from None

    if len(validated_message.body_content) > validated_max_chars:
        raise MsGraphTeamsChatContentTooLarge() from None

    return validated_message


@runtime_checkable
class MsGraphTeamsChatContentReadClient(Protocol):
    def read_teams_chat_message_content(
        self,
        *,
        message: MsGraphTeamsChatMessageReference,
        max_chars: int,
    ) -> MsGraphTeamsChatMessage:
        ...


class MsGraphTeamsChatContentReader:
    """Exact Teams chat message content reader."""

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
        message: MsGraphTeamsChatMessageReference,
        max_chars: int,
    ) -> MsGraphTeamsChatMessage:
        validated_reference = validate_msgraph_teams_chat_message_reference(message)
        validated_max_chars = _validate_max_chars(max_chars)

        quoted_mailbox = quote(validated_reference.mailbox_user_id, safe="")
        quoted_chat = quote(validated_reference.chat_remote_id, safe="")
        quoted_message = quote(validated_reference.remote_id, safe="")
        path = f"/users/{quoted_mailbox}/chats/{quoted_chat}/messages/{quoted_message}"

        payload = self._transport.get_initial_json(
            path=path,
            headers=_PREFER_UNKNOWN_ENUM,
            not_found_is_dependency=True,
        )

        parsed = parse_msgraph_teams_chat_message(
            payload,
            expected_mailbox_user_id=validated_reference.mailbox_user_id,
            expected_chat_id=validated_reference.chat_remote_id,
            max_chars=ABSOLUTE_TEAMS_CHAT_MESSAGE_MAX_CHARS,
        )

        return validate_msgraph_teams_chat_message_content(
            parsed,
            reference=validated_reference,
            max_chars=validated_max_chars,
        )
