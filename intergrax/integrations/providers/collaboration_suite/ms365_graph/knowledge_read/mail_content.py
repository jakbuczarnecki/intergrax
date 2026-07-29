# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Mail knowledge-read: text message content and participants."""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.common import (
    MsGraphKnowledgeTransport,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_folders import (
    validate_msgraph_mail_folder_id,
    validate_msgraph_mailbox_user_id,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_messages import (
    MsGraphMailMessageChange,
    MsGraphMailMessageChangeKind,
    parse_msgraph_mail_message_change,
    validate_msgraph_mail_message_change,
    validate_msgraph_mail_message_id,
)

DEFAULT_MAIL_CONTENT_MAX_CHARS = 2_000_000
ABSOLUTE_MAIL_CONTENT_MAX_CHARS = 8_000_000

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_MAIL_CONTENT_RESPONSE = "unexpected Microsoft Graph Mail content response"
_INVALID_MAIL_CONTENT_REQUEST = "invalid Microsoft Graph Mail content request"
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_MAX_PARTICIPANT_ADDRESS_LEN = 2048
_MAX_PARTICIPANT_DISPLAY_NAME_LEN = 1024
_MAX_CONVERSATION_ID_LEN = 2048
_MAX_INTERNET_MESSAGE_ID_LEN = 4096
_MAX_SUBJECT_LEN = 4096
_MAX_CHANGE_KEY_LEN = 2048
_MAX_MESSAGE_TEXT_LEN = 16_000

_IMMUTABLE_TEXT_BODY_HEADERS = {
    "Prefer": 'IdType="ImmutableId", outlook.body-content-type="text"',
}

_OBSERVATION_SELECT = (
    "id,parentFolderId,changeKey,lastModifiedDateTime,isRead,isDraft,hasAttachments,importance"
)

_CONTENT_SELECT = (
    "id,parentFolderId,changeKey,conversationId,internetMessageId,subject,body,uniqueBody,"
    "from,sender,replyTo,toRecipients,ccRecipients,bccRecipients,createdDateTime,"
    "lastModifiedDateTime,receivedDateTime,sentDateTime,isRead,isDraft,hasAttachments,importance"
)


class MsGraphMailMessageChanged(IntegrationDependencyError):
    """Mail message identity, folder or revision changed during read."""

    def __init__(self) -> None:
        super().__init__("Microsoft Graph Mail message changed during read")


class MsGraphMailContentTooLarge(IntegrationConfigurationError):
    """Mail message text exceeds the configured character limit."""

    def __init__(self) -> None:
        super().__init__("Microsoft Graph Mail message exceeds the configured content limit")


def _validate_participant_address(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    if len(trimmed) > _MAX_PARTICIPANT_ADDRESS_LEN:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    return trimmed


def _validate_participant_display_name(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        return None
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    if len(trimmed) > _MAX_PARTICIPANT_DISPLAY_NAME_LEN:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    return trimmed


def _validate_optional_opaque_string(value: object, *, max_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    if len(trimmed) > max_length:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    return trimmed


def _validate_content_revision(value: object) -> str:
    result = _validate_optional_opaque_string(value, max_length=_MAX_CHANGE_KEY_LEN)
    if result is None:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    return result


def _validate_subject(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    if "\x00" in value:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    if len(value) > _MAX_SUBJECT_LEN:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    return value


def _validate_body_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    if "\x00" in value:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    if len(value) > _MAX_MESSAGE_TEXT_LEN:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    return value


def _validate_optional_body_text(value: object) -> str | None:
    if value is None:
        return None
    return _validate_body_text(value)


def _validate_participant_tuple(value: object) -> tuple[MsGraphMailParticipant, ...]:
    if type(value) is not tuple:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE)
    validated: list[MsGraphMailParticipant] = []
    for item in value:
        validated.append(validate_msgraph_mail_participant(item))
    return tuple(validated)


class MsGraphMailParticipant(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    display_name: str | None = Field(default=None, repr=False)
    address: str = Field(repr=False)

    @field_validator("display_name", mode="before")
    @classmethod
    def _validate_display_name_field(cls, value: object) -> str | None:
        return _validate_participant_display_name(value)

    @field_validator("address", mode="before")
    @classmethod
    def _validate_address_field(cls, value: object) -> str:
        return _validate_participant_address(value)


class MsGraphMailMessageContent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    remote_id: str
    parent_folder_id: str
    content_revision: str = Field(repr=False)

    conversation_id: str | None = None
    internet_message_id: str | None = Field(default=None, repr=False)
    subject: str | None = Field(default=None, repr=False)

    body_text: str = Field(repr=False)
    unique_body_text: str | None = Field(default=None, repr=False)

    from_participant: MsGraphMailParticipant | None = Field(default=None, repr=False)
    sender_participant: MsGraphMailParticipant | None = Field(default=None, repr=False)

    reply_to: tuple[MsGraphMailParticipant, ...] = Field(default=(), repr=False)
    to_recipients: tuple[MsGraphMailParticipant, ...] = Field(default=(), repr=False)
    cc_recipients: tuple[MsGraphMailParticipant, ...] = Field(default=(), repr=False)
    bcc_recipients: tuple[MsGraphMailParticipant, ...] = Field(default=(), repr=False)

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("remote_id", mode="before")
    @classmethod
    def _validate_remote_id(cls, value: object) -> str:
        return validate_msgraph_mail_message_id(value)

    @field_validator("parent_folder_id", mode="before")
    @classmethod
    def _validate_parent_folder_id(cls, value: object) -> str:
        return validate_msgraph_mail_folder_id(value)

    @field_validator("content_revision", mode="before")
    @classmethod
    def _validate_content_revision_field(cls, value: object) -> str:
        return _validate_content_revision(value)

    @field_validator("conversation_id", mode="before")
    @classmethod
    def _validate_conversation_id(cls, value: object) -> str | None:
        return _validate_optional_opaque_string(value, max_length=_MAX_CONVERSATION_ID_LEN)

    @field_validator("internet_message_id", mode="before")
    @classmethod
    def _validate_internet_message_id(cls, value: object) -> str | None:
        return _validate_optional_opaque_string(value, max_length=_MAX_INTERNET_MESSAGE_ID_LEN)

    @field_validator("subject", mode="before")
    @classmethod
    def _validate_subject_field(cls, value: object) -> str | None:
        return _validate_subject(value)

    @field_validator("body_text", mode="before")
    @classmethod
    def _validate_body_text_field(cls, value: object) -> str:
        return _validate_body_text(value)

    @field_validator("unique_body_text", mode="before")
    @classmethod
    def _validate_unique_body_text_field(cls, value: object) -> str | None:
        return _validate_optional_body_text(value)

    @field_validator(
        "reply_to",
        "to_recipients",
        "cc_recipients",
        "bcc_recipients",
        mode="before",
    )
    @classmethod
    def _validate_recipient_tuples(cls, value: object) -> tuple[MsGraphMailParticipant, ...]:
        return _validate_participant_tuple(value)


def _safe_construct_participant(**kwargs: object) -> MsGraphMailParticipant:
    try:
        return MsGraphMailParticipant(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None


def _safe_construct_message_content(**kwargs: object) -> MsGraphMailMessageContent:
    try:
        return MsGraphMailMessageContent(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None


def parse_msgraph_mail_participant(payload: object) -> MsGraphMailParticipant:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    email_address = payload.get("emailAddress")
    if not isinstance(email_address, dict):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    if "address" not in email_address:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    try:
        address = _validate_participant_address(email_address.get("address"))
    except ValueError:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    if "name" not in email_address:
        display_name = None
    else:
        name_value = email_address.get("name")
        try:
            display_name = _validate_participant_display_name(name_value)
        except ValueError:
            raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    return _safe_construct_participant(display_name=display_name, address=address)


def _parse_text_body_field(payload: object) -> str:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    if "contentType" not in payload or "content" not in payload:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    content_type = payload.get("contentType")
    content = payload.get("content")
    if not isinstance(content_type, str) or not content_type.strip():
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    if content_type.lower() != "text":
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    if not isinstance(content, str):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    if "\x00" in content:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    return content


def _parse_optional_unique_body(payload: dict[str, object]) -> str | None:
    if "uniqueBody" not in payload:
        return None
    unique_body = payload.get("uniqueBody")
    if unique_body is None:
        return None
    if not isinstance(unique_body, dict):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    return _parse_text_body_field(unique_body)


def _parse_optional_participant(payload: dict[str, object], key: str) -> MsGraphMailParticipant | None:
    if key not in payload:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    return parse_msgraph_mail_participant(value)


def _parse_participant_list(payload: dict[str, object], key: str) -> tuple[MsGraphMailParticipant, ...]:
    if key not in payload:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    return tuple(parse_msgraph_mail_participant(item) for item in value)


def _combined_content_length(body_text: str, unique_body_text: str | None) -> int:
    return len(body_text) + len(unique_body_text or "")


def _enforce_content_limit(
    body_text: str,
    unique_body_text: str | None,
    *,
    max_chars: int,
) -> None:
    if _combined_content_length(body_text, unique_body_text) > max_chars:
        raise MsGraphMailContentTooLarge() from None


def _require_active_message(message: object) -> MsGraphMailMessageChange:
    validated = validate_msgraph_mail_message_change(message)
    if validated.kind is not MsGraphMailMessageChangeKind.ACTIVE:
        raise MsGraphMailMessageChanged() from None
    return validated


def _compare_message_observation(
    payload: dict[str, object],
    *,
    message: MsGraphMailMessageChange,
) -> None:
    try:
        response_id = validate_msgraph_mail_message_id(payload.get("id"))
        response_folder = validate_msgraph_mail_folder_id(payload.get("parentFolderId"))
        response_change_key = _validate_content_revision(payload.get("changeKey"))
    except ValueError:
        raise MsGraphMailMessageChanged() from None
    if (
        response_id != message.remote_id
        or response_folder != message.parent_folder_id
        or response_change_key != message.change_key
    ):
        raise MsGraphMailMessageChanged() from None


def read_and_validate_current_mail_message_observation(
    *,
    message: MsGraphMailMessageChange,
    transport: MsGraphKnowledgeTransport,
) -> MsGraphMailMessageChange:
    validated_message = _require_active_message(message)
    quoted_mailbox = quote(validated_message.mailbox_user_id, safe="")
    quoted_message_id = quote(validated_message.remote_id, safe="")
    path = f"/users/{quoted_mailbox}/messages/{quoted_message_id}"
    payload = transport.get_initial_json(
        path=path,
        params={"$select": _OBSERVATION_SELECT},
        headers={"Prefer": 'IdType="ImmutableId"'},
        not_found_is_dependency=True,
    )
    _compare_message_observation(payload, message=validated_message)
    observed = parse_msgraph_mail_message_change(
        payload,
        expected_mailbox_user_id=validated_message.mailbox_user_id,
        expected_folder_id=validated_message.scope_folder_id,
    )
    if observed.kind is not MsGraphMailMessageChangeKind.ACTIVE:
        raise MsGraphMailMessageChanged() from None
    return observed


def validate_msgraph_mail_participant(value: object) -> MsGraphMailParticipant:
    if isinstance(value, MsGraphMailParticipant):
        source: object = value.model_dump(mode="python")
    elif isinstance(value, dict):
        source = value
    else:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
    try:
        return MsGraphMailParticipant.model_validate(source)
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None


def validate_msgraph_mail_message_content(
    value: object,
    *,
    message: MsGraphMailMessageChange,
    max_chars: int,
) -> MsGraphMailMessageContent:
    if not isinstance(value, MsGraphMailMessageContent):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None

    validated_message = _require_active_message(message)

    try:
        raw = value.model_dump(mode="python")
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None

    try:
        from_participant = (
            validate_msgraph_mail_participant(raw["from_participant"])
            if raw.get("from_participant") is not None
            else None
        )
        sender_participant = (
            validate_msgraph_mail_participant(raw["sender_participant"])
            if raw.get("sender_participant") is not None
            else None
        )
        reply_to = _validate_participant_tuple(raw.get("reply_to", ()))
        to_recipients = _validate_participant_tuple(raw.get("to_recipients", ()))
        cc_recipients = _validate_participant_tuple(raw.get("cc_recipients", ()))
        cc_recipients_bcc = _validate_participant_tuple(raw.get("bcc_recipients", ()))
        body_text = _validate_body_text(raw.get("body_text", ""))
        unique_body_text = _validate_optional_body_text(raw.get("unique_body_text"))
    except (KeyError, ValueError):
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None

    if (
        raw.get("mailbox_user_id") != validated_message.mailbox_user_id
        or raw.get("remote_id") != validated_message.remote_id
        or raw.get("parent_folder_id") != validated_message.parent_folder_id
        or raw.get("content_revision") != validated_message.change_key
    ):
        raise MsGraphMailMessageChanged() from None

    _enforce_content_limit(body_text, unique_body_text, max_chars=max_chars)

    try:
        return _safe_construct_message_content(
            mailbox_user_id=validated_message.mailbox_user_id,
            remote_id=validated_message.remote_id,
            parent_folder_id=validated_message.parent_folder_id,
            content_revision=validated_message.change_key,
            conversation_id=raw.get("conversation_id"),
            internet_message_id=raw.get("internet_message_id"),
            subject=raw.get("subject"),
            body_text=body_text,
            unique_body_text=unique_body_text,
            from_participant=from_participant,
            sender_participant=sender_participant,
            reply_to=reply_to,
            to_recipients=to_recipients,
            cc_recipients=cc_recipients,
            bcc_recipients=cc_recipients_bcc,
        )
    except MsGraphMailContentTooLarge:
        raise
    except ValueError:
        raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None


def _validate_max_chars(max_chars: object) -> int:
    if type(max_chars) is not int:
        raise IntegrationConfigurationError(_INVALID_MAIL_CONTENT_REQUEST) from None
    if max_chars < 1 or max_chars > ABSOLUTE_MAIL_CONTENT_MAX_CHARS:
        raise IntegrationConfigurationError(_INVALID_MAIL_CONTENT_REQUEST) from None
    return max_chars


@runtime_checkable
class MsGraphMailContentReadClient(Protocol):
    def read_mail_message_content(
        self,
        *,
        message: MsGraphMailMessageChange,
        max_chars: int,
    ) -> MsGraphMailMessageContent:
        ...


class MsGraphMailContentReader:
    """Mail message text content and participants reader."""

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
        message: MsGraphMailMessageChange,
        max_chars: int,
    ) -> MsGraphMailMessageContent:
        validated_message = _require_active_message(message)
        validated_max_chars = _validate_max_chars(max_chars)

        quoted_mailbox = quote(validated_message.mailbox_user_id, safe="")
        quoted_message_id = quote(validated_message.remote_id, safe="")
        path = f"/users/{quoted_mailbox}/messages/{quoted_message_id}"
        payload = self._transport.get_initial_json(
            path=path,
            params={"$select": _CONTENT_SELECT},
            headers=_IMMUTABLE_TEXT_BODY_HEADERS,
            not_found_is_dependency=True,
        )

        _compare_message_observation(payload, message=validated_message)

        canonical = parse_msgraph_mail_message_change(
            payload,
            expected_mailbox_user_id=validated_message.mailbox_user_id,
            expected_folder_id=validated_message.scope_folder_id,
        )
        if canonical.kind is not MsGraphMailMessageChangeKind.ACTIVE:
            raise MsGraphMailMessageChanged() from None

        if "body" not in payload:
            raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None
        try:
            body_text = _parse_text_body_field(payload.get("body"))
            unique_body_text = _parse_optional_unique_body(payload)
            from_participant = _parse_optional_participant(payload, "from")
            sender_participant = _parse_optional_participant(payload, "sender")
            reply_to = _parse_participant_list(payload, "replyTo")
            to_recipients = _parse_participant_list(payload, "toRecipients")
            cc_recipients = _parse_participant_list(payload, "ccRecipients")
            bcc_recipients = _parse_participant_list(payload, "bccRecipients")
        except ValueError:
            raise ValueError(_MALFORMED_MAIL_CONTENT_RESPONSE) from None

        _enforce_content_limit(body_text, unique_body_text, max_chars=validated_max_chars)

        return validate_msgraph_mail_message_content(
            _safe_construct_message_content(
                mailbox_user_id=canonical.mailbox_user_id,
                remote_id=canonical.remote_id,
                parent_folder_id=canonical.parent_folder_id,
                content_revision=canonical.change_key,
                conversation_id=canonical.conversation_id,
                internet_message_id=canonical.internet_message_id,
                subject=canonical.subject,
                body_text=body_text,
                unique_body_text=unique_body_text,
                from_participant=from_participant,
                sender_participant=sender_participant,
                reply_to=reply_to,
                to_recipients=to_recipients,
                cc_recipients=cc_recipients,
                bcc_recipients=bcc_recipients,
            ),
            message=validated_message,
            max_chars=validated_max_chars,
        )
