# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Teams Chat knowledge-read: hosted content inventory and bounded bytes."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable
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
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_folders import (
    validate_msgraph_mailbox_user_id,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
    validate_msgraph_teams_chat_hosted_content_id,
    validate_msgraph_teams_chat_id,
    validate_msgraph_teams_chat_message_id,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_messages import (
    MsGraphTeamsChatMessage,
    MsGraphTeamsChatMessageChanged,
    MsGraphTeamsChatMessageState,
    read_and_validate_current_teams_chat_message_observation,
    validate_msgraph_teams_chat_message,
)

DEFAULT_TEAMS_CHAT_HOSTED_CONTENT_MAX_BYTES = 10 * 1024 * 1024
ABSOLUTE_TEAMS_CHAT_HOSTED_CONTENT_MAX_BYTES = 25 * 1024 * 1024

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_HOSTED_CONTENT_RESPONSE = (
    "unexpected Microsoft Graph Teams hosted content response"
)
_INVALID_HOSTED_CONTENT_REQUEST = "invalid Microsoft Graph Teams hosted content request"
_INVALID_HOSTED_CONTENT_CONTINUATION = (
    "invalid Microsoft Graph Teams hosted content continuation"
)
_INVALID_HOSTED_CONTENT_RESPONSE = "Microsoft Graph Teams hosted content response is invalid"
_CONTENT_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONTENT_TYPE_LEN = 1024
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_PREFER_UNKNOWN_ENUM = {"Prefer": "include-unknown-enum-members"}


class MsGraphTeamsChatHostedContentTooLarge(IntegrationConfigurationError):
    def __init__(self) -> None:
        super().__init__(
            "Microsoft Graph Teams hosted content exceeds the configured limit"
        )


class MsGraphTeamsChatHostedContent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    chat_remote_id: str
    message_remote_id: str
    message_revision: str = Field(repr=False)

    remote_id: str

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("chat_remote_id", mode="before")
    @classmethod
    def _validate_chat_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_chat_id(value)

    @field_validator("message_remote_id", mode="before")
    @classmethod
    def _validate_message_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_chat_message_id(value)

    @field_validator("message_revision", mode="before")
    @classmethod
    def _validate_message_revision(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        trimmed = value.strip()
        if not trimmed or _ASCII_CONTROL.search(trimmed):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        if len(trimmed) > 4096:
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return trimmed

    @field_validator("remote_id", mode="before")
    @classmethod
    def _validate_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_chat_hosted_content_id(value)


class MsGraphTeamsChatHostedContentPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    chat_remote_id: str
    message_remote_id: str
    message_revision: str = Field(repr=False)

    items: tuple[MsGraphTeamsChatHostedContent, ...]

    continuation: MsGraphKnowledgeContinuation | None = Field(default=None, repr=False)

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("chat_remote_id", mode="before")
    @classmethod
    def _validate_chat_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_chat_id(value)

    @field_validator("message_remote_id", mode="before")
    @classmethod
    def _validate_message_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_chat_message_id(value)

    @field_validator("message_revision", mode="before")
    @classmethod
    def _validate_message_revision(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        trimmed = value.strip()
        if not trimmed or _ASCII_CONTROL.search(trimmed):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return trimmed

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, value: object) -> tuple[MsGraphTeamsChatHostedContent, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphTeamsChatHostedContent):
                raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return value

    @field_validator("continuation", mode="before")
    @classmethod
    def _validate_continuation(cls, value: object) -> MsGraphKnowledgeContinuation | None:
        if value is None:
            return None
        if not isinstance(value, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        if value.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_page_shape(self) -> MsGraphTeamsChatHostedContentPage:
        remote_ids = [item.remote_id for item in self.items]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        for item in self.items:
            if item.mailbox_user_id != self.mailbox_user_id:
                raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
            if item.chat_remote_id != self.chat_remote_id:
                raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
            if item.message_remote_id != self.message_remote_id:
                raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
            if item.message_revision != self.message_revision:
                raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return self


class MsGraphTeamsChatHostedContentBytes(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    chat_remote_id: str
    message_remote_id: str
    message_revision: str = Field(repr=False)

    hosted_content_remote_id: str

    content_type: str | None = None

    data: bytes = Field(repr=False)
    size_bytes: int
    content_hash: str

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("chat_remote_id", mode="before")
    @classmethod
    def _validate_chat_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_chat_id(value)

    @field_validator("message_remote_id", mode="before")
    @classmethod
    def _validate_message_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_chat_message_id(value)

    @field_validator("hosted_content_remote_id", mode="before")
    @classmethod
    def _validate_hosted_content_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_chat_hosted_content_id(value)

    @field_validator("content_type", mode="before")
    @classmethod
    def _validate_content_type(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        trimmed = value.strip()
        if not trimmed:
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        if _ASCII_CONTROL.search(trimmed):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        if len(trimmed) > _MAX_CONTENT_TYPE_LEN:
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return trimmed

    @field_validator("data", mode="before")
    @classmethod
    def _validate_data(cls, value: object) -> bytes:
        if type(value) is not bytes:
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return value

    @field_validator("content_hash", mode="before")
    @classmethod
    def _validate_content_hash(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        if not _CONTENT_HASH_PATTERN.fullmatch(value):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_bytes_shape(self) -> MsGraphTeamsChatHostedContentBytes:
        if self.size_bytes != len(self.data):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        expected_hash = hashlib.sha256(self.data).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE)
        return self


def _safe_construct_hosted_content(**kwargs: object) -> MsGraphTeamsChatHostedContent:
    try:
        return MsGraphTeamsChatHostedContent(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None


def _safe_construct_hosted_content_page(**kwargs: object) -> MsGraphTeamsChatHostedContentPage:
    try:
        return MsGraphTeamsChatHostedContentPage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None


def _safe_construct_hosted_content_bytes(**kwargs: object) -> MsGraphTeamsChatHostedContentBytes:
    try:
        return MsGraphTeamsChatHostedContentBytes(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None


def _require_active_message(message: object) -> MsGraphTeamsChatMessage:
    validated = validate_msgraph_teams_chat_message(message)
    if validated.state is not MsGraphTeamsChatMessageState.ACTIVE:
        raise MsGraphTeamsChatMessageChanged() from None
    return validated


def parse_msgraph_teams_chat_hosted_content(
    payload: object,
    *,
    message: MsGraphTeamsChatMessage,
) -> MsGraphTeamsChatHostedContent:
    validated_message = _require_active_message(message)
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    if "id" not in payload:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    if "contentBytes" in payload and payload.get("contentBytes") is not None:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    if "contentType" in payload and payload.get("contentType") is not None:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    try:
        remote_id = validate_msgraph_teams_chat_hosted_content_id(payload.get("id"))
    except ValueError:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    return _safe_construct_hosted_content(
        mailbox_user_id=validated_message.mailbox_user_id,
        chat_remote_id=validated_message.chat_remote_id,
        message_remote_id=validated_message.remote_id,
        message_revision=validated_message.revision,
        remote_id=remote_id,
    )


def validate_msgraph_teams_chat_hosted_content(
    value: object,
) -> MsGraphTeamsChatHostedContent:
    if not isinstance(value, MsGraphTeamsChatHostedContent):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    try:
        return MsGraphTeamsChatHostedContent.model_validate(value.model_dump(mode="python"))
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None


def validate_msgraph_teams_chat_hosted_content_page(
    value: object,
    *,
    message: MsGraphTeamsChatMessage,
    graph_base_url: str,
) -> MsGraphTeamsChatHostedContentPage:
    if not isinstance(value, MsGraphTeamsChatHostedContentPage):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    validated_message = _require_active_message(message)

    try:
        raw_mailbox = value.mailbox_user_id
        raw_chat_id = value.chat_remote_id
        raw_message_id = value.message_remote_id
        raw_revision = value.message_revision
        raw_items = value.items
        raw_continuation = value.continuation
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    if raw_mailbox != validated_message.mailbox_user_id:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    if raw_chat_id != validated_message.chat_remote_id:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    if raw_message_id != validated_message.remote_id:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
    if raw_revision != validated_message.revision:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    if type(raw_items) is not tuple:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    validated_items: list[MsGraphTeamsChatHostedContent] = []
    for item in raw_items:
        if not isinstance(item, MsGraphTeamsChatHostedContent):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
        validated_item = validate_msgraph_teams_chat_hosted_content(item)
        if (
            validated_item.mailbox_user_id != validated_message.mailbox_user_id
            or validated_item.chat_remote_id != validated_message.chat_remote_id
            or validated_item.message_remote_id != validated_message.remote_id
            or validated_item.message_revision != validated_message.revision
        ):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
        validated_items.append(validated_item)

    remote_ids = [item.remote_id for item in validated_items]
    if len(remote_ids) != len(set(remote_ids)):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    validated_continuation: MsGraphKnowledgeContinuation | None = None
    if raw_continuation is not None:
        if not isinstance(raw_continuation, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None
        try:
            validated_continuation = validate_msgraph_teams_chat_hosted_contents_continuation(
                raw_continuation,
                mailbox_user_id=validated_message.mailbox_user_id,
                chat_id=validated_message.chat_remote_id,
                message_id=validated_message.remote_id,
                graph_base_url=graph_base_url,
            )
        except IntegrationConfigurationError:
            raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    try:
        return MsGraphTeamsChatHostedContentPage(
            mailbox_user_id=validated_message.mailbox_user_id,
            chat_remote_id=validated_message.chat_remote_id,
            message_remote_id=validated_message.remote_id,
            message_revision=validated_message.revision,
            items=tuple(validated_items),
            continuation=validated_continuation,
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None


def validate_msgraph_teams_chat_hosted_content_bytes(
    value: object,
    *,
    message: MsGraphTeamsChatMessage,
    hosted_content: MsGraphTeamsChatHostedContent,
    max_bytes: int,
) -> MsGraphTeamsChatHostedContentBytes:
    if type(max_bytes) is not int or max_bytes < 1 or max_bytes > ABSOLUTE_TEAMS_CHAT_HOSTED_CONTENT_MAX_BYTES:
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    validated_message = _require_active_message(message)
    validated_hosted = validate_msgraph_teams_chat_hosted_content(hosted_content)

    if not isinstance(value, MsGraphTeamsChatHostedContentBytes):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    if (
        validated_hosted.mailbox_user_id != validated_message.mailbox_user_id
        or validated_hosted.chat_remote_id != validated_message.chat_remote_id
        or validated_hosted.message_remote_id != validated_message.remote_id
        or validated_hosted.message_revision != validated_message.revision
    ):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    try:
        reconstructed = MsGraphTeamsChatHostedContentBytes.model_validate(
            value.model_dump(mode="python")
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    if (
        reconstructed.mailbox_user_id != validated_message.mailbox_user_id
        or reconstructed.chat_remote_id != validated_message.chat_remote_id
        or reconstructed.message_remote_id != validated_message.remote_id
        or reconstructed.message_revision != validated_message.revision
        or reconstructed.hosted_content_remote_id != validated_hosted.remote_id
    ):
        raise ValueError(_MALFORMED_HOSTED_CONTENT_RESPONSE) from None

    if reconstructed.size_bytes > max_bytes:
        raise MsGraphTeamsChatHostedContentTooLarge() from None

    return reconstructed


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


def _extract_hosted_contents_path(
    path: str,
    *,
    graph_base_path: str,
) -> tuple[str, str, str] | None:
    normalized = path.rstrip("/") or "/"
    base = graph_base_path.rstrip("/") or "/"

    patterns: list[tuple[str, bool, bool, bool]] = [
        (
            rf"^{re.escape(base)}/users/([^/]+)/chats/([^/]+)/messages/([^/]+)/hostedContents$",
            False,
            False,
            False,
        ),
        (
            rf"^{re.escape(base)}/users\('((?:[^']|'')*)'\)/chats\('((?:[^']|'')*)'\)/messages\('((?:[^']|'')*)'\)/hostedContents$",
            True,
            True,
            True,
        ),
        (
            rf"^{re.escape(base)}/users/([^/]+)/chats\('((?:[^']|'')*)'\)/messages/([^/]+)/hostedContents$",
            False,
            True,
            False,
        ),
        (
            rf"^{re.escape(base)}/users\('((?:[^']|'')*)'\)/chats/([^/]+)/messages/([^/]+)/hostedContents$",
            True,
            False,
            False,
        ),
        (
            rf"^{re.escape(base)}/users/([^/]+)/chats/([^/]+)/messages\('((?:[^']|'')*)'\)/hostedContents$",
            False,
            False,
            True,
        ),
        (
            rf"^{re.escape(base)}/users\('((?:[^']|'')*)'\)/chats\('((?:[^']|'')*)'\)/messages/([^/]+)/hostedContents$",
            True,
            True,
            False,
        ),
        (
            rf"^{re.escape(base)}/users/([^/]+)/chats\('((?:[^']|'')*)'\)/messages\('((?:[^']|'')*)'\)/hostedContents$",
            False,
            True,
            True,
        ),
        (
            rf"^{re.escape(base)}/users\('((?:[^']|'')*)'\)/chats/([^/]+)/messages\('((?:[^']|'')*)'\)/hostedContents$",
            True,
            False,
            True,
        ),
    ]

    for pattern, mailbox_odata, chat_odata, message_odata in patterns:
        match = re.fullmatch(pattern, normalized, re.IGNORECASE)
        if match is None:
            continue
        mailbox_segment = match.group(1)
        chat_segment = match.group(2)
        message_segment = match.group(3)
        if not mailbox_segment or not chat_segment or not message_segment:
            return None
        return (
            _decode_path_segment(mailbox_segment, odata_literal=mailbox_odata),
            _decode_path_segment(chat_segment, odata_literal=chat_odata),
            _decode_path_segment(message_segment, odata_literal=message_odata),
        )
    return None


def validate_msgraph_teams_chat_hosted_contents_continuation(
    continuation: object,
    *,
    mailbox_user_id: str,
    chat_id: str,
    message_id: str,
    graph_base_url: str,
) -> MsGraphKnowledgeContinuation:
    if not isinstance(continuation, MsGraphKnowledgeContinuation):
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None

    try:
        revalidated = MsGraphKnowledgeContinuation.model_validate(
            continuation.model_dump(mode="python")
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None

    if revalidated.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None

    try:
        validated_url = validate_msgraph_continuation_url(
            revalidated.url,
            graph_base_url=graph_base_url,
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None

    parsed = urlparse(validated_url)
    extracted = _extract_hosted_contents_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted is None:
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None

    extracted_mailbox, extracted_chat, extracted_message = extracted
    try:
        validated_mailbox = validate_msgraph_mailbox_user_id(mailbox_user_id)
        validated_chat = validate_msgraph_teams_chat_id(chat_id)
        validated_message = validate_msgraph_teams_chat_message_id(message_id)
        validated_extracted_mailbox = validate_msgraph_mailbox_user_id(extracted_mailbox)
        validated_extracted_chat = validate_msgraph_teams_chat_id(extracted_chat)
        validated_extracted_message = validate_msgraph_teams_chat_message_id(extracted_message)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None

    if (
        validated_extracted_mailbox != validated_mailbox
        or validated_extracted_chat != validated_chat
        or validated_extracted_message != validated_message
    ):
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_CONTINUATION) from None

    return revalidated


def _response_status_code(response: object) -> int:
    try:
        status_code = response.status_code
    except AttributeError:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if type(status_code) is not int:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if status_code < 100 or status_code > 599:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    return status_code


def _response_headers(response: object) -> Mapping[str, str]:
    try:
        headers = response.headers
    except AttributeError:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if not isinstance(headers, Mapping):
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    return headers


def _parse_content_length(headers: Mapping[str, str]) -> int | None:
    raw_value: str | None = None
    try:
        header_items = headers.items()
    except Exception:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    try:
        for key, value in header_items:
            if not isinstance(key, str):
                raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
            if key.lower() == "content-length":
                if raw_value is not None:
                    raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
                if not isinstance(value, str):
                    raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
                raw_value = value
    except IntegrationDependencyError:
        raise
    except Exception:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if raw_value is None:
        return None
    trimmed = raw_value.strip()
    if not trimmed or not trimmed.isdigit():
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    parsed = int(trimmed)
    if parsed < 0:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    return parsed


def _parse_content_type_header(headers: Mapping[str, str]) -> str | None:
    raw_value: str | None = None
    try:
        header_items = headers.items()
    except Exception:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    try:
        for key, value in header_items:
            if not isinstance(key, str):
                raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
            if key.lower() == "content-type":
                if raw_value is not None:
                    raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
                if not isinstance(value, str):
                    raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
                raw_value = value
    except IntegrationDependencyError:
        raise
    except Exception:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if raw_value is None:
        return None
    trimmed = raw_value.strip()
    if not trimmed:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if _ASCII_CONTROL.search(trimmed):
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if len(trimmed) > _MAX_CONTENT_TYPE_LEN:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    return trimmed


def _raise_for_hosted_content_download_response(response: object) -> None:
    status_code = _response_status_code(response)
    if status_code == 200:
        return
    if status_code == 206:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if 300 <= status_code <= 399:
        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
    if status_code in {400, 401, 403}:
        raise IntegrationConfigurationError("Microsoft Graph knowledge configuration failure") from None
    if status_code == 404:
        raise IntegrationDependencyError("Microsoft Graph knowledge dependency failure") from None
    if status_code in {408, 410, 429} or status_code >= 500:
        raise IntegrationDependencyError("Microsoft Graph knowledge dependency failure") from None
    raise IntegrationDependencyError("Microsoft Graph knowledge dependency failure") from None


def _execute_transport(transport_fn: Any) -> object:
    try:
        return transport_fn()
    except (
        IntegrationConfigurationError,
        IntegrationDependencyError,
        MsGraphTeamsChatMessageChanged,
        MsGraphTeamsChatHostedContentTooLarge,
    ):
        raise
    except Exception:
        raise IntegrationDependencyError(
            "Microsoft Graph knowledge dependency is unavailable"
        ) from None


def _validate_max_bytes(max_bytes: object) -> int:
    if type(max_bytes) is not int:
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_REQUEST)
    if max_bytes < 1 or max_bytes > ABSOLUTE_TEAMS_CHAT_HOSTED_CONTENT_MAX_BYTES:
        raise IntegrationConfigurationError(_INVALID_HOSTED_CONTENT_REQUEST)
    return max_bytes


@runtime_checkable
class MsGraphTeamsChatHostedContentReadClient(Protocol):
    def read_teams_chat_hosted_contents_page(
        self,
        *,
        message: MsGraphTeamsChatMessage,
        continuation: MsGraphKnowledgeContinuation | None,
    ) -> MsGraphTeamsChatHostedContentPage:
        ...

    def read_teams_chat_hosted_content_bytes(
        self,
        *,
        message: MsGraphTeamsChatMessage,
        hosted_content: MsGraphTeamsChatHostedContent,
        max_bytes: int,
    ) -> MsGraphTeamsChatHostedContentBytes:
        ...


class MsGraphTeamsChatHostedContentReader:
    """Teams-hosted content inventory and bounded byte download."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        transport: MsGraphKnowledgeTransport,
        graph_http_client: Any,
    ) -> None:
        self._config = config
        self._transport = transport
        self._graph_http_client = graph_http_client

    def read_hosted_contents_page(
        self,
        *,
        message: MsGraphTeamsChatMessage,
        continuation: MsGraphKnowledgeContinuation | None,
    ) -> MsGraphTeamsChatHostedContentPage:
        validated_message = _require_active_message(message)
        read_and_validate_current_teams_chat_message_observation(
            message=validated_message,
            transport=self._transport,
        )

        if continuation is None:
            quoted_mailbox = quote(validated_message.mailbox_user_id, safe="")
            quoted_chat = quote(validated_message.chat_remote_id, safe="")
            quoted_message = quote(validated_message.remote_id, safe="")
            path = (
                f"/users/{quoted_mailbox}/chats/{quoted_chat}/messages/"
                f"{quoted_message}/hostedContents"
            )
            payload = self._transport.get_initial_json(
                path=path,
                headers=_PREFER_UNKNOWN_ENUM,
            )
        else:
            validated_continuation = validate_msgraph_teams_chat_hosted_contents_continuation(
                continuation,
                mailbox_user_id=validated_message.mailbox_user_id,
                chat_id=validated_message.chat_remote_id,
                message_id=validated_message.remote_id,
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
        parsed_items = tuple(
            parse_msgraph_teams_chat_hosted_content(raw_item, message=validated_message)
            for raw_item in collection_page.items
        )
        page = validate_msgraph_teams_chat_hosted_content_page(
            _safe_construct_hosted_content_page(
                mailbox_user_id=validated_message.mailbox_user_id,
                chat_remote_id=validated_message.chat_remote_id,
                message_remote_id=validated_message.remote_id,
                message_revision=validated_message.revision,
                items=parsed_items,
                continuation=collection_page.continuation,
            ),
            message=validated_message,
            graph_base_url=self._config.graph_base_url,
        )

        read_and_validate_current_teams_chat_message_observation(
            message=validated_message,
            transport=self._transport,
        )
        return page

    def read_hosted_content_bytes(
        self,
        *,
        message: MsGraphTeamsChatMessage,
        hosted_content: MsGraphTeamsChatHostedContent,
        max_bytes: int,
    ) -> MsGraphTeamsChatHostedContentBytes:
        validated_message = _require_active_message(message)
        validated_hosted = validate_msgraph_teams_chat_hosted_content(hosted_content)
        validated_max_bytes = _validate_max_bytes(max_bytes)

        if (
            validated_hosted.mailbox_user_id != validated_message.mailbox_user_id
            or validated_hosted.chat_remote_id != validated_message.chat_remote_id
            or validated_hosted.message_remote_id != validated_message.remote_id
            or validated_hosted.message_revision != validated_message.revision
        ):
            raise MsGraphTeamsChatMessageChanged() from None

        read_and_validate_current_teams_chat_message_observation(
            message=validated_message,
            transport=self._transport,
        )

        quoted_mailbox = quote(validated_message.mailbox_user_id, safe="")
        quoted_chat = quote(validated_message.chat_remote_id, safe="")
        quoted_message = quote(validated_message.remote_id, safe="")
        quoted_hosted = quote(validated_hosted.remote_id, safe="")
        path = (
            f"/users/{quoted_mailbox}/chats/{quoted_chat}/messages/{quoted_message}"
            f"/hostedContents/{quoted_hosted}/$value"
        )

        def _do_stream() -> object:
            return self._graph_http_client.stream(
                "GET",
                path,
                headers={"Accept": "application/octet-stream"},
                follow_redirects=False,
            )

        try:
            stream_context = _execute_transport(_do_stream)
        except Exception:
            raise IntegrationDependencyError(
                "Microsoft Graph knowledge dependency is unavailable"
            ) from None

        try:
            with stream_context as response:
                _raise_for_hosted_content_download_response(response)
                headers = _response_headers(response)
                content_length = _parse_content_length(headers)
                content_type = _parse_content_type_header(headers)
                if content_length is not None and content_length > validated_max_bytes:
                    raise MsGraphTeamsChatHostedContentTooLarge() from None

                try:
                    iter_bytes = response.iter_bytes
                except AttributeError:
                    raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
                if not callable(iter_bytes):
                    raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None

                buffer = bytearray()
                for chunk in iter_bytes():
                    if type(chunk) is not bytes:
                        raise IntegrationDependencyError(_INVALID_HOSTED_CONTENT_RESPONSE) from None
                    buffer.extend(chunk)
                    if len(buffer) > validated_max_bytes:
                        raise MsGraphTeamsChatHostedContentTooLarge() from None

                data = bytes(buffer)
                if content_length is not None and len(data) != content_length:
                    raise MsGraphTeamsChatMessageChanged() from None
        except (
            IntegrationConfigurationError,
            IntegrationDependencyError,
            MsGraphTeamsChatMessageChanged,
            MsGraphTeamsChatHostedContentTooLarge,
        ):
            raise
        except Exception:
            raise IntegrationDependencyError(
                "Microsoft Graph knowledge dependency is unavailable"
            ) from None

        read_and_validate_current_teams_chat_message_observation(
            message=validated_message,
            transport=self._transport,
        )

        content_hash = hashlib.sha256(data).hexdigest()
        return validate_msgraph_teams_chat_hosted_content_bytes(
            _safe_construct_hosted_content_bytes(
                mailbox_user_id=validated_message.mailbox_user_id,
                chat_remote_id=validated_message.chat_remote_id,
                message_remote_id=validated_message.remote_id,
                message_revision=validated_message.revision,
                hosted_content_remote_id=validated_hosted.remote_id,
                content_type=content_type,
                data=data,
                size_bytes=len(data),
                content_hash=content_hash,
            ),
            message=validated_message,
            hosted_content=validated_hosted,
            max_bytes=validated_max_bytes,
        )
