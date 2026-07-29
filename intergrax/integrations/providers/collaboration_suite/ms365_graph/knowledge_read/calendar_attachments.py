# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Calendar knowledge-read: attachment inventory and bounded file content."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping, Protocol, runtime_checkable
from urllib.parse import quote, unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_content import (
    MsGraphCalendarEventChanged,
    read_and_validate_current_calendar_event_observation,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_events import (
    MsGraphCalendarEventChange,
    MsGraphCalendarEventChangeKind,
    validate_msgraph_calendar_event_change,
    validate_msgraph_calendar_event_id,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_inventory import (
    validate_msgraph_calendar_attachment_id,
    validate_msgraph_calendar_id,
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

DEFAULT_CALENDAR_ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024
ABSOLUTE_CALENDAR_ATTACHMENT_MAX_BYTES = 25 * 1024 * 1024

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE = "unexpected Microsoft Graph Calendar attachments response"
_INVALID_CALENDAR_ATTACHMENTS_REQUEST = "invalid Microsoft Graph Calendar attachments request"
_INVALID_CALENDAR_ATTACHMENTS_CONTINUATION = "invalid Microsoft Graph Calendar attachments continuation"
_UNSUPPORTED_ATTACHMENT_CONTENT = (
    "Microsoft Graph Calendar attachment content is not supported for this attachment type"
)
_INVALID_ATTACHMENT_RESPONSE = "Microsoft Graph Calendar attachment response is invalid"
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_CONTENT_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_ATTACHMENT_NAME_LEN = 4096
_MAX_CONTENT_TYPE_LEN = 1024
_MAX_CONTENT_ID_LEN = 4096
_MAX_ODATA_TYPE_LEN = 1024
_MIN_PAGE_LIMIT = 1
_MAX_PAGE_LIMIT = 200

_ODATA_FILE_ATTACHMENT = "#microsoft.graph.fileAttachment"
_ODATA_ITEM_ATTACHMENT = "#microsoft.graph.itemAttachment"
_ODATA_REFERENCE_ATTACHMENT = "#microsoft.graph.referenceAttachment"

_ATTACHMENTS_SELECT = (
    "id,name,contentType,size,isInline,contentId,lastModifiedDateTime"
)


class MsGraphCalendarAttachmentTooLarge(IntegrationConfigurationError):
    """Calendar attachment exceeds the configured byte limit."""

    def __init__(self) -> None:
        super().__init__("Microsoft Graph Calendar attachment exceeds the configured content limit")


class MsGraphCalendarAttachmentKind(StrEnum):
    FILE = "file"
    ITEM = "item"
    REFERENCE = "reference"
    UNKNOWN = "unknown"


def _validate_attachment_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    if "\x00" in value:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    if len(value) > _MAX_ATTACHMENT_NAME_LEN:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    return value


def _validate_optional_trimmed_string(value: object, *, max_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        return None
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    if len(trimmed) > max_length:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    return trimmed


def _validate_present_optional_string(value: object, *, max_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    if len(trimmed) > max_length:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    return trimmed


def _validate_model_optional_string(value: object, *, max_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    if len(trimmed) > max_length:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    return trimmed


def _validate_content_revision(value: object) -> str:
    result = _validate_optional_trimmed_string(value, max_length=2048)
    if result is None:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    return result


def _normalize_model_datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    return value.astimezone(timezone.utc)


def _parse_timezone_aware_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    if trimmed.endswith("Z"):
        trimmed = f"{trimmed[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(trimmed)
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    return parsed.astimezone(timezone.utc)


def _validate_unknown_odata_type(value: str) -> None:
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    if len(trimmed) > _MAX_ODATA_TYPE_LEN:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)


def _map_attachment_kind(odata_type: object) -> MsGraphCalendarAttachmentKind:
    if not isinstance(odata_type, str):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    trimmed = odata_type.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    if trimmed == _ODATA_FILE_ATTACHMENT:
        return MsGraphCalendarAttachmentKind.FILE
    if trimmed == _ODATA_ITEM_ATTACHMENT:
        return MsGraphCalendarAttachmentKind.ITEM
    if trimmed == _ODATA_REFERENCE_ATTACHMENT:
        return MsGraphCalendarAttachmentKind.REFERENCE
    _validate_unknown_odata_type(trimmed)
    return MsGraphCalendarAttachmentKind.UNKNOWN


def _require_active_event(event: object) -> MsGraphCalendarEventChange:
    validated = validate_msgraph_calendar_event_change(event)
    if validated.kind is not MsGraphCalendarEventChangeKind.ACTIVE:
        raise MsGraphCalendarEventChanged() from None
    return validated


class MsGraphCalendarAttachment(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    calendar_remote_id: str
    event_remote_id: str
    event_revision: str = Field(repr=False)

    remote_id: str
    kind: MsGraphCalendarAttachmentKind

    name: str = Field(repr=False)
    content_type: str | None = Field(default=None, repr=False)

    size_bytes: int
    is_inline: bool
    content_id: str | None = Field(default=None, repr=False)
    last_modified_at: datetime

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("calendar_remote_id", mode="before")
    @classmethod
    def _validate_calendar_remote_id(cls, value: object) -> str:
        return validate_msgraph_calendar_id(value)

    @field_validator("event_revision", mode="before")
    @classmethod
    def _validate_event_revision(cls, value: object) -> str:
        return _validate_content_revision(value)

    @field_validator("remote_id", mode="before")
    @classmethod
    def _validate_remote_id(cls, value: object) -> str:
        return validate_msgraph_calendar_attachment_id(value)

    @field_validator("event_remote_id", mode="before")
    @classmethod
    def _validate_event_remote_id(cls, value: object) -> str:
        return validate_msgraph_calendar_event_id(value)

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value: object) -> str:
        return _validate_attachment_name(value)

    @field_validator("content_type", "content_id", mode="before")
    @classmethod
    def _validate_optional_strings(cls, value: object, info: Any) -> str | None:
        max_length = _MAX_CONTENT_TYPE_LEN if info.field_name == "content_type" else _MAX_CONTENT_ID_LEN
        return _validate_model_optional_string(value, max_length=max_length)

    @field_validator("size_bytes", mode="before")
    @classmethod
    def _validate_size_bytes(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        if value < 0:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        return value

    @field_validator("is_inline", mode="before")
    @classmethod
    def _validate_is_inline(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        return value

    @field_validator("last_modified_at", mode="before")
    @classmethod
    def _validate_last_modified_at(cls, value: object) -> datetime:
        return _normalize_model_datetime(value)

    @property
    def binary_content_supported(self) -> bool:
        return self.kind is MsGraphCalendarAttachmentKind.FILE


class MsGraphCalendarAttachmentPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    calendar_remote_id: str
    event_remote_id: str
    event_revision: str = Field(repr=False)

    items: tuple[MsGraphCalendarAttachment, ...]

    continuation: MsGraphKnowledgeContinuation | None = Field(
        default=None,
        repr=False,
    )

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, value: object) -> tuple[MsGraphCalendarAttachment, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        return value

    @field_validator("continuation", mode="before")
    @classmethod
    def _validate_continuation(cls, value: object) -> MsGraphKnowledgeContinuation | None:
        if value is None:
            return None
        if not isinstance(value, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        if value.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        return value

    @property
    def has_more(self) -> bool:
        return self.continuation is not None


class MsGraphCalendarFileAttachmentContent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    calendar_remote_id: str
    event_remote_id: str
    event_revision: str = Field(repr=False)

    attachment_remote_id: str
    name: str = Field(repr=False)
    content_type: str | None = Field(default=None, repr=False)

    is_inline: bool
    content_id: str | None = Field(default=None, repr=False)

    data: bytes = Field(repr=False)
    size_bytes: int
    content_hash: str

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("calendar_remote_id", mode="before")
    @classmethod
    def _validate_calendar_remote_id(cls, value: object) -> str:
        return validate_msgraph_calendar_id(value)

    @field_validator("event_remote_id", mode="before")
    @classmethod
    def _validate_event_remote_id(cls, value: object) -> str:
        return validate_msgraph_calendar_event_id(value)

    @field_validator("event_revision", mode="before")
    @classmethod
    def _validate_event_revision(cls, value: object) -> str:
        return _validate_content_revision(value)

    @field_validator("attachment_remote_id", mode="before")
    @classmethod
    def _validate_attachment_remote_id_field(cls, value: object) -> str:
        return validate_msgraph_calendar_attachment_id(value)

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value: object) -> str:
        return _validate_attachment_name(value)

    @field_validator("content_type", "content_id", mode="before")
    @classmethod
    def _validate_optional_strings(cls, value: object, info: Any) -> str | None:
        max_length = _MAX_CONTENT_TYPE_LEN if info.field_name == "content_type" else _MAX_CONTENT_ID_LEN
        return _validate_model_optional_string(value, max_length=max_length)

    @field_validator("is_inline", mode="before")
    @classmethod
    def _validate_is_inline(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        return value

    @field_validator("data", mode="before")
    @classmethod
    def _validate_data(cls, value: object) -> bytes:
        if type(value) is not bytes:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        return value

    @field_validator("size_bytes", mode="before")
    @classmethod
    def _validate_size_bytes(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        if value < 0:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        return value

    @field_validator("content_hash", mode="before")
    @classmethod
    def _validate_content_hash_format(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        if not _CONTENT_HASH_PATTERN.match(value):
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_content_shape(self) -> MsGraphCalendarFileAttachmentContent:
        if self.size_bytes != len(self.data):
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        expected_hash = hashlib.sha256(self.data).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        return self


def _safe_construct_attachment(**kwargs: object) -> MsGraphCalendarAttachment:
    try:
        return MsGraphCalendarAttachment(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None


def _safe_construct_attachment_page(**kwargs: object) -> MsGraphCalendarAttachmentPage:
    try:
        return MsGraphCalendarAttachmentPage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None


def _safe_construct_file_attachment_content(**kwargs: object) -> MsGraphCalendarFileAttachmentContent:
    try:
        return MsGraphCalendarFileAttachmentContent(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None


def parse_msgraph_calendar_attachment(
    payload: object,
    *,
    event: MsGraphCalendarEventChange,
) -> MsGraphCalendarAttachment:
    validated_event = _require_active_event(event)
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None

    required_keys = ("@odata.type", "id", "name", "size", "isInline", "lastModifiedDateTime")
    for key in required_keys:
        if key not in payload:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None

    try:
        kind = _map_attachment_kind(payload.get("@odata.type"))
        remote_id = validate_msgraph_calendar_attachment_id(payload.get("id"))
        name = _validate_attachment_name(payload.get("name"))
        size_bytes = payload.get("size")
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        is_inline = payload.get("isInline")
        if type(is_inline) is not bool:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        last_modified_at = _parse_timezone_aware_datetime(payload.get("lastModifiedDateTime"))

        if "contentType" not in payload:
            content_type = None
        elif payload.get("contentType") is None:
            content_type = None
        else:
            content_type = _validate_present_optional_string(
                payload.get("contentType"),
                max_length=_MAX_CONTENT_TYPE_LEN,
            )

        if "contentId" not in payload:
            content_id = None
        elif payload.get("contentId") is None:
            content_id = None
        else:
            content_id = _validate_present_optional_string(
                payload.get("contentId"),
                max_length=_MAX_CONTENT_ID_LEN,
            )
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None

    return _safe_construct_attachment(
        mailbox_user_id=validated_event.mailbox_user_id,
        calendar_remote_id=validated_event.calendar_remote_id,
        event_remote_id=validated_event.remote_id,
        event_revision=validated_event.change_key,
        remote_id=remote_id,
        kind=kind,
        name=name,
        content_type=content_type,
        size_bytes=size_bytes,
        is_inline=is_inline,
        content_id=content_id,
        last_modified_at=last_modified_at,
    )


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


def _extract_calendar_event_attachments_path(
    path: str,
    *,
    graph_base_path: str,
) -> tuple[str, str, str] | None:
    normalized = path.rstrip("/") or "/"
    base = graph_base_path.rstrip("/") or "/"

    slash_match = re.fullmatch(
        rf"^{re.escape(base)}/users/([^/]+)/calendars/([^/]+)/events/([^/]+)/attachments$",
        normalized,
        re.IGNORECASE,
    )
    if slash_match is not None:
        mailbox_segment = slash_match.group(1)
        calendar_segment = slash_match.group(2)
        event_segment = slash_match.group(3)
        if not mailbox_segment or not calendar_segment or not event_segment:
            return None
        return (
            unquote(mailbox_segment),
            unquote(calendar_segment),
            unquote(event_segment),
        )

    odata_match = re.fullmatch(
        rf"^{re.escape(base)}/users\('((?:[^']|'')*)'\)/calendars\('((?:[^']|'')*)'\)/events\('((?:[^']|'')*)'\)/attachments$",
        normalized,
        re.IGNORECASE,
    )
    if odata_match is not None:
        mailbox_literal = odata_match.group(1)
        calendar_literal = odata_match.group(2)
        event_literal = odata_match.group(3)
        if not mailbox_literal or not calendar_literal or not event_literal:
            return None
        return (
            _decode_path_segment(mailbox_literal, odata_literal=True),
            _decode_path_segment(calendar_literal, odata_literal=True),
            _decode_path_segment(event_literal, odata_literal=True),
        )

    odata_calendar_match = re.fullmatch(
        rf"^{re.escape(base)}/users/([^/]+)/calendars\('((?:[^']|'')*)'\)/events/([^/]+)/attachments$",
        normalized,
        re.IGNORECASE,
    )
    if odata_calendar_match is not None:
        mailbox_segment = odata_calendar_match.group(1)
        calendar_literal = odata_calendar_match.group(2)
        event_segment = odata_calendar_match.group(3)
        if not mailbox_segment or not calendar_literal or not event_segment:
            return None
        return (
            unquote(mailbox_segment),
            _decode_path_segment(calendar_literal, odata_literal=True),
            unquote(event_segment),
        )

    odata_event_match = re.fullmatch(
        rf"^{re.escape(base)}/users/([^/]+)/calendars/([^/]+)/events\('((?:[^']|'')*)'\)/attachments$",
        normalized,
        re.IGNORECASE,
    )
    if odata_event_match is not None:
        mailbox_segment = odata_event_match.group(1)
        calendar_segment = odata_event_match.group(2)
        event_literal = odata_event_match.group(3)
        if not mailbox_segment or not calendar_segment or not event_literal:
            return None
        return (
            unquote(mailbox_segment),
            unquote(calendar_segment),
            _decode_path_segment(event_literal, odata_literal=True),
        )

    odata_mailbox_match = re.fullmatch(
        rf"^{re.escape(base)}/users\('((?:[^']|'')*)'\)/calendars/([^/]+)/events/([^/]+)/attachments$",
        normalized,
        re.IGNORECASE,
    )
    if odata_mailbox_match is not None:
        mailbox_literal = odata_mailbox_match.group(1)
        calendar_segment = odata_mailbox_match.group(2)
        event_segment = odata_mailbox_match.group(3)
        if not mailbox_literal or not calendar_segment or not event_segment:
            return None
        return (
            _decode_path_segment(mailbox_literal, odata_literal=True),
            unquote(calendar_segment),
            unquote(event_segment),
        )

    odata_mailbox_calendar_match = re.fullmatch(
        rf"^{re.escape(base)}/users\('((?:[^']|'')*)'\)/calendars\('((?:[^']|'')*)'\)/events/([^/]+)/attachments$",
        normalized,
        re.IGNORECASE,
    )
    if odata_mailbox_calendar_match is not None:
        mailbox_literal = odata_mailbox_calendar_match.group(1)
        calendar_literal = odata_mailbox_calendar_match.group(2)
        event_segment = odata_mailbox_calendar_match.group(3)
        if not mailbox_literal or not calendar_literal or not event_segment:
            return None
        return (
            _decode_path_segment(mailbox_literal, odata_literal=True),
            _decode_path_segment(calendar_literal, odata_literal=True),
            unquote(event_segment),
        )

    odata_calendar_event_match = re.fullmatch(
        rf"^{re.escape(base)}/users/([^/]+)/calendars\('((?:[^']|'')*)'\)/events\('((?:[^']|'')*)'\)/attachments$",
        normalized,
        re.IGNORECASE,
    )
    if odata_calendar_event_match is not None:
        mailbox_segment = odata_calendar_event_match.group(1)
        calendar_literal = odata_calendar_event_match.group(2)
        event_literal = odata_calendar_event_match.group(3)
        if not mailbox_segment or not calendar_literal or not event_literal:
            return None
        return (
            unquote(mailbox_segment),
            _decode_path_segment(calendar_literal, odata_literal=True),
            _decode_path_segment(event_literal, odata_literal=True),
        )

    odata_mailbox_event_match = re.fullmatch(
        rf"^{re.escape(base)}/users\('((?:[^']|'')*)'\)/calendars/([^/]+)/events\('((?:[^']|'')*)'\)/attachments$",
        normalized,
        re.IGNORECASE,
    )
    if odata_mailbox_event_match is not None:
        mailbox_literal = odata_mailbox_event_match.group(1)
        calendar_segment = odata_mailbox_event_match.group(2)
        event_literal = odata_mailbox_event_match.group(3)
        if not mailbox_literal or not calendar_segment or not event_literal:
            return None
        return (
            _decode_path_segment(mailbox_literal, odata_literal=True),
            unquote(calendar_segment),
            _decode_path_segment(event_literal, odata_literal=True),
        )

    return None


def validate_msgraph_calendar_attachments_continuation(
    continuation: object,
    *,
    mailbox_user_id: str,
    calendar_id: str,
    event_id: str,
    graph_base_url: str,
) -> MsGraphKnowledgeContinuation:
    if not isinstance(continuation, MsGraphKnowledgeContinuation):
        raise IntegrationConfigurationError(_INVALID_CALENDAR_ATTACHMENTS_CONTINUATION) from None

    try:
        revalidated = MsGraphKnowledgeContinuation.model_validate(
            continuation.model_dump(mode="python")
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise IntegrationConfigurationError(_INVALID_CALENDAR_ATTACHMENTS_CONTINUATION) from None

    if revalidated.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_ATTACHMENTS_CONTINUATION) from None

    try:
        validated_url = validate_msgraph_continuation_url(
            revalidated.url,
            graph_base_url=graph_base_url,
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_ATTACHMENTS_CONTINUATION) from None

    parsed = urlparse(validated_url)
    extracted = _extract_calendar_event_attachments_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted is None:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_ATTACHMENTS_CONTINUATION) from None

    extracted_mailbox_user_id, extracted_calendar_id, extracted_event_id = extracted
    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(mailbox_user_id)
        validated_calendar_id = validate_msgraph_calendar_id(calendar_id)
        validated_event_id = validate_msgraph_calendar_event_id(event_id)
        validated_extracted_mailbox_user_id = validate_msgraph_mailbox_user_id(
            extracted_mailbox_user_id
        )
        validated_extracted_calendar_id = validate_msgraph_calendar_id(extracted_calendar_id)
        validated_extracted_event_id = validate_msgraph_calendar_event_id(extracted_event_id)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_ATTACHMENTS_CONTINUATION) from None

    if (
        validated_extracted_mailbox_user_id != validated_mailbox_user_id
        or validated_extracted_calendar_id != validated_calendar_id
        or validated_extracted_event_id != validated_event_id
    ):
        raise IntegrationConfigurationError(_INVALID_CALENDAR_ATTACHMENTS_CONTINUATION) from None

    return revalidated


def validate_msgraph_calendar_attachment(value: object) -> MsGraphCalendarAttachment:
    if not isinstance(value, MsGraphCalendarAttachment):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None
    try:
        return MsGraphCalendarAttachment.model_validate(value.model_dump(mode="python"))
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None


def validate_msgraph_calendar_attachment_page(
    value: object,
    *,
    event: MsGraphCalendarEventChange,
    graph_base_url: str,
) -> MsGraphCalendarAttachmentPage:
    if not isinstance(value, MsGraphCalendarAttachmentPage):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None

    validated_event = _require_active_event(event)

    try:
        raw = value.model_dump(mode="python")
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None

    if (
        raw.get("mailbox_user_id") != validated_event.mailbox_user_id
        or raw.get("calendar_remote_id") != validated_event.calendar_remote_id
        or raw.get("event_remote_id") != validated_event.remote_id
        or raw.get("event_revision") != validated_event.change_key
    ):
        raise MsGraphCalendarEventChanged() from None

    try:
        raw_items = value.items
        raw_continuation = value.continuation
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None
    if type(raw_items) is not tuple:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None

    validated_items: list[MsGraphCalendarAttachment] = []
    for item in raw_items:
        if not isinstance(item, MsGraphCalendarAttachment):
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None
        validated_item = validate_msgraph_calendar_attachment(item)
        if (
            validated_item.mailbox_user_id != validated_event.mailbox_user_id
            or validated_item.calendar_remote_id != validated_event.calendar_remote_id
            or validated_item.event_remote_id != validated_event.remote_id
            or validated_item.event_revision != validated_event.change_key
        ):
            raise MsGraphCalendarEventChanged() from None
        validated_items.append(validated_item)

    remote_ids = [item.remote_id for item in validated_items]
    if len(remote_ids) != len(set(remote_ids)):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None

    validated_continuation: MsGraphKnowledgeContinuation | None
    if raw_continuation is None:
        validated_continuation = None
    else:
        if not isinstance(raw_continuation, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None
        try:
            validated_continuation = validate_msgraph_calendar_attachments_continuation(
                raw_continuation,
                mailbox_user_id=validated_event.mailbox_user_id,
                calendar_id=validated_event.calendar_remote_id,
                event_id=validated_event.remote_id,
                graph_base_url=graph_base_url,
            )
        except IntegrationConfigurationError:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None

    return _safe_construct_attachment_page(
        mailbox_user_id=validated_event.mailbox_user_id,
        calendar_remote_id=validated_event.calendar_remote_id,
        event_remote_id=validated_event.remote_id,
        event_revision=validated_event.change_key,
        items=tuple(validated_items),
        continuation=validated_continuation,
    )


def validate_msgraph_calendar_file_attachment_content(
    value: object,
    *,
    event: MsGraphCalendarEventChange,
    attachment: MsGraphCalendarAttachment,
    max_bytes: int,
) -> MsGraphCalendarFileAttachmentContent:
    validated_max_bytes = _validate_max_bytes(max_bytes)

    if not isinstance(value, MsGraphCalendarFileAttachmentContent):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None

    validated_event = _require_active_event(event)
    validated_attachment = validate_msgraph_calendar_attachment(attachment)

    if validated_attachment.kind is not MsGraphCalendarAttachmentKind.FILE:
        raise IntegrationConfigurationError(_UNSUPPORTED_ATTACHMENT_CONTENT) from None

    if (
        validated_attachment.mailbox_user_id != validated_event.mailbox_user_id
        or validated_attachment.calendar_remote_id != validated_event.calendar_remote_id
        or validated_attachment.event_remote_id != validated_event.remote_id
        or validated_attachment.event_revision != validated_event.change_key
    ):
        raise MsGraphCalendarEventChanged() from None

    try:
        raw = value.model_dump(mode="python")
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None

    try:
        result_mailbox_user_id = validate_msgraph_mailbox_user_id(raw["mailbox_user_id"])
        result_calendar_remote_id = validate_msgraph_calendar_id(raw["calendar_remote_id"])
        result_event_remote_id = validate_msgraph_calendar_event_id(raw["event_remote_id"])
        result_event_revision = _validate_content_revision(raw["event_revision"])
        result_attachment_remote_id = validate_msgraph_calendar_attachment_id(
            raw["attachment_remote_id"]
        )
        result_name = _validate_attachment_name(raw["name"])
        result_is_inline = raw["is_inline"]
        if type(result_is_inline) is not bool:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        data = raw["data"]
        if type(data) is not bytes:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        size_bytes = raw["size_bytes"]
        if type(size_bytes) is not int:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
        content_hash = raw["content_hash"]
        if not isinstance(content_hash, str) or not _CONTENT_HASH_PATTERN.match(content_hash):
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE)
    except KeyError:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None

    if (
        result_mailbox_user_id != validated_event.mailbox_user_id
        or result_calendar_remote_id != validated_event.calendar_remote_id
        or result_event_remote_id != validated_event.remote_id
        or result_event_revision != validated_event.change_key
        or result_attachment_remote_id != validated_attachment.remote_id
    ):
        raise MsGraphCalendarEventChanged() from None

    if size_bytes != len(data):
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None

    if size_bytes != validated_attachment.size_bytes:
        raise MsGraphCalendarEventChanged() from None

    if size_bytes > validated_max_bytes:
        raise MsGraphCalendarAttachmentTooLarge() from None

    if hashlib.sha256(data).hexdigest() != content_hash:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None

    if result_name != validated_attachment.name:
        raise MsGraphCalendarEventChanged() from None
    if result_is_inline != validated_attachment.is_inline:
        raise MsGraphCalendarEventChanged() from None

    if "content_type" in raw and raw["content_type"] is not None:
        try:
            result_content_type = _validate_model_optional_string(
                raw["content_type"],
                max_length=_MAX_CONTENT_TYPE_LEN,
            )
        except ValueError:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None
        if result_content_type != validated_attachment.content_type:
            raise MsGraphCalendarEventChanged() from None

    if "content_id" in raw and raw["content_id"] is not None:
        try:
            result_content_id = _validate_model_optional_string(
                raw["content_id"],
                max_length=_MAX_CONTENT_ID_LEN,
            )
        except ValueError:
            raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None
        if result_content_id != validated_attachment.content_id:
            raise MsGraphCalendarEventChanged() from None

    try:
        return _safe_construct_file_attachment_content(
            mailbox_user_id=validated_event.mailbox_user_id,
            calendar_remote_id=validated_event.calendar_remote_id,
            event_remote_id=validated_event.remote_id,
            event_revision=validated_event.change_key,
            attachment_remote_id=validated_attachment.remote_id,
            name=validated_attachment.name,
            content_type=validated_attachment.content_type,
            is_inline=validated_attachment.is_inline,
            content_id=validated_attachment.content_id,
            data=data,
            size_bytes=size_bytes,
            content_hash=content_hash,
        )
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_ATTACHMENTS_RESPONSE) from None


def _validate_page_limit(limit: object) -> int:
    if type(limit) is not int:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_ATTACHMENTS_REQUEST) from None
    if limit < _MIN_PAGE_LIMIT or limit > _MAX_PAGE_LIMIT:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_ATTACHMENTS_REQUEST) from None
    return limit


def _validate_max_bytes(max_bytes: object) -> int:
    if type(max_bytes) is not int:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_ATTACHMENTS_REQUEST) from None
    if max_bytes < 1 or max_bytes > ABSOLUTE_CALENDAR_ATTACHMENT_MAX_BYTES:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_ATTACHMENTS_REQUEST) from None
    return max_bytes


def _response_status_code(response: object) -> int:
    try:
        status_code = response.status_code
    except AttributeError:
        raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None
    if type(status_code) is not int:
        raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None
    if status_code < 100 or status_code > 599:
        raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None
    return status_code


def _response_headers(response: object) -> Mapping[str, str]:
    try:
        headers = response.headers
    except AttributeError:
        raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None
    if not isinstance(headers, Mapping):
        raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None
    return headers


def _parse_content_length(headers: Mapping[str, str]) -> int | None:
    raw_value: str | None = None
    try:
        header_items = headers.items()
    except Exception:
        raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None
    try:
        for key, value in header_items:
            if not isinstance(key, str):
                raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None
            if key.lower() == "content-length":
                if raw_value is not None:
                    raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None
                if not isinstance(value, str):
                    raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None
                raw_value = value
    except IntegrationDependencyError:
        raise
    except Exception:
        raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None
    if raw_value is None:
        return None
    trimmed = raw_value.strip()
    if not trimmed or not trimmed.isdigit():
        raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None
    parsed = int(trimmed)
    if parsed < 0:
        raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None
    return parsed


def _raise_for_attachment_download_response(response: object) -> None:
    status_code = _response_status_code(response)
    if status_code == 200:
        return
    if status_code == 206:
        raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None
    if 300 <= status_code <= 399:
        raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None
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
        MsGraphCalendarEventChanged,
        MsGraphCalendarAttachmentTooLarge,
    ):
        raise
    except Exception:
        raise IntegrationDependencyError(
            "Microsoft Graph knowledge dependency is unavailable"
        ) from None


@runtime_checkable
class MsGraphCalendarAttachmentsReadClient(Protocol):
    def read_calendar_attachments_page(
        self,
        *,
        event: MsGraphCalendarEventChange,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphCalendarAttachmentPage:
        ...

    def read_calendar_file_attachment_content(
        self,
        *,
        event: MsGraphCalendarEventChange,
        attachment: MsGraphCalendarAttachment,
        max_bytes: int,
    ) -> MsGraphCalendarFileAttachmentContent:
        ...


class MsGraphCalendarAttachmentsReader:
    """Calendar attachment inventory and bounded fileAttachment content reader."""

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

    def read_attachments_page(
        self,
        *,
        event: MsGraphCalendarEventChange,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphCalendarAttachmentPage:
        validated_event = _require_active_event(event)
        validated_limit = _validate_page_limit(limit)

        read_and_validate_current_calendar_event_observation(
            event=validated_event,
            transport=self._transport,
        )

        if continuation is None:
            quoted_mailbox = quote(validated_event.mailbox_user_id, safe="")
            quoted_calendar = quote(validated_event.calendar_remote_id, safe="")
            quoted_event_id = quote(validated_event.remote_id, safe="")
            path = (
                f"/users/{quoted_mailbox}/calendars/{quoted_calendar}"
                f"/events/{quoted_event_id}/attachments"
            )
            payload = self._transport.get_initial_json(
                path=path,
                params={"$top": validated_limit, "$select": _ATTACHMENTS_SELECT},
                headers={"Prefer": 'IdType="ImmutableId"'},
                not_found_is_dependency=True,
            )
        else:
            validated_continuation = validate_msgraph_calendar_attachments_continuation(
                continuation,
                mailbox_user_id=validated_event.mailbox_user_id,
                calendar_id=validated_event.calendar_remote_id,
                event_id=validated_event.remote_id,
                graph_base_url=self._config.graph_base_url,
            )
            payload = self._transport.get_continuation_json(
                continuation=validated_continuation,
                headers={"Prefer": 'IdType="ImmutableId"'},
                not_found_is_dependency=True,
            )

        collection_page = parse_msgraph_collection_page(
            payload,
            graph_base_url=self._config.graph_base_url,
            delta_mode=False,
        )
        parsed_items = tuple(
            parse_msgraph_calendar_attachment(raw_item, event=validated_event)
            for raw_item in collection_page.items
        )
        page = validate_msgraph_calendar_attachment_page(
            _safe_construct_attachment_page(
                mailbox_user_id=validated_event.mailbox_user_id,
                calendar_remote_id=validated_event.calendar_remote_id,
                event_remote_id=validated_event.remote_id,
                event_revision=validated_event.change_key,
                items=parsed_items,
                continuation=collection_page.continuation,
            ),
            event=validated_event,
            graph_base_url=self._config.graph_base_url,
        )

        read_and_validate_current_calendar_event_observation(
            event=validated_event,
            transport=self._transport,
        )
        return page

    def read_file_attachment_content(
        self,
        *,
        event: MsGraphCalendarEventChange,
        attachment: MsGraphCalendarAttachment,
        max_bytes: int,
    ) -> MsGraphCalendarFileAttachmentContent:
        validated_event = _require_active_event(event)
        validated_attachment = validate_msgraph_calendar_attachment(attachment)
        validated_max_bytes = _validate_max_bytes(max_bytes)

        if (
            validated_attachment.mailbox_user_id != validated_event.mailbox_user_id
            or validated_attachment.calendar_remote_id != validated_event.calendar_remote_id
            or validated_attachment.event_remote_id != validated_event.remote_id
            or validated_attachment.event_revision != validated_event.change_key
        ):
            raise MsGraphCalendarEventChanged() from None

        if validated_attachment.kind is not MsGraphCalendarAttachmentKind.FILE:
            raise IntegrationConfigurationError(_UNSUPPORTED_ATTACHMENT_CONTENT) from None

        if validated_attachment.size_bytes > validated_max_bytes:
            raise MsGraphCalendarAttachmentTooLarge() from None

        read_and_validate_current_calendar_event_observation(
            event=validated_event,
            transport=self._transport,
        )

        quoted_mailbox = quote(validated_event.mailbox_user_id, safe="")
        quoted_calendar = quote(validated_event.calendar_remote_id, safe="")
        quoted_event_id = quote(validated_event.remote_id, safe="")
        quoted_attachment_id = quote(validated_attachment.remote_id, safe="")
        path = (
            f"/users/{quoted_mailbox}/calendars/{quoted_calendar}"
            f"/events/{quoted_event_id}/attachments/{quoted_attachment_id}/$value"
        )

        def _do_stream() -> object:
            return self._graph_http_client.stream(
                "GET",
                path,
                headers={
                    "Accept": "application/octet-stream",
                    "Prefer": 'IdType="ImmutableId"',
                },
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
                _raise_for_attachment_download_response(response)
                headers = _response_headers(response)
                content_length = _parse_content_length(headers)
                if content_length is not None and content_length > validated_max_bytes:
                    raise MsGraphCalendarAttachmentTooLarge() from None

                try:
                    iter_bytes = response.iter_bytes
                except AttributeError:
                    raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None
                if not callable(iter_bytes):
                    raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None

                buffer = bytearray()
                for chunk in iter_bytes():
                    if type(chunk) is not bytes:
                        raise IntegrationDependencyError(_INVALID_ATTACHMENT_RESPONSE) from None
                    buffer.extend(chunk)
                    if len(buffer) > validated_max_bytes:
                        raise MsGraphCalendarAttachmentTooLarge() from None

                data = bytes(buffer)
                if content_length is not None and len(data) != content_length:
                    raise MsGraphCalendarEventChanged() from None
                if validated_attachment.size_bytes != len(data):
                    raise MsGraphCalendarEventChanged() from None
        except (
            IntegrationConfigurationError,
            IntegrationDependencyError,
            MsGraphCalendarEventChanged,
            MsGraphCalendarAttachmentTooLarge,
        ):
            raise
        except Exception:
            raise IntegrationDependencyError(
                "Microsoft Graph knowledge dependency is unavailable"
            ) from None

        read_and_validate_current_calendar_event_observation(
            event=validated_event,
            transport=self._transport,
        )

        content_hash = hashlib.sha256(data).hexdigest()
        return validate_msgraph_calendar_file_attachment_content(
            _safe_construct_file_attachment_content(
                mailbox_user_id=validated_event.mailbox_user_id,
                calendar_remote_id=validated_event.calendar_remote_id,
                event_remote_id=validated_event.remote_id,
                event_revision=validated_event.change_key,
                attachment_remote_id=validated_attachment.remote_id,
                name=validated_attachment.name,
                content_type=validated_attachment.content_type,
                is_inline=validated_attachment.is_inline,
                content_id=validated_attachment.content_id,
                data=data,
                size_bytes=len(data),
                content_hash=content_hash,
            ),
            event=validated_event,
            attachment=validated_attachment,
            max_bytes=validated_max_bytes,
        )
