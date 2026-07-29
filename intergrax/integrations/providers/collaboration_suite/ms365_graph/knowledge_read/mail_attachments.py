# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Mail knowledge-read: attachment inventory and bounded file content."""

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
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.common import (
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeTransport,
    parse_msgraph_collection_page,
    validate_msgraph_continuation_url,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_content import (
    MsGraphMailMessageChanged,
    read_and_validate_current_mail_message_observation,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_folders import (
    validate_msgraph_mail_folder_id,
    validate_msgraph_mailbox_user_id,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_messages import (
    MsGraphMailMessageChange,
    MsGraphMailMessageChangeKind,
    validate_msgraph_mail_message_change,
    validate_msgraph_mail_message_id,
)

DEFAULT_MAIL_ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024
ABSOLUTE_MAIL_ATTACHMENT_MAX_BYTES = 25 * 1024 * 1024

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_MAIL_ATTACHMENTS_RESPONSE = "unexpected Microsoft Graph Mail attachments response"
_INVALID_MAIL_ATTACHMENTS_REQUEST = "invalid Microsoft Graph Mail attachments request"
_INVALID_MAIL_ATTACHMENTS_CONTINUATION = "invalid Microsoft Graph Mail attachments continuation"
_UNSUPPORTED_ATTACHMENT_CONTENT = (
    "Microsoft Graph Mail attachment content is not supported for this attachment type"
)
_INVALID_ATTACHMENT_RESPONSE = "Microsoft Graph Mail attachment response is invalid"
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_CONTENT_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_ATTACHMENT_ID_LEN = 2048
_MAX_ATTACHMENT_NAME_LEN = 4096
_MAX_CONTENT_TYPE_LEN = 1024
_MAX_CONTENT_ID_LEN = 4096
_MIN_PAGE_LIMIT = 1
_MAX_PAGE_LIMIT = 200

_ODATA_FILE_ATTACHMENT = "#microsoft.graph.fileAttachment"
_ODATA_ITEM_ATTACHMENT = "#microsoft.graph.itemAttachment"
_ODATA_REFERENCE_ATTACHMENT = "#microsoft.graph.referenceAttachment"

_ATTACHMENTS_SELECT = (
    "id,name,contentType,size,isInline,contentId,lastModifiedDateTime"
)

_SLASH_MESSAGE_ATTACHMENTS_RE = re.compile(
    r"^messages/([^/]+)/attachments$",
    re.IGNORECASE,
)
_ODATA_MESSAGE_ATTACHMENTS_RE = re.compile(
    r"^messages\('((?:[^']|'')*)'\)/attachments$",
    re.IGNORECASE,
)


class MsGraphMailAttachmentTooLarge(IntegrationConfigurationError):
    """Mail attachment exceeds the configured byte limit."""

    def __init__(self) -> None:
        super().__init__("Microsoft Graph Mail attachment exceeds the configured content limit")


class MsGraphMailAttachmentKind(StrEnum):
    FILE = "file"
    ITEM = "item"
    REFERENCE = "reference"
    UNKNOWN = "unknown"


def _validate_attachment_remote_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    if len(trimmed) > _MAX_ATTACHMENT_ID_LEN:
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    return trimmed


def _validate_attachment_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    if "\x00" in value:
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    if len(value) > _MAX_ATTACHMENT_NAME_LEN:
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    return value


def _validate_optional_trimmed_string(value: object, *, max_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        return None
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    if len(trimmed) > max_length:
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    return trimmed


def _validate_content_revision(value: object) -> str:
    result = _validate_optional_trimmed_string(value, max_length=2048)
    if result is None:
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    return result


def _normalize_model_datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    return value.astimezone(timezone.utc)


def _parse_timezone_aware_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    if trimmed.endswith("Z"):
        trimmed = f"{trimmed[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(trimmed)
    except ValueError:
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    return parsed.astimezone(timezone.utc)


def _map_attachment_kind(odata_type: object) -> MsGraphMailAttachmentKind:
    if not isinstance(odata_type, str):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    trimmed = odata_type.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    if trimmed == _ODATA_FILE_ATTACHMENT:
        return MsGraphMailAttachmentKind.FILE
    if trimmed == _ODATA_ITEM_ATTACHMENT:
        return MsGraphMailAttachmentKind.ITEM
    if trimmed == _ODATA_REFERENCE_ATTACHMENT:
        return MsGraphMailAttachmentKind.REFERENCE
    return MsGraphMailAttachmentKind.UNKNOWN


def _require_active_message(message: object) -> MsGraphMailMessageChange:
    validated = validate_msgraph_mail_message_change(message)
    if validated.kind is not MsGraphMailMessageChangeKind.ACTIVE:
        raise MsGraphMailMessageChanged() from None
    return validated


class MsGraphMailAttachment(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    message_remote_id: str
    scope_folder_id: str
    message_revision: str = Field(repr=False)

    remote_id: str
    kind: MsGraphMailAttachmentKind

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

    @field_validator("scope_folder_id", mode="before")
    @classmethod
    def _validate_scope_folder_id(cls, value: object) -> str:
        return validate_msgraph_mail_folder_id(value)

    @field_validator("message_revision", mode="before")
    @classmethod
    def _validate_message_revision(cls, value: object) -> str:
        return _validate_content_revision(value)

    @field_validator("remote_id", mode="before")
    @classmethod
    def _validate_remote_id(cls, value: object) -> str:
        return _validate_attachment_remote_id(value)

    @field_validator("message_remote_id", mode="before")
    @classmethod
    def _validate_message_remote_id(cls, value: object) -> str:
        return validate_msgraph_mail_message_id(value)

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value: object) -> str:
        return _validate_attachment_name(value)

    @field_validator("content_type", "content_id", mode="before")
    @classmethod
    def _validate_optional_strings(cls, value: object, info: Any) -> str | None:
        max_length = _MAX_CONTENT_TYPE_LEN if info.field_name == "content_type" else _MAX_CONTENT_ID_LEN
        return _validate_optional_trimmed_string(value, max_length=max_length)

    @field_validator("size_bytes", mode="before")
    @classmethod
    def _validate_size_bytes(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        if value < 0:
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        return value

    @field_validator("is_inline", mode="before")
    @classmethod
    def _validate_is_inline(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        return value

    @field_validator("last_modified_at", mode="before")
    @classmethod
    def _validate_last_modified_at(cls, value: object) -> datetime:
        return _normalize_model_datetime(value)

    @property
    def binary_content_supported(self) -> bool:
        return self.kind is MsGraphMailAttachmentKind.FILE


class MsGraphMailAttachmentPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    message_remote_id: str
    scope_folder_id: str
    message_revision: str = Field(repr=False)

    items: tuple[MsGraphMailAttachment, ...]

    continuation: MsGraphKnowledgeContinuation | None = Field(
        default=None,
        repr=False,
    )

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, value: object) -> tuple[MsGraphMailAttachment, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        return value

    @field_validator("continuation", mode="before")
    @classmethod
    def _validate_continuation(cls, value: object) -> MsGraphKnowledgeContinuation | None:
        if value is None:
            return None
        if not isinstance(value, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        if value.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        return value

    @property
    def has_more(self) -> bool:
        return self.continuation is not None


class MsGraphMailFileAttachmentContent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    message_remote_id: str
    scope_folder_id: str
    message_revision: str = Field(repr=False)

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

    @field_validator("message_remote_id", mode="before")
    @classmethod
    def _validate_message_remote_id(cls, value: object) -> str:
        return validate_msgraph_mail_message_id(value)

    @field_validator("scope_folder_id", mode="before")
    @classmethod
    def _validate_scope_folder_id(cls, value: object) -> str:
        return validate_msgraph_mail_folder_id(value)

    @field_validator("message_revision", mode="before")
    @classmethod
    def _validate_message_revision(cls, value: object) -> str:
        return _validate_content_revision(value)

    @field_validator("attachment_remote_id", mode="before")
    @classmethod
    def _validate_attachment_remote_id_field(cls, value: object) -> str:
        return _validate_attachment_remote_id(value)

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value: object) -> str:
        return _validate_attachment_name(value)

    @field_validator("content_type", "content_id", mode="before")
    @classmethod
    def _validate_optional_strings(cls, value: object, info: Any) -> str | None:
        max_length = _MAX_CONTENT_TYPE_LEN if info.field_name == "content_type" else _MAX_CONTENT_ID_LEN
        return _validate_optional_trimmed_string(value, max_length=max_length)

    @field_validator("is_inline", mode="before")
    @classmethod
    def _validate_is_inline(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        return value

    @field_validator("data", mode="before")
    @classmethod
    def _validate_data(cls, value: object) -> bytes:
        if type(value) is not bytes:
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        return value

    @field_validator("size_bytes", mode="before")
    @classmethod
    def _validate_size_bytes(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        if value < 0:
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        return value

    @field_validator("content_hash", mode="before")
    @classmethod
    def _validate_content_hash_format(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        if not _CONTENT_HASH_PATTERN.match(value):
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_content_shape(self) -> MsGraphMailFileAttachmentContent:
        if self.size_bytes != len(self.data):
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        expected_hash = hashlib.sha256(self.data).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        return self


def _safe_construct_attachment(**kwargs: object) -> MsGraphMailAttachment:
    try:
        return MsGraphMailAttachment(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None


def _safe_construct_attachment_page(**kwargs: object) -> MsGraphMailAttachmentPage:
    try:
        return MsGraphMailAttachmentPage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None


def _safe_construct_file_attachment_content(**kwargs: object) -> MsGraphMailFileAttachmentContent:
    try:
        return MsGraphMailFileAttachmentContent(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None


def parse_msgraph_mail_attachment(
    payload: object,
    *,
    message: MsGraphMailMessageChange,
) -> MsGraphMailAttachment:
    validated_message = _require_active_message(message)
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None

    required_keys = ("@odata.type", "id", "name", "size", "isInline", "lastModifiedDateTime")
    for key in required_keys:
        if key not in payload:
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None

    try:
        kind = _map_attachment_kind(payload.get("@odata.type"))
        remote_id = _validate_attachment_remote_id(payload.get("id"))
        name = _validate_attachment_name(payload.get("name"))
        size_bytes = payload.get("size")
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        is_inline = payload.get("isInline")
        if type(is_inline) is not bool:
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        last_modified_at = _parse_timezone_aware_datetime(payload.get("lastModifiedDateTime"))

        if "contentType" not in payload:
            content_type = None
        elif payload.get("contentType") is None:
            content_type = None
        else:
            content_type = _validate_optional_trimmed_string(
                payload.get("contentType"),
                max_length=_MAX_CONTENT_TYPE_LEN,
            )

        if "contentId" not in payload:
            content_id = None
        elif payload.get("contentId") is None:
            content_id = None
        else:
            content_id = _validate_optional_trimmed_string(
                payload.get("contentId"),
                max_length=_MAX_CONTENT_ID_LEN,
            )
    except ValueError:
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None

    return _safe_construct_attachment(
        mailbox_user_id=validated_message.mailbox_user_id,
        message_remote_id=validated_message.remote_id,
        scope_folder_id=validated_message.scope_folder_id,
        message_revision=validated_message.change_key,
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


def _decode_odata_message_literal(literal: str) -> str:
    return literal.replace("''", "'")


def _extract_message_attachments_path(
    path: str,
    *,
    graph_base_path: str,
) -> tuple[str, str] | None:
    normalized = path.rstrip("/") or "/"
    expected_prefix = f"{graph_base_path.rstrip('/')}/users/"
    if not normalized.startswith(expected_prefix):
        return None

    remainder = normalized[len(expected_prefix) :]
    slash_index = remainder.find("/")
    if slash_index <= 0:
        return None

    mailbox_segment = remainder[:slash_index]
    after_mailbox = remainder[slash_index + 1 :]

    if "/$value" in after_mailbox.lower():
        return None
    if "mailFolders" in after_mailbox:
        return None
    if "delta" in after_mailbox.lower():
        return None
    if after_mailbox.count("/") > 2:
        return None

    slash_match = _SLASH_MESSAGE_ATTACHMENTS_RE.match(after_mailbox)
    if slash_match is not None:
        message_segment = slash_match.group(1)
        if not message_segment:
            return None
        return unquote(mailbox_segment), unquote(message_segment)

    odata_match = _ODATA_MESSAGE_ATTACHMENTS_RE.match(after_mailbox)
    if odata_match is not None:
        message_literal = odata_match.group(1)
        if not message_literal:
            return None
        return unquote(mailbox_segment), _decode_odata_message_literal(message_literal)

    return None


def validate_msgraph_mail_attachments_continuation(
    continuation: object,
    *,
    mailbox_user_id: str,
    message_id: str,
    graph_base_url: str,
) -> MsGraphKnowledgeContinuation:
    if not isinstance(continuation, MsGraphKnowledgeContinuation):
        raise IntegrationConfigurationError(_INVALID_MAIL_ATTACHMENTS_CONTINUATION) from None

    try:
        revalidated = MsGraphKnowledgeContinuation.model_validate(
            continuation.model_dump(mode="python")
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise IntegrationConfigurationError(_INVALID_MAIL_ATTACHMENTS_CONTINUATION) from None

    if revalidated.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
        raise IntegrationConfigurationError(_INVALID_MAIL_ATTACHMENTS_CONTINUATION) from None

    try:
        validated_url = validate_msgraph_continuation_url(
            revalidated.url,
            graph_base_url=graph_base_url,
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MAIL_ATTACHMENTS_CONTINUATION) from None

    parsed = urlparse(validated_url)
    extracted = _extract_message_attachments_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted is None:
        raise IntegrationConfigurationError(_INVALID_MAIL_ATTACHMENTS_CONTINUATION) from None

    extracted_mailbox_user_id, extracted_message_id = extracted
    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(mailbox_user_id)
        validated_message_id = validate_msgraph_mail_message_id(message_id)
        validated_extracted_mailbox_user_id = validate_msgraph_mailbox_user_id(
            extracted_mailbox_user_id
        )
        validated_extracted_message_id = validate_msgraph_mail_message_id(extracted_message_id)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MAIL_ATTACHMENTS_CONTINUATION) from None

    if (
        validated_extracted_mailbox_user_id != validated_mailbox_user_id
        or validated_extracted_message_id != validated_message_id
    ):
        raise IntegrationConfigurationError(_INVALID_MAIL_ATTACHMENTS_CONTINUATION) from None

    return revalidated


def validate_msgraph_mail_attachment(value: object) -> MsGraphMailAttachment:
    if not isinstance(value, MsGraphMailAttachment):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None
    try:
        return MsGraphMailAttachment.model_validate(value.model_dump(mode="python"))
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None


def validate_msgraph_mail_attachment_page(
    value: object,
    *,
    message: MsGraphMailMessageChange,
    graph_base_url: str,
) -> MsGraphMailAttachmentPage:
    if not isinstance(value, MsGraphMailAttachmentPage):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None

    validated_message = _require_active_message(message)

    try:
        raw = value.model_dump(mode="python")
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None

    if (
        raw.get("mailbox_user_id") != validated_message.mailbox_user_id
        or raw.get("message_remote_id") != validated_message.remote_id
        or raw.get("scope_folder_id") != validated_message.scope_folder_id
        or raw.get("message_revision") != validated_message.change_key
    ):
        raise MsGraphMailMessageChanged() from None

    try:
        raw_items = value.items
        raw_continuation = value.continuation
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None
    if type(raw_items) is not tuple:
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None

    validated_items: list[MsGraphMailAttachment] = []
    for item in raw_items:
        if not isinstance(item, MsGraphMailAttachment):
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None
        validated_item = validate_msgraph_mail_attachment(item)
        if (
            validated_item.mailbox_user_id != validated_message.mailbox_user_id
            or validated_item.message_remote_id != validated_message.remote_id
            or validated_item.scope_folder_id != validated_message.scope_folder_id
            or validated_item.message_revision != validated_message.change_key
        ):
            raise MsGraphMailMessageChanged() from None
        validated_items.append(validated_item)

    remote_ids = [item.remote_id for item in validated_items]
    if len(remote_ids) != len(set(remote_ids)):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None

    validated_continuation: MsGraphKnowledgeContinuation | None
    if raw_continuation is None:
        validated_continuation = None
    else:
        if not isinstance(raw_continuation, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None
        try:
            validated_continuation = validate_msgraph_mail_attachments_continuation(
                raw_continuation,
                mailbox_user_id=validated_message.mailbox_user_id,
                message_id=validated_message.remote_id,
                graph_base_url=graph_base_url,
            )
        except IntegrationConfigurationError:
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None

    return _safe_construct_attachment_page(
        mailbox_user_id=validated_message.mailbox_user_id,
        message_remote_id=validated_message.remote_id,
        scope_folder_id=validated_message.scope_folder_id,
        message_revision=validated_message.change_key,
        items=tuple(validated_items),
        continuation=validated_continuation,
    )


def validate_msgraph_mail_file_attachment_content(
    value: object,
    *,
    message: MsGraphMailMessageChange,
    attachment: MsGraphMailAttachment,
    max_bytes: int,
) -> MsGraphMailFileAttachmentContent:
    if not isinstance(value, MsGraphMailFileAttachmentContent):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None

    validated_message = _require_active_message(message)
    validated_attachment = validate_msgraph_mail_attachment(attachment)

    if (
        validated_attachment.mailbox_user_id != validated_message.mailbox_user_id
        or validated_attachment.message_remote_id != validated_message.remote_id
        or validated_attachment.scope_folder_id != validated_message.scope_folder_id
        or validated_attachment.message_revision != validated_message.change_key
    ):
        raise MsGraphMailMessageChanged() from None

    try:
        raw = value.model_dump(mode="python")
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None

    if (
        raw.get("mailbox_user_id") != validated_message.mailbox_user_id
        or raw.get("message_remote_id") != validated_message.remote_id
        or raw.get("scope_folder_id") != validated_message.scope_folder_id
        or raw.get("message_revision") != validated_message.change_key
        or raw.get("attachment_remote_id") != validated_attachment.remote_id
    ):
        raise MsGraphMailMessageChanged() from None

    try:
        data = raw["data"]
        if type(data) is not bytes:
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        size_bytes = raw["size_bytes"]
        if type(size_bytes) is not int or size_bytes != len(data):
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
        if size_bytes > max_bytes:
            raise MsGraphMailAttachmentTooLarge() from None
        content_hash = raw["content_hash"]
        if hashlib.sha256(data).hexdigest() != content_hash:
            raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE)
    except KeyError:
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None
    except MsGraphMailAttachmentTooLarge:
        raise

    try:
        return _safe_construct_file_attachment_content(
            mailbox_user_id=validated_message.mailbox_user_id,
            message_remote_id=validated_message.remote_id,
            scope_folder_id=validated_message.scope_folder_id,
            message_revision=validated_message.change_key,
            attachment_remote_id=validated_attachment.remote_id,
            name=raw.get("name", ""),
            content_type=raw.get("content_type"),
            is_inline=raw.get("is_inline", False),
            content_id=raw.get("content_id"),
            data=data,
            size_bytes=size_bytes,
            content_hash=content_hash,
        )
    except ValueError:
        raise ValueError(_MALFORMED_MAIL_ATTACHMENTS_RESPONSE) from None


def _validate_page_limit(limit: object) -> int:
    if type(limit) is not int:
        raise IntegrationConfigurationError(_INVALID_MAIL_ATTACHMENTS_REQUEST) from None
    if limit < _MIN_PAGE_LIMIT or limit > _MAX_PAGE_LIMIT:
        raise IntegrationConfigurationError(_INVALID_MAIL_ATTACHMENTS_REQUEST) from None
    return limit


def _validate_max_bytes(max_bytes: object) -> int:
    if type(max_bytes) is not int:
        raise IntegrationConfigurationError(_INVALID_MAIL_ATTACHMENTS_REQUEST) from None
    if max_bytes < 1 or max_bytes > ABSOLUTE_MAIL_ATTACHMENT_MAX_BYTES:
        raise IntegrationConfigurationError(_INVALID_MAIL_ATTACHMENTS_REQUEST) from None
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
        MsGraphMailMessageChanged,
        MsGraphMailAttachmentTooLarge,
    ):
        raise
    except Exception:
        raise IntegrationDependencyError(
            "Microsoft Graph knowledge dependency is unavailable"
        ) from None


@runtime_checkable
class MsGraphMailAttachmentsReadClient(Protocol):
    def read_mail_attachments_page(
        self,
        *,
        message: MsGraphMailMessageChange,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphMailAttachmentPage:
        ...

    def read_mail_file_attachment_content(
        self,
        *,
        message: MsGraphMailMessageChange,
        attachment: MsGraphMailAttachment,
        max_bytes: int,
    ) -> MsGraphMailFileAttachmentContent:
        ...


class MsGraphMailAttachmentsReader:
    """Mail attachment inventory and bounded fileAttachment content reader."""

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
        message: MsGraphMailMessageChange,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphMailAttachmentPage:
        validated_message = _require_active_message(message)
        validated_limit = _validate_page_limit(limit)

        read_and_validate_current_mail_message_observation(
            message=validated_message,
            transport=self._transport,
        )

        if continuation is None:
            quoted_mailbox = quote(validated_message.mailbox_user_id, safe="")
            quoted_message_id = quote(validated_message.remote_id, safe="")
            path = f"/users/{quoted_mailbox}/messages/{quoted_message_id}/attachments"
            payload = self._transport.get_initial_json(
                path=path,
                params={"$top": validated_limit, "$select": _ATTACHMENTS_SELECT},
                headers={"Prefer": 'IdType="ImmutableId"'},
                not_found_is_dependency=True,
            )
        else:
            validated_continuation = validate_msgraph_mail_attachments_continuation(
                continuation,
                mailbox_user_id=validated_message.mailbox_user_id,
                message_id=validated_message.remote_id,
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
            parse_msgraph_mail_attachment(raw_item, message=validated_message)
            for raw_item in collection_page.items
        )
        page = validate_msgraph_mail_attachment_page(
            _safe_construct_attachment_page(
                mailbox_user_id=validated_message.mailbox_user_id,
                message_remote_id=validated_message.remote_id,
                scope_folder_id=validated_message.scope_folder_id,
                message_revision=validated_message.change_key,
                items=parsed_items,
                continuation=collection_page.continuation,
            ),
            message=validated_message,
            graph_base_url=self._config.graph_base_url,
        )

        read_and_validate_current_mail_message_observation(
            message=validated_message,
            transport=self._transport,
        )
        return page

    def read_file_attachment_content(
        self,
        *,
        message: MsGraphMailMessageChange,
        attachment: MsGraphMailAttachment,
        max_bytes: int,
    ) -> MsGraphMailFileAttachmentContent:
        validated_message = _require_active_message(message)
        validated_attachment = validate_msgraph_mail_attachment(attachment)
        validated_max_bytes = _validate_max_bytes(max_bytes)

        if (
            validated_attachment.mailbox_user_id != validated_message.mailbox_user_id
            or validated_attachment.message_remote_id != validated_message.remote_id
            or validated_attachment.scope_folder_id != validated_message.scope_folder_id
            or validated_attachment.message_revision != validated_message.change_key
        ):
            raise MsGraphMailMessageChanged() from None

        if validated_attachment.kind is not MsGraphMailAttachmentKind.FILE:
            raise IntegrationConfigurationError(_UNSUPPORTED_ATTACHMENT_CONTENT) from None

        if validated_attachment.size_bytes > validated_max_bytes:
            raise MsGraphMailAttachmentTooLarge() from None

        read_and_validate_current_mail_message_observation(
            message=validated_message,
            transport=self._transport,
        )

        quoted_mailbox = quote(validated_message.mailbox_user_id, safe="")
        quoted_message_id = quote(validated_message.remote_id, safe="")
        quoted_attachment_id = quote(validated_attachment.remote_id, safe="")
        path = (
            f"/users/{quoted_mailbox}/messages/{quoted_message_id}"
            f"/attachments/{quoted_attachment_id}/$value"
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
                    raise MsGraphMailAttachmentTooLarge() from None

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
                        raise MsGraphMailAttachmentTooLarge() from None

                data = bytes(buffer)
                if content_length is not None and len(data) != content_length:
                    raise MsGraphMailMessageChanged() from None
                if validated_attachment.size_bytes != len(data):
                    raise MsGraphMailMessageChanged() from None
        except (
            IntegrationConfigurationError,
            IntegrationDependencyError,
            MsGraphMailMessageChanged,
            MsGraphMailAttachmentTooLarge,
        ):
            raise
        except Exception:
            raise IntegrationDependencyError(
                "Microsoft Graph knowledge dependency is unavailable"
            ) from None

        read_and_validate_current_mail_message_observation(
            message=validated_message,
            transport=self._transport,
        )

        content_hash = hashlib.sha256(data).hexdigest()
        return validate_msgraph_mail_file_attachment_content(
            _safe_construct_file_attachment_content(
                mailbox_user_id=validated_message.mailbox_user_id,
                message_remote_id=validated_message.remote_id,
                scope_folder_id=validated_message.scope_folder_id,
                message_revision=validated_message.change_key,
                attachment_remote_id=validated_attachment.remote_id,
                name=validated_attachment.name,
                content_type=validated_attachment.content_type,
                is_inline=validated_attachment.is_inline,
                content_id=validated_attachment.content_id,
                data=data,
                size_bytes=len(data),
                content_hash=content_hash,
            ),
            message=validated_message,
            attachment=validated_attachment,
            max_bytes=validated_max_bytes,
        )
