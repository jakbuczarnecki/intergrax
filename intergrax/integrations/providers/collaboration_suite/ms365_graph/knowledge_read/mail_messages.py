# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Mail knowledge-read: per-folder message metadata delta for one known folder."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Protocol, runtime_checkable
from urllib.parse import quote, unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from intergrax.integrations.contracts.base import IntegrationConfigurationError
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
    validate_msgraph_mail_folder_id,
    validate_msgraph_mailbox_user_id,
)

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_MAIL_MESSAGES_RESPONSE = "unexpected Microsoft Graph mail messages response"
_INVALID_MAIL_MESSAGES_REQUEST = "invalid Microsoft Graph mail messages request"
_INVALID_MAIL_MESSAGES_CONTINUATION = "invalid Microsoft Graph mail messages delta continuation"
_MAX_MESSAGE_ID_LEN = 2048
_MAX_CHANGE_KEY_LEN = 2048
_MAX_CONVERSATION_ID_LEN = 2048
_MAX_INTERNET_MESSAGE_ID_LEN = 4096
_MAX_REMOVED_REASON_LEN = 256
_MAX_SUBJECT_LEN = 4096
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_MIN_DELTA_LIMIT = 1
_MAX_DELTA_LIMIT = 200

_IMMUTABLE_ID_HEADERS = {
    "Prefer": 'IdType="ImmutableId"',
}

_MAIL_MESSAGES_SELECT = (
    "id,parentFolderId,changeKey,conversationId,internetMessageId,subject,"
    "createdDateTime,lastModifiedDateTime,receivedDateTime,sentDateTime,"
    "isRead,isDraft,hasAttachments,importance"
)

_SLASH_MESSAGES_DELTA_RE = re.compile(
    r"^mailFolders/([^/]+)/messages/delta$",
    re.IGNORECASE,
)
_ODATA_MESSAGES_DELTA_RE = re.compile(
    r"^mailFolders\('((?:[^']|'')*)'\)/messages/delta$",
    re.IGNORECASE,
)


class MsGraphMailMessageChangeKind(StrEnum):
    ACTIVE = "active"
    REMOVED = "removed"


class MsGraphMailImportance(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    UNKNOWN = "unknown"


def validate_msgraph_mail_message_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    if len(trimmed) > _MAX_MESSAGE_ID_LEN:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    return trimmed


def _validate_optional_opaque_string(
    value: object,
    *,
    max_length: int,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    if len(trimmed) > max_length:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    return trimmed


def _validate_subject(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    if "\x00" in value:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    if len(value) > _MAX_SUBJECT_LEN:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    return value


def _normalize_model_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    return value.astimezone(timezone.utc)


def _parse_timezone_aware_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    if trimmed.endswith("Z"):
        trimmed = f"{trimmed[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(trimmed)
    except ValueError:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    return parsed.astimezone(timezone.utc)


def _parse_optional_provider_datetime(mapping: dict[str, object], key: str) -> datetime | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if value is None:
        return None
    return _parse_timezone_aware_datetime(value)


def _parse_required_provider_datetime(mapping: dict[str, object], key: str) -> datetime:
    if key not in mapping:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    value = mapping[key]
    if value is None:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    return _parse_timezone_aware_datetime(value)


def _parse_required_bool(mapping: dict[str, object], key: str) -> bool:
    if key not in mapping:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    value = mapping[key]
    if type(value) is not bool:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    return value


def _parse_optional_provider_string(mapping: dict[str, object], key: str) -> str | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    return trimmed


def _map_importance(value: object) -> MsGraphMailImportance:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
    lowered = trimmed.lower()
    if lowered == "low":
        return MsGraphMailImportance.LOW
    if lowered == "normal":
        return MsGraphMailImportance.NORMAL
    if lowered == "high":
        return MsGraphMailImportance.HIGH
    return MsGraphMailImportance.UNKNOWN


class MsGraphMailMessageChange(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    scope_folder_id: str
    remote_id: str

    kind: MsGraphMailMessageChangeKind

    parent_folder_id: str | None = None
    change_key: str | None = Field(default=None, repr=False)

    conversation_id: str | None = None
    internet_message_id: str | None = Field(default=None, repr=False)
    subject: str | None = Field(default=None, repr=False)

    created_at: datetime | None = None
    last_modified_at: datetime | None = None
    received_at: datetime | None = None
    sent_at: datetime | None = None

    is_read: bool | None = None
    is_draft: bool | None = None
    has_attachments: bool | None = None

    importance: MsGraphMailImportance | None = None

    removed_reason: str | None = None

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("scope_folder_id", "parent_folder_id", mode="before")
    @classmethod
    def _validate_folder_ids(cls, value: object) -> str | None:
        if value is None:
            return None
        return validate_msgraph_mail_folder_id(value)

    @field_validator("remote_id", mode="before")
    @classmethod
    def _validate_remote_id(cls, value: object) -> str:
        return validate_msgraph_mail_message_id(value)

    @field_validator("change_key", mode="before")
    @classmethod
    def _validate_change_key(cls, value: object) -> str | None:
        return _validate_optional_opaque_string(value, max_length=_MAX_CHANGE_KEY_LEN)

    @field_validator("conversation_id", mode="before")
    @classmethod
    def _validate_conversation_id(cls, value: object) -> str | None:
        return _validate_optional_opaque_string(value, max_length=_MAX_CONVERSATION_ID_LEN)

    @field_validator("internet_message_id", mode="before")
    @classmethod
    def _validate_internet_message_id(cls, value: object) -> str | None:
        return _validate_optional_opaque_string(value, max_length=_MAX_INTERNET_MESSAGE_ID_LEN)

    @field_validator("removed_reason", mode="before")
    @classmethod
    def _validate_removed_reason(cls, value: object) -> str | None:
        return _validate_optional_opaque_string(value, max_length=_MAX_REMOVED_REASON_LEN)

    @field_validator("subject", mode="before")
    @classmethod
    def _validate_subject_field(cls, value: object) -> str | None:
        return _validate_subject(value)

    @field_validator(
        "created_at",
        "last_modified_at",
        "received_at",
        "sent_at",
        mode="before",
    )
    @classmethod
    def _validate_datetime_fields(cls, value: object) -> datetime | None:
        return _normalize_model_datetime(value)

    @field_validator("is_read", "is_draft", "has_attachments", mode="before")
    @classmethod
    def _validate_optional_bools(cls, value: object) -> bool | None:
        if value is None:
            return None
        if type(value) is not bool:
            raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_kind_rules(self) -> MsGraphMailMessageChange:
        if self.kind == MsGraphMailMessageChangeKind.ACTIVE:
            if self.parent_folder_id is None or self.parent_folder_id != self.scope_folder_id:
                raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
            if self.change_key is None or self.last_modified_at is None:
                raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
            if self.is_read is None or self.is_draft is None or self.has_attachments is None:
                raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
            if self.importance is None:
                raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
            if self.removed_reason is not None:
                raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
        elif self.kind == MsGraphMailMessageChangeKind.REMOVED:
            if self.removed_reason is None:
                raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
        return self

    @property
    def is_removed(self) -> bool:
        return self.kind is MsGraphMailMessageChangeKind.REMOVED


class MsGraphMailMessageDeltaPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    items: tuple[MsGraphMailMessageChange, ...]
    continuation: MsGraphKnowledgeContinuation = Field(repr=False)

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items_input(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphMailMessageChange):
                raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
        return value

    @field_validator("continuation", mode="before")
    @classmethod
    def _validate_continuation_input(cls, value: object) -> object:
        if not isinstance(value, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
        if value.kind not in {
            MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            MsGraphKnowledgeContinuationKind.DELTA,
        }:
            raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_page_shape(self) -> MsGraphMailMessageDeltaPage:
        remote_ids = [item.remote_id for item in self.items]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE)
        return self

    @property
    def has_more(self) -> bool:
        return self.continuation.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE

    @property
    def is_complete(self) -> bool:
        return self.continuation.kind is MsGraphKnowledgeContinuationKind.DELTA


def _safe_construct_message_change(**kwargs: object) -> MsGraphMailMessageChange:
    try:
        return MsGraphMailMessageChange(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None


def _safe_construct_message_delta_page(**kwargs: object) -> MsGraphMailMessageDeltaPage:
    try:
        return MsGraphMailMessageDeltaPage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None


@runtime_checkable
class MsGraphMailMessagesReadClient(Protocol):
    def read_mail_messages_delta_page(
        self,
        *,
        mailbox_user_id: str,
        folder_id: str,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphMailMessageDeltaPage:
        ...


def validate_msgraph_mail_message_change(value: object) -> MsGraphMailMessageChange:
    """Deep-revalidate a Mail message change against the full model contract."""
    if not isinstance(value, MsGraphMailMessageChange):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None
    try:
        return MsGraphMailMessageChange.model_validate(value.model_dump(mode="python"))
    except (ValueError, TypeError, AttributeError, OverflowError, ValidationError):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None


def validate_msgraph_mail_message_delta_page(
    value: object,
    *,
    mailbox_user_id: str,
    folder_id: str,
    graph_base_url: str,
) -> MsGraphMailMessageDeltaPage:
    """Deep-revalidate a Mail messages delta page and every nested change."""
    if not isinstance(value, MsGraphMailMessageDeltaPage):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None

    try:
        raw_items = value.items
        raw_continuation = value.continuation
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None

    if type(raw_items) is not tuple:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None

    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(mailbox_user_id)
        validated_folder_id = validate_msgraph_mail_folder_id(folder_id)
    except ValueError:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None

    validated_items: list[MsGraphMailMessageChange] = []
    for item in raw_items:
        if not isinstance(item, MsGraphMailMessageChange):
            raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None
        validated_change = validate_msgraph_mail_message_change(item)
        if (
            validated_change.mailbox_user_id != validated_mailbox_user_id
            or validated_change.scope_folder_id != validated_folder_id
        ):
            raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None
        validated_items.append(validated_change)

    remote_ids = [item.remote_id for item in validated_items]
    if len(remote_ids) != len(set(remote_ids)):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None

    if not isinstance(raw_continuation, MsGraphKnowledgeContinuation):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None

    try:
        revalidated_continuation = validate_msgraph_mail_messages_delta_continuation(
            raw_continuation,
            mailbox_user_id=validated_mailbox_user_id,
            folder_id=validated_folder_id,
            graph_base_url=graph_base_url,
        )
    except IntegrationConfigurationError:
        raise

    try:
        return MsGraphMailMessageDeltaPage(
            items=tuple(validated_items),
            continuation=revalidated_continuation,
        )
    except (ValueError, TypeError, AttributeError, OverflowError, ValidationError):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None


def parse_msgraph_mail_message_change(
    payload: object,
    *,
    expected_mailbox_user_id: str,
    expected_folder_id: str,
) -> MsGraphMailMessageChange:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None

    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(expected_mailbox_user_id)
        validated_folder_id = validate_msgraph_mail_folder_id(expected_folder_id)
    except ValueError:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None

    try:
        remote_id = validate_msgraph_mail_message_id(payload.get("id"))
    except ValueError:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None

    if "@removed" in payload:
        removed = payload["@removed"]
        if not isinstance(removed, dict):
            raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None
        reason = removed.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None
        normalized_reason = reason.strip()
        if _ASCII_CONTROL.search(normalized_reason):
            raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None
        if len(normalized_reason) > _MAX_REMOVED_REASON_LEN:
            raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None
        return _safe_construct_message_change(
            mailbox_user_id=validated_mailbox_user_id,
            scope_folder_id=validated_folder_id,
            remote_id=remote_id,
            kind=MsGraphMailMessageChangeKind.REMOVED,
            removed_reason=normalized_reason,
        )

    try:
        parent_folder_id = validate_msgraph_mail_folder_id(payload.get("parentFolderId"))
        if parent_folder_id != validated_folder_id:
            raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None
        change_key = _validate_optional_opaque_string(
            payload.get("changeKey"),
            max_length=_MAX_CHANGE_KEY_LEN,
        )
        if change_key is None:
            raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None
        last_modified_at = _parse_required_provider_datetime(payload, "lastModifiedDateTime")
        is_read = _parse_required_bool(payload, "isRead")
        is_draft = _parse_required_bool(payload, "isDraft")
        has_attachments = _parse_required_bool(payload, "hasAttachments")
        importance = _map_importance(payload.get("importance"))
    except ValueError:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None

    try:
        conversation_id = (
            _parse_optional_provider_string(payload, "conversationId")
            if "conversationId" in payload
            else None
        )
        internet_message_id_raw = payload.get("internetMessageId") if "internetMessageId" in payload else None
        if internet_message_id_raw is None:
            internet_message_id = None
        else:
            internet_message_id = _validate_optional_opaque_string(
                internet_message_id_raw,
                max_length=_MAX_INTERNET_MESSAGE_ID_LEN,
            )
        subject = _validate_subject(payload.get("subject")) if "subject" in payload else None
        created_at = _parse_optional_provider_datetime(payload, "createdDateTime")
        received_at = _parse_optional_provider_datetime(payload, "receivedDateTime")
        sent_at = _parse_optional_provider_datetime(payload, "sentDateTime")
    except ValueError:
        raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None

    return _safe_construct_message_change(
        mailbox_user_id=validated_mailbox_user_id,
        scope_folder_id=validated_folder_id,
        remote_id=remote_id,
        kind=MsGraphMailMessageChangeKind.ACTIVE,
        parent_folder_id=parent_folder_id,
        change_key=change_key,
        conversation_id=conversation_id,
        internet_message_id=internet_message_id,
        subject=subject,
        created_at=created_at,
        last_modified_at=last_modified_at,
        received_at=received_at,
        sent_at=sent_at,
        is_read=is_read,
        is_draft=is_draft,
        has_attachments=has_attachments,
        importance=importance,
    )


def _graph_base_path(graph_base_url: str) -> str:
    parsed_base = urlparse(graph_base_url)
    return parsed_base.path.rstrip("/") or "/"


def _decode_odata_folder_literal(literal: str) -> str:
    return literal.replace("''", "'")


def _extract_mail_messages_delta_path(
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

    slash_match = _SLASH_MESSAGES_DELTA_RE.match(after_mailbox)
    if slash_match is not None:
        folder_segment = slash_match.group(1)
        if not folder_segment:
            return None
        return unquote(mailbox_segment), unquote(folder_segment)

    odata_match = _ODATA_MESSAGES_DELTA_RE.match(after_mailbox)
    if odata_match is not None:
        folder_literal = odata_match.group(1)
        if not folder_literal:
            return None
        return unquote(mailbox_segment), _decode_odata_folder_literal(folder_literal)

    return None


def validate_msgraph_mail_messages_delta_continuation(
    continuation: object,
    *,
    mailbox_user_id: str,
    folder_id: str,
    graph_base_url: str,
) -> MsGraphKnowledgeContinuation:
    if not isinstance(continuation, MsGraphKnowledgeContinuation):
        raise IntegrationConfigurationError(_INVALID_MAIL_MESSAGES_CONTINUATION) from None
    if continuation.kind not in {
        MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        MsGraphKnowledgeContinuationKind.DELTA,
    }:
        raise IntegrationConfigurationError(_INVALID_MAIL_MESSAGES_CONTINUATION) from None

    try:
        validated_url = validate_msgraph_continuation_url(
            continuation.url,
            graph_base_url=graph_base_url,
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MAIL_MESSAGES_CONTINUATION) from None

    parsed = urlparse(validated_url)
    extracted = _extract_mail_messages_delta_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted is None:
        raise IntegrationConfigurationError(_INVALID_MAIL_MESSAGES_CONTINUATION) from None

    extracted_mailbox_user_id, extracted_folder_id = extracted
    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(mailbox_user_id)
        validated_folder_id = validate_msgraph_mail_folder_id(folder_id)
        validated_extracted_mailbox_user_id = validate_msgraph_mailbox_user_id(
            extracted_mailbox_user_id
        )
        validated_extracted_folder_id = validate_msgraph_mail_folder_id(extracted_folder_id)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MAIL_MESSAGES_CONTINUATION) from None

    if (
        validated_extracted_mailbox_user_id != validated_mailbox_user_id
        or validated_extracted_folder_id != validated_folder_id
    ):
        raise IntegrationConfigurationError(_INVALID_MAIL_MESSAGES_CONTINUATION) from None

    return continuation


def _deduplicate_message_changes(
    items: tuple[MsGraphMailMessageChange, ...],
) -> tuple[MsGraphMailMessageChange, ...]:
    last_by_id: dict[str, MsGraphMailMessageChange] = {}
    order: list[str] = []
    for item in items:
        if item.remote_id not in last_by_id:
            order.append(item.remote_id)
        else:
            order.remove(item.remote_id)
            order.append(item.remote_id)
        last_by_id[item.remote_id] = item
    return tuple(last_by_id[remote_id] for remote_id in order)


def _validate_delta_limit(limit: object) -> int:
    if type(limit) is not int:
        raise IntegrationConfigurationError(_INVALID_MAIL_MESSAGES_REQUEST)
    if limit < _MIN_DELTA_LIMIT or limit > _MAX_DELTA_LIMIT:
        raise IntegrationConfigurationError(_INVALID_MAIL_MESSAGES_REQUEST)
    return limit


def _validate_mail_messages_request_input(
    *,
    mailbox_user_id: object,
    folder_id: object,
) -> tuple[str, str]:
    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(mailbox_user_id)
        validated_folder_id = validate_msgraph_mail_folder_id(folder_id)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MAIL_MESSAGES_REQUEST) from None
    return validated_mailbox_user_id, validated_folder_id


def _build_mail_message_delta_page(
    *,
    raw_items: tuple[dict[str, object], ...],
    continuation: MsGraphKnowledgeContinuation,
    expected_mailbox_user_id: str,
    expected_folder_id: str,
) -> MsGraphMailMessageDeltaPage:
    parsed_items = tuple(
        parse_msgraph_mail_message_change(
            raw_item,
            expected_mailbox_user_id=expected_mailbox_user_id,
            expected_folder_id=expected_folder_id,
        )
        for raw_item in raw_items
    )
    deduplicated = _deduplicate_message_changes(parsed_items)
    return _safe_construct_message_delta_page(items=deduplicated, continuation=continuation)


class MsGraphMailMessagesReader:
    """Mailbox folder message metadata delta reader over the shared Graph knowledge transport."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        transport: MsGraphKnowledgeTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    def read_delta_page(
        self,
        *,
        mailbox_user_id: str,
        folder_id: str,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphMailMessageDeltaPage:
        validated_mailbox_user_id, validated_folder_id = _validate_mail_messages_request_input(
            mailbox_user_id=mailbox_user_id,
            folder_id=folder_id,
        )
        validated_limit = _validate_delta_limit(limit)

        if continuation is None:
            quoted_mailbox_user_id = quote(validated_mailbox_user_id, safe="")
            quoted_folder_id = quote(validated_folder_id, safe="")
            path = (
                f"/users/{quoted_mailbox_user_id}/mailFolders/"
                f"{quoted_folder_id}/messages/delta"
            )
            payload = self._transport.get_initial_json(
                path=path,
                params={
                    "$top": validated_limit,
                    "$select": _MAIL_MESSAGES_SELECT,
                },
                headers=_IMMUTABLE_ID_HEADERS,
                not_found_is_dependency=True,
            )
        else:
            validated_continuation = validate_msgraph_mail_messages_delta_continuation(
                continuation,
                mailbox_user_id=validated_mailbox_user_id,
                folder_id=validated_folder_id,
                graph_base_url=self._config.graph_base_url,
            )
            payload = self._transport.get_continuation_json(
                continuation=validated_continuation,
                headers=_IMMUTABLE_ID_HEADERS,
                not_found_is_dependency=True,
            )

        collection_page = parse_msgraph_collection_page(
            payload,
            graph_base_url=self._config.graph_base_url,
            delta_mode=True,
        )
        if collection_page.continuation is None:
            raise ValueError(_MALFORMED_MAIL_MESSAGES_RESPONSE) from None

        return _build_mail_message_delta_page(
            raw_items=collection_page.items,
            continuation=collection_page.continuation,
            expected_mailbox_user_id=validated_mailbox_user_id,
            expected_folder_id=validated_folder_id,
        )
