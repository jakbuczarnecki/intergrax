# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Teams Chat knowledge-read: caller-visible chat inventory."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Protocol, runtime_checkable
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

MSGRAPH_TEAMS_CHAT_SOURCE_KIND = "teams_chat"

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_CHATS_RESPONSE = "unexpected Microsoft Graph Teams chats response"
_INVALID_CHATS_REQUEST = "invalid Microsoft Graph Teams chats request"
_INVALID_CHATS_CONTINUATION = "invalid Microsoft Graph Teams chats continuation"
_MAX_MSGRAPH_ID_LEN = 4096
_MAX_TOPIC_LEN = 4096
_MAX_TENANT_ID_LEN = 2048
_MAX_ENUM_STRING_LEN = 1024
_MIN_CHAT_LIMIT = 1
_MAX_CHAT_LIMIT = 50
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_PREFER_UNKNOWN_ENUM = {"Prefer": "include-unknown-enum-members"}


class MsGraphTeamsChatType(StrEnum):
    ONE_ON_ONE = "one_on_one"
    GROUP = "group"
    MEETING = "meeting"
    UNKNOWN = "unknown"


class MsGraphTeamsChatMigrationMode(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    UNKNOWN = "unknown"


_CHAT_TYPE_MAP: dict[str, MsGraphTeamsChatType] = {
    "oneonone": MsGraphTeamsChatType.ONE_ON_ONE,
    "group": MsGraphTeamsChatType.GROUP,
    "meeting": MsGraphTeamsChatType.MEETING,
}

_MIGRATION_MODE_MAP: dict[str, MsGraphTeamsChatMigrationMode] = {
    "inprogress": MsGraphTeamsChatMigrationMode.IN_PROGRESS,
    "completed": MsGraphTeamsChatMigrationMode.COMPLETED,
}


def _validate_msgraph_opaque_id(value: object, *, error: str = _MALFORMED_CHATS_RESPONSE) -> str:
    if not isinstance(value, str):
        raise ValueError(error)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(error)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(error)
    if len(trimmed) > _MAX_MSGRAPH_ID_LEN:
        raise ValueError(error)
    return trimmed


def validate_msgraph_teams_chat_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def validate_msgraph_teams_chat_member_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def validate_msgraph_teams_chat_message_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def validate_msgraph_teams_chat_attachment_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def validate_msgraph_teams_chat_hosted_content_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def _validate_enum_string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    if len(trimmed) > _MAX_ENUM_STRING_LEN:
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    return trimmed


def _map_chat_type(value: object) -> MsGraphTeamsChatType:
    trimmed = _validate_enum_string(value)
    normalized = trimmed.lower().replace("_", "")
    return _CHAT_TYPE_MAP.get(normalized, MsGraphTeamsChatType.UNKNOWN)


def _map_migration_mode(value: object) -> MsGraphTeamsChatMigrationMode:
    trimmed = _validate_enum_string(value)
    normalized = trimmed.lower().replace("_", "")
    return _MIGRATION_MODE_MAP.get(normalized, MsGraphTeamsChatMigrationMode.UNKNOWN)


def _validate_exact_bool(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    return value


def _parse_timezone_aware_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    if trimmed.endswith("Z"):
        trimmed = f"{trimmed[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(trimmed)
    except ValueError:
        raise ValueError(_MALFORMED_CHATS_RESPONSE) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    return parsed.astimezone(timezone.utc)


def _normalize_model_datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    return value.astimezone(timezone.utc)


def _validate_optional_topic(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        return None
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    if len(trimmed) > _MAX_TOPIC_LEN:
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    return trimmed


def _validate_optional_tenant_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    if len(trimmed) > _MAX_TENANT_ID_LEN:
        raise ValueError(_MALFORMED_CHATS_RESPONSE)
    return trimmed


class MsGraphTeamsChat(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    remote_id: str

    chat_type: MsGraphTeamsChatType

    topic: str | None = Field(default=None, repr=False)
    tenant_id: str | None = Field(default=None, repr=False)

    created_at: datetime
    last_updated_at: datetime
    original_created_at: datetime | None = None

    is_hidden_for_all_members: bool

    migration_mode: MsGraphTeamsChatMigrationMode | None = None

    has_online_meeting_info: bool

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("remote_id", mode="before")
    @classmethod
    def _validate_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_chat_id(value)

    @field_validator("topic", mode="before")
    @classmethod
    def _validate_topic(cls, value: object) -> str | None:
        return _validate_optional_topic(value)

    @field_validator("tenant_id", mode="before")
    @classmethod
    def _validate_tenant_id(cls, value: object) -> str | None:
        return _validate_optional_tenant_id(value)

    @field_validator("created_at", "last_updated_at", mode="before")
    @classmethod
    def _validate_required_datetimes(cls, value: object) -> datetime:
        return _normalize_model_datetime(value)

    @field_validator("original_created_at", mode="before")
    @classmethod
    def _validate_original_created_at(cls, value: object) -> datetime | None:
        if value is None:
            return None
        return _normalize_model_datetime(value)

    @field_validator("is_hidden_for_all_members", "has_online_meeting_info", mode="before")
    @classmethod
    def _validate_bools(cls, value: object) -> bool:
        return _validate_exact_bool(value)

    @model_validator(mode="after")
    def _validate_datetime_order(self) -> MsGraphTeamsChat:
        if self.last_updated_at < self.created_at:
            raise ValueError(_MALFORMED_CHATS_RESPONSE)
        return self


class MsGraphTeamsChatPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    items: tuple[MsGraphTeamsChat, ...]

    continuation: MsGraphKnowledgeContinuation | None = Field(default=None, repr=False)

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, value: object) -> tuple[MsGraphTeamsChat, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_CHATS_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphTeamsChat):
                raise ValueError(_MALFORMED_CHATS_RESPONSE)
        return value

    @field_validator("continuation", mode="before")
    @classmethod
    def _validate_continuation(cls, value: object) -> MsGraphKnowledgeContinuation | None:
        if value is None:
            return None
        if not isinstance(value, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_CHATS_RESPONSE)
        if value.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError(_MALFORMED_CHATS_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_page_shape(self) -> MsGraphTeamsChatPage:
        remote_ids = [item.remote_id for item in self.items]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError(_MALFORMED_CHATS_RESPONSE)
        for item in self.items:
            if item.mailbox_user_id != self.mailbox_user_id:
                raise ValueError(_MALFORMED_CHATS_RESPONSE)
        return self

    @property
    def has_more(self) -> bool:
        return self.continuation is not None


def _safe_construct_chat(**kwargs: object) -> MsGraphTeamsChat:
    try:
        return MsGraphTeamsChat(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CHATS_RESPONSE) from None


def _safe_construct_chat_page(**kwargs: object) -> MsGraphTeamsChatPage:
    try:
        return MsGraphTeamsChatPage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CHATS_RESPONSE) from None


class MsGraphTeamsChatChanged(IntegrationDependencyError):
    def __init__(self) -> None:
        super().__init__("Microsoft Graph Teams chat changed during read")


def parse_msgraph_teams_chat(
    payload: object,
    *,
    expected_mailbox_user_id: str,
) -> MsGraphTeamsChat:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_CHATS_RESPONSE) from None

    required_keys = (
        "id",
        "chatType",
        "createdDateTime",
        "lastUpdatedDateTime",
        "isHiddenForAllMembers",
    )
    for key in required_keys:
        if key not in payload:
            raise ValueError(_MALFORMED_CHATS_RESPONSE) from None

    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(expected_mailbox_user_id)
        remote_id = validate_msgraph_teams_chat_id(payload.get("id"))
        chat_type = _map_chat_type(payload.get("chatType"))
        created_at = _parse_timezone_aware_datetime(payload.get("createdDateTime"))
        last_updated_at = _parse_timezone_aware_datetime(payload.get("lastUpdatedDateTime"))
        is_hidden = _validate_exact_bool(payload.get("isHiddenForAllMembers"))
    except ValueError:
        raise ValueError(_MALFORMED_CHATS_RESPONSE) from None

    topic: str | None = None
    if "topic" in payload:
        try:
            topic = _validate_optional_topic(payload.get("topic"))
        except ValueError:
            raise ValueError(_MALFORMED_CHATS_RESPONSE) from None

    tenant_id: str | None = None
    if "tenantId" in payload:
        try:
            tenant_id = _validate_optional_tenant_id(payload.get("tenantId"))
        except ValueError:
            raise ValueError(_MALFORMED_CHATS_RESPONSE) from None

    original_created_at: datetime | None = None
    if "originalCreatedDateTime" in payload and payload.get("originalCreatedDateTime") is not None:
        try:
            original_created_at = _parse_timezone_aware_datetime(
                payload.get("originalCreatedDateTime")
            )
        except ValueError:
            raise ValueError(_MALFORMED_CHATS_RESPONSE) from None

    migration_mode: MsGraphTeamsChatMigrationMode | None = None
    if "migrationMode" in payload and payload.get("migrationMode") is not None:
        try:
            migration_mode = _map_migration_mode(payload.get("migrationMode"))
        except ValueError:
            raise ValueError(_MALFORMED_CHATS_RESPONSE) from None

    if "onlineMeetingInfo" not in payload or payload.get("onlineMeetingInfo") is None:
        has_online_meeting_info = False
    elif isinstance(payload.get("onlineMeetingInfo"), dict):
        has_online_meeting_info = True
    else:
        raise ValueError(_MALFORMED_CHATS_RESPONSE) from None

    return _safe_construct_chat(
        mailbox_user_id=validated_mailbox_user_id,
        remote_id=remote_id,
        chat_type=chat_type,
        topic=topic,
        tenant_id=tenant_id,
        created_at=created_at,
        last_updated_at=last_updated_at,
        original_created_at=original_created_at,
        is_hidden_for_all_members=is_hidden,
        migration_mode=migration_mode,
        has_online_meeting_info=has_online_meeting_info,
    )


def validate_msgraph_teams_chat(value: object) -> MsGraphTeamsChat:
    if not isinstance(value, MsGraphTeamsChat):
        raise ValueError(_MALFORMED_CHATS_RESPONSE) from None
    try:
        return MsGraphTeamsChat.model_validate(value.model_dump(mode="python"))
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CHATS_RESPONSE) from None


def validate_msgraph_teams_chat_page(
    value: object,
    *,
    mailbox_user_id: str,
    graph_base_url: str,
) -> MsGraphTeamsChatPage:
    if not isinstance(value, MsGraphTeamsChatPage):
        raise ValueError(_MALFORMED_CHATS_RESPONSE) from None

    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(mailbox_user_id)
    except ValueError:
        raise ValueError(_MALFORMED_CHATS_RESPONSE) from None

    try:
        raw_mailbox_user_id = value.mailbox_user_id
        raw_items = value.items
        raw_continuation = value.continuation
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_CHATS_RESPONSE) from None

    try:
        validated_page_mailbox = validate_msgraph_mailbox_user_id(raw_mailbox_user_id)
    except ValueError:
        raise ValueError(_MALFORMED_CHATS_RESPONSE) from None

    if validated_page_mailbox != validated_mailbox_user_id:
        raise ValueError(_MALFORMED_CHATS_RESPONSE) from None

    if type(raw_items) is not tuple:
        raise ValueError(_MALFORMED_CHATS_RESPONSE) from None

    validated_items: list[MsGraphTeamsChat] = []
    for item in raw_items:
        if not isinstance(item, MsGraphTeamsChat):
            raise ValueError(_MALFORMED_CHATS_RESPONSE) from None
        validated_item = validate_msgraph_teams_chat(item)
        if validated_item.mailbox_user_id != validated_mailbox_user_id:
            raise ValueError(_MALFORMED_CHATS_RESPONSE) from None
        validated_items.append(validated_item)

    remote_ids = [item.remote_id for item in validated_items]
    if len(remote_ids) != len(set(remote_ids)):
        raise ValueError(_MALFORMED_CHATS_RESPONSE) from None

    validated_continuation: MsGraphKnowledgeContinuation | None = None
    if raw_continuation is not None:
        if not isinstance(raw_continuation, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_CHATS_RESPONSE) from None
        try:
            validated_continuation = validate_msgraph_teams_chats_continuation(
                raw_continuation,
                mailbox_user_id=validated_mailbox_user_id,
                graph_base_url=graph_base_url,
            )
        except IntegrationConfigurationError:
            raise ValueError(_MALFORMED_CHATS_RESPONSE) from None

    try:
        return MsGraphTeamsChatPage(
            mailbox_user_id=validated_mailbox_user_id,
            items=tuple(validated_items),
            continuation=validated_continuation,
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CHATS_RESPONSE) from None


def _graph_base_path(graph_base_url: str) -> str:
    parsed_base = urlparse(graph_base_url)
    return parsed_base.path.rstrip("/") or "/"


def _decode_odata_literal(literal: str) -> str:
    return literal.replace("''", "'")


def _extract_chats_path(
    path: str,
    *,
    graph_base_path: str,
) -> str | None:
    normalized = path.rstrip("/") or "/"
    base = graph_base_path.rstrip("/") or "/"

    slash_match = re.fullmatch(
        rf"^{re.escape(base)}/users/([^/]+)/chats$",
        normalized,
        re.IGNORECASE,
    )
    if slash_match is not None:
        mailbox_segment = slash_match.group(1)
        if not mailbox_segment:
            return None
        return unquote(mailbox_segment)

    odata_match = re.fullmatch(
        rf"^{re.escape(base)}/users\('((?:[^']|'')*)'\)/chats$",
        normalized,
        re.IGNORECASE,
    )
    if odata_match is not None:
        mailbox_literal = odata_match.group(1)
        if not mailbox_literal:
            return None
        decoded = unquote(mailbox_literal)
        return _decode_odata_literal(decoded)

    return None


def validate_msgraph_teams_chats_continuation(
    continuation: object,
    *,
    mailbox_user_id: str,
    graph_base_url: str,
) -> MsGraphKnowledgeContinuation:
    if not isinstance(continuation, MsGraphKnowledgeContinuation):
        raise IntegrationConfigurationError(_INVALID_CHATS_CONTINUATION) from None

    try:
        revalidated = MsGraphKnowledgeContinuation.model_validate(
            continuation.model_dump(mode="python")
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise IntegrationConfigurationError(_INVALID_CHATS_CONTINUATION) from None

    if revalidated.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
        raise IntegrationConfigurationError(_INVALID_CHATS_CONTINUATION) from None

    try:
        validated_url = validate_msgraph_continuation_url(
            revalidated.url,
            graph_base_url=graph_base_url,
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_CHATS_CONTINUATION) from None

    parsed = urlparse(validated_url)
    extracted_mailbox = _extract_chats_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted_mailbox is None:
        raise IntegrationConfigurationError(_INVALID_CHATS_CONTINUATION) from None

    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(mailbox_user_id)
        validated_extracted_mailbox = validate_msgraph_mailbox_user_id(extracted_mailbox)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_CHATS_CONTINUATION) from None

    if validated_extracted_mailbox != validated_mailbox_user_id:
        raise IntegrationConfigurationError(_INVALID_CHATS_CONTINUATION) from None

    return revalidated


def _validate_chat_limit(limit: object) -> int:
    if type(limit) is not int:
        raise IntegrationConfigurationError(_INVALID_CHATS_REQUEST)
    if limit < _MIN_CHAT_LIMIT or limit > _MAX_CHAT_LIMIT:
        raise IntegrationConfigurationError(_INVALID_CHATS_REQUEST)
    return limit


def _compare_chat_observation(
    payload: dict[str, object],
    *,
    chat: MsGraphTeamsChat,
) -> None:
    try:
        response_id = validate_msgraph_teams_chat_id(payload.get("id"))
        response_type = _map_chat_type(payload.get("chatType"))
        response_last_updated = _parse_timezone_aware_datetime(
            payload.get("lastUpdatedDateTime")
        )
    except ValueError:
        raise MsGraphTeamsChatChanged() from None
    if (
        response_id != chat.remote_id
        or response_type is not chat.chat_type
        or response_last_updated != chat.last_updated_at
    ):
        raise MsGraphTeamsChatChanged() from None


def read_and_validate_current_teams_chat_observation(
    *,
    chat: MsGraphTeamsChat,
    transport: MsGraphKnowledgeTransport,
) -> MsGraphTeamsChat:
    validated_chat = validate_msgraph_teams_chat(chat)
    quoted_mailbox = quote(validated_chat.mailbox_user_id, safe="")
    quoted_chat = quote(validated_chat.remote_id, safe="")
    path = f"/users/{quoted_mailbox}/chats/{quoted_chat}"
    payload = transport.get_initial_json(
        path=path,
        headers=_PREFER_UNKNOWN_ENUM,
        not_found_is_dependency=True,
    )
    _compare_chat_observation(payload, chat=validated_chat)
    observed = parse_msgraph_teams_chat(
        payload,
        expected_mailbox_user_id=validated_chat.mailbox_user_id,
    )
    if (
        observed.remote_id != validated_chat.remote_id
        or observed.chat_type is not validated_chat.chat_type
        or observed.last_updated_at != validated_chat.last_updated_at
    ):
        raise MsGraphTeamsChatChanged() from None
    return observed


@runtime_checkable
class MsGraphTeamsChatsReadClient(Protocol):
    def read_teams_chats_page(
        self,
        *,
        mailbox_user_id: str,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphTeamsChatPage:
        ...


class MsGraphTeamsChatsReader:
    """Caller-visible Teams chat inventory reader."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        transport: MsGraphKnowledgeTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    def read_chats_page(
        self,
        *,
        mailbox_user_id: str,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphTeamsChatPage:
        try:
            validated_mailbox_user_id = validate_msgraph_mailbox_user_id(mailbox_user_id)
        except ValueError:
            raise IntegrationConfigurationError(_INVALID_CHATS_REQUEST) from None

        validated_limit = _validate_chat_limit(limit)

        if continuation is None:
            quoted_mailbox = quote(validated_mailbox_user_id, safe="")
            path = f"/users/{quoted_mailbox}/chats"
            payload = self._transport.get_initial_json(
                path=path,
                params={"$top": validated_limit},
                headers=_PREFER_UNKNOWN_ENUM,
            )
        else:
            validated_continuation = validate_msgraph_teams_chats_continuation(
                continuation,
                mailbox_user_id=validated_mailbox_user_id,
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
            parse_msgraph_teams_chat(
                raw_item,
                expected_mailbox_user_id=validated_mailbox_user_id,
            )
            for raw_item in collection_page.items
        )
        return validate_msgraph_teams_chat_page(
            _safe_construct_chat_page(
                mailbox_user_id=validated_mailbox_user_id,
                items=parsed_items,
                continuation=collection_page.continuation,
            ),
            mailbox_user_id=validated_mailbox_user_id,
            graph_base_url=self._config.graph_base_url,
        )
