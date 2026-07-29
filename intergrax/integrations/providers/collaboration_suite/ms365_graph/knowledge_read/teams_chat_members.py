# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Teams Chat knowledge-read: complete member roster snapshots."""

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
    validate_msgraph_mailbox_user_id,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
    MsGraphTeamsChat,
    read_and_validate_current_teams_chat_observation,
    validate_msgraph_teams_chat,
    validate_msgraph_teams_chat_id,
    validate_msgraph_teams_chat_member_id,
)

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_MEMBERS_RESPONSE = "unexpected Microsoft Graph Teams chat members response"
_INVALID_MEMBERS_CONTINUATION = "invalid Microsoft Graph Teams chat members continuation"
_MAX_DISPLAY_NAME_LEN = 1024
_MAX_EMAIL_LEN = 2048
_MAX_TENANT_ID_LEN = 2048
_MAX_ODATA_TYPE_LEN = 1024
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_PREFER_UNKNOWN_ENUM = {"Prefer": "include-unknown-enum-members"}

_ODATA_AAD_USER = "#microsoft.graph.aadUserConversationMember"
_ODATA_ANONYMOUS_GUEST = "#microsoft.graph.anonymousGuestConversationMember"
_ODATA_MICROSOFT_ACCOUNT = "#microsoft.graph.microsoftAccountUserConversationMember"
_ODATA_SKYPE_USER = "#microsoft.graph.skypeUserConversationMember"
_ODATA_SKYPE_FOR_BUSINESS = "#microsoft.graph.skypeForBusinessUserConversationMember"
_ODATA_ACS_USER = "#microsoft.graph.azureCommunicationServicesUserConversationMember"


class MsGraphTeamsChatMemberKind(StrEnum):
    AAD_USER = "aad_user"
    ANONYMOUS_GUEST = "anonymous_guest"
    MICROSOFT_ACCOUNT = "microsoft_account"
    SKYPE_USER = "skype_user"
    SKYPE_FOR_BUSINESS_USER = "skype_for_business_user"
    AZURE_COMMUNICATION_SERVICES_USER = "azure_communication_services_user"
    UNKNOWN = "unknown"


class MsGraphTeamsChatMemberRole(StrEnum):
    OWNER = "owner"
    GUEST = "guest"
    UNKNOWN = "unknown"


_MEMBER_KIND_MAP: dict[str, MsGraphTeamsChatMemberKind] = {
    _ODATA_AAD_USER: MsGraphTeamsChatMemberKind.AAD_USER,
    _ODATA_ANONYMOUS_GUEST: MsGraphTeamsChatMemberKind.ANONYMOUS_GUEST,
    _ODATA_MICROSOFT_ACCOUNT: MsGraphTeamsChatMemberKind.MICROSOFT_ACCOUNT,
    _ODATA_SKYPE_USER: MsGraphTeamsChatMemberKind.SKYPE_USER,
    _ODATA_SKYPE_FOR_BUSINESS: MsGraphTeamsChatMemberKind.SKYPE_FOR_BUSINESS_USER,
    _ODATA_ACS_USER: MsGraphTeamsChatMemberKind.AZURE_COMMUNICATION_SERVICES_USER,
}

_ROLE_MAP: dict[str, MsGraphTeamsChatMemberRole] = {
    "owner": MsGraphTeamsChatMemberRole.OWNER,
    "guest": MsGraphTeamsChatMemberRole.GUEST,
}


def _validate_optional_trimmed_string(value: object, *, max_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        return None
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    if len(trimmed) > max_length:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    return trimmed


def _validate_optional_opaque_string(value: object, *, max_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    if len(trimmed) > max_length:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    return trimmed


def _parse_timezone_aware_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    if trimmed.endswith("Z"):
        trimmed = f"{trimmed[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(trimmed)
    except ValueError:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    return parsed.astimezone(timezone.utc)


def _normalize_model_datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    return value.astimezone(timezone.utc)


def _validate_unknown_odata_type(value: str) -> None:
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    if len(trimmed) > _MAX_ODATA_TYPE_LEN:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)


def _map_member_kind(odata_type: object) -> MsGraphTeamsChatMemberKind:
    if not isinstance(odata_type, str):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    trimmed = odata_type.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    mapped = _MEMBER_KIND_MAP.get(trimmed)
    if mapped is not None:
        return mapped
    _validate_unknown_odata_type(trimmed)
    return MsGraphTeamsChatMemberKind.UNKNOWN


def _map_member_role(value: object) -> MsGraphTeamsChatMemberRole:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    if len(trimmed) > _MAX_ODATA_TYPE_LEN:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    normalized = trimmed.lower()
    return _ROLE_MAP.get(normalized, MsGraphTeamsChatMemberRole.UNKNOWN)


def _deduplicate_roles(
    roles: list[MsGraphTeamsChatMemberRole],
) -> tuple[MsGraphTeamsChatMemberRole, ...]:
    seen: set[MsGraphTeamsChatMemberRole] = set()
    result: list[MsGraphTeamsChatMemberRole] = []
    for role in roles:
        if role in seen:
            continue
        seen.add(role)
        result.append(role)
    return tuple(result)


class MsGraphTeamsChatMember(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    chat_remote_id: str
    chat_revision: datetime

    remote_id: str
    member_kind: MsGraphTeamsChatMemberKind

    provider_user_id: str | None = None
    tenant_id: str | None = Field(default=None, repr=False)

    display_name: str | None = Field(default=None, repr=False)
    email: str | None = Field(default=None, repr=False)

    roles: tuple[MsGraphTeamsChatMemberRole, ...] = ()

    visible_history_start_at: datetime | None = None

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("chat_remote_id", mode="before")
    @classmethod
    def _validate_chat_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_chat_id(value)

    @field_validator("chat_revision", mode="before")
    @classmethod
    def _validate_chat_revision(cls, value: object) -> datetime:
        return _normalize_model_datetime(value)

    @field_validator("remote_id", mode="before")
    @classmethod
    def _validate_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_chat_member_id(value)

    @field_validator("provider_user_id", mode="before")
    @classmethod
    def _validate_provider_user_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return validate_msgraph_teams_chat_member_id(value)

    @field_validator("tenant_id", mode="before")
    @classmethod
    def _validate_tenant_id(cls, value: object) -> str | None:
        return _validate_optional_opaque_string(value, max_length=_MAX_TENANT_ID_LEN)

    @field_validator("display_name", mode="before")
    @classmethod
    def _validate_display_name(cls, value: object) -> str | None:
        return _validate_optional_trimmed_string(value, max_length=_MAX_DISPLAY_NAME_LEN)

    @field_validator("email", mode="before")
    @classmethod
    def _validate_email(cls, value: object) -> str | None:
        return _validate_optional_trimmed_string(value, max_length=_MAX_EMAIL_LEN)

    @field_validator("roles", mode="before")
    @classmethod
    def _validate_roles(cls, value: object) -> tuple[MsGraphTeamsChatMemberRole, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphTeamsChatMemberRole):
                raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
        return _deduplicate_roles(list(value))

    @field_validator("visible_history_start_at", mode="before")
    @classmethod
    def _validate_visible_history(cls, value: object) -> datetime | None:
        if value is None:
            return None
        return _normalize_model_datetime(value)


class MsGraphTeamsChatMemberPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    chat_remote_id: str
    chat_revision: datetime

    items: tuple[MsGraphTeamsChatMember, ...]

    continuation: MsGraphKnowledgeContinuation | None = Field(default=None, repr=False)

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("chat_remote_id", mode="before")
    @classmethod
    def _validate_chat_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_chat_id(value)

    @field_validator("chat_revision", mode="before")
    @classmethod
    def _validate_chat_revision(cls, value: object) -> datetime:
        return _normalize_model_datetime(value)

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, value: object) -> tuple[MsGraphTeamsChatMember, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphTeamsChatMember):
                raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
        return value

    @field_validator("continuation", mode="before")
    @classmethod
    def _validate_continuation(cls, value: object) -> MsGraphKnowledgeContinuation | None:
        if value is None:
            return None
        if not isinstance(value, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
        if value.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_page_shape(self) -> MsGraphTeamsChatMemberPage:
        remote_ids = [item.remote_id for item in self.items]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
        for item in self.items:
            if item.mailbox_user_id != self.mailbox_user_id:
                raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
            if item.chat_remote_id != self.chat_remote_id:
                raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
            if item.chat_revision != self.chat_revision:
                raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
        return self


def _safe_construct_member(**kwargs: object) -> MsGraphTeamsChatMember:
    try:
        return MsGraphTeamsChatMember(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None


def _safe_construct_member_page(**kwargs: object) -> MsGraphTeamsChatMemberPage:
    try:
        return MsGraphTeamsChatMemberPage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None


def parse_msgraph_teams_chat_member(
    payload: object,
    *,
    chat: MsGraphTeamsChat,
) -> MsGraphTeamsChatMember:
    validated_chat = validate_msgraph_teams_chat(chat)
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    required_keys = ("@odata.type", "id", "roles")
    for key in required_keys:
        if key not in payload:
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    try:
        member_kind = _map_member_kind(payload.get("@odata.type"))
        remote_id = validate_msgraph_teams_chat_member_id(payload.get("id"))
        raw_roles = payload.get("roles")
        if not isinstance(raw_roles, list):
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
        roles = _deduplicate_roles(
            [_map_member_role(role_value) for role_value in raw_roles]
        )
    except ValueError:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    provider_user_id: str | None = None
    if "userId" in payload and payload.get("userId") is not None:
        try:
            provider_user_id = validate_msgraph_teams_chat_member_id(payload.get("userId"))
        except ValueError:
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    tenant_id: str | None = None
    if "tenantId" in payload:
        try:
            tenant_id = _validate_optional_opaque_string(
                payload.get("tenantId"),
                max_length=_MAX_TENANT_ID_LEN,
            )
        except ValueError:
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    display_name: str | None = None
    if "displayName" in payload:
        try:
            display_name = _validate_optional_trimmed_string(
                payload.get("displayName"),
                max_length=_MAX_DISPLAY_NAME_LEN,
            )
        except ValueError:
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    email: str | None = None
    if "email" in payload:
        try:
            email = _validate_optional_trimmed_string(
                payload.get("email"),
                max_length=_MAX_EMAIL_LEN,
            )
        except ValueError:
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    visible_history_start_at: datetime | None = None
    if (
        "visibleHistoryStartDateTime" in payload
        and payload.get("visibleHistoryStartDateTime") is not None
    ):
        try:
            visible_history_start_at = _parse_timezone_aware_datetime(
                payload.get("visibleHistoryStartDateTime")
            )
        except ValueError:
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    return _safe_construct_member(
        mailbox_user_id=validated_chat.mailbox_user_id,
        chat_remote_id=validated_chat.remote_id,
        chat_revision=validated_chat.last_updated_at,
        remote_id=remote_id,
        member_kind=member_kind,
        provider_user_id=provider_user_id,
        tenant_id=tenant_id,
        display_name=display_name,
        email=email,
        roles=roles,
        visible_history_start_at=visible_history_start_at,
    )


def validate_msgraph_teams_chat_member(value: object) -> MsGraphTeamsChatMember:
    if not isinstance(value, MsGraphTeamsChatMember):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None
    try:
        return MsGraphTeamsChatMember.model_validate(value.model_dump(mode="python"))
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None


def validate_msgraph_teams_chat_member_page(
    value: object,
    *,
    chat: MsGraphTeamsChat,
    graph_base_url: str,
) -> MsGraphTeamsChatMemberPage:
    if not isinstance(value, MsGraphTeamsChatMemberPage):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    validated_chat = validate_msgraph_teams_chat(chat)

    try:
        raw_mailbox_user_id = value.mailbox_user_id
        raw_chat_remote_id = value.chat_remote_id
        raw_chat_revision = value.chat_revision
        raw_items = value.items
        raw_continuation = value.continuation
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    if raw_mailbox_user_id != validated_chat.mailbox_user_id:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None
    if raw_chat_remote_id != validated_chat.remote_id:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None
    if not isinstance(raw_chat_revision, datetime):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None
    if raw_chat_revision != validated_chat.last_updated_at:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    if type(raw_items) is not tuple:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    validated_items: list[MsGraphTeamsChatMember] = []
    for item in raw_items:
        if not isinstance(item, MsGraphTeamsChatMember):
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None
        validated_item = validate_msgraph_teams_chat_member(item)
        if (
            validated_item.mailbox_user_id != validated_chat.mailbox_user_id
            or validated_item.chat_remote_id != validated_chat.remote_id
            or validated_item.chat_revision != validated_chat.last_updated_at
        ):
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None
        validated_items.append(validated_item)

    remote_ids = [item.remote_id for item in validated_items]
    if len(remote_ids) != len(set(remote_ids)):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    validated_continuation: MsGraphKnowledgeContinuation | None = None
    if raw_continuation is not None:
        if not isinstance(raw_continuation, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None
        try:
            validated_continuation = validate_msgraph_teams_chat_members_continuation(
                raw_continuation,
                mailbox_user_id=validated_chat.mailbox_user_id,
                chat_id=validated_chat.remote_id,
                graph_base_url=graph_base_url,
            )
        except IntegrationConfigurationError:
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    try:
        return MsGraphTeamsChatMemberPage(
            mailbox_user_id=validated_chat.mailbox_user_id,
            chat_remote_id=validated_chat.remote_id,
            chat_revision=validated_chat.last_updated_at,
            items=tuple(validated_items),
            continuation=validated_continuation,
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None


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


def _extract_chat_members_path(
    path: str,
    *,
    graph_base_path: str,
) -> tuple[str, str] | None:
    normalized = path.rstrip("/") or "/"
    base = graph_base_path.rstrip("/") or "/"

    patterns: list[tuple[str, bool, bool]] = [
        (
            rf"^{re.escape(base)}/users/([^/]+)/chats/([^/]+)/members$",
            False,
            False,
        ),
        (
            rf"^{re.escape(base)}/users\('((?:[^']|'')*)'\)/chats\('((?:[^']|'')*)'\)/members$",
            True,
            True,
        ),
        (
            rf"^{re.escape(base)}/users/([^/]+)/chats\('((?:[^']|'')*)'\)/members$",
            False,
            True,
        ),
        (
            rf"^{re.escape(base)}/users\('((?:[^']|'')*)'\)/chats/([^/]+)/members$",
            True,
            False,
        ),
        (
            rf"^{re.escape(base)}/users/([^/]+)/chats/\('((?:[^']|'')*)'\)/members$",
            False,
            True,
        ),
    ]

    for pattern, mailbox_odata, chat_odata in patterns:
        match = re.fullmatch(pattern, normalized, re.IGNORECASE)
        if match is None:
            continue
        mailbox_segment = match.group(1)
        chat_segment = match.group(2)
        if not mailbox_segment or not chat_segment:
            return None
        mailbox_id = _decode_path_segment(mailbox_segment, odata_literal=mailbox_odata)
        chat_id = _decode_path_segment(chat_segment, odata_literal=chat_odata)
        return mailbox_id, chat_id

    return None


def validate_msgraph_teams_chat_members_continuation(
    continuation: object,
    *,
    mailbox_user_id: str,
    chat_id: str,
    graph_base_url: str,
) -> MsGraphKnowledgeContinuation:
    if not isinstance(continuation, MsGraphKnowledgeContinuation):
        raise IntegrationConfigurationError(_INVALID_MEMBERS_CONTINUATION) from None

    try:
        revalidated = MsGraphKnowledgeContinuation.model_validate(
            continuation.model_dump(mode="python")
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise IntegrationConfigurationError(_INVALID_MEMBERS_CONTINUATION) from None

    if revalidated.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
        raise IntegrationConfigurationError(_INVALID_MEMBERS_CONTINUATION) from None

    try:
        validated_url = validate_msgraph_continuation_url(
            revalidated.url,
            graph_base_url=graph_base_url,
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MEMBERS_CONTINUATION) from None

    parsed = urlparse(validated_url)
    extracted = _extract_chat_members_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted is None:
        raise IntegrationConfigurationError(_INVALID_MEMBERS_CONTINUATION) from None

    extracted_mailbox, extracted_chat = extracted
    try:
        validated_mailbox = validate_msgraph_mailbox_user_id(mailbox_user_id)
        validated_chat = validate_msgraph_teams_chat_id(chat_id)
        validated_extracted_mailbox = validate_msgraph_mailbox_user_id(extracted_mailbox)
        validated_extracted_chat = validate_msgraph_teams_chat_id(extracted_chat)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MEMBERS_CONTINUATION) from None

    if (
        validated_extracted_mailbox != validated_mailbox
        or validated_extracted_chat != validated_chat
    ):
        raise IntegrationConfigurationError(_INVALID_MEMBERS_CONTINUATION) from None

    return revalidated


@runtime_checkable
class MsGraphTeamsChatMembersReadClient(Protocol):
    def read_teams_chat_members_page(
        self,
        *,
        chat: MsGraphTeamsChat,
        continuation: MsGraphKnowledgeContinuation | None,
    ) -> MsGraphTeamsChatMemberPage:
        ...


class MsGraphTeamsChatMembersReader:
    """Complete Teams chat member roster reader with chat observation checks."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        transport: MsGraphKnowledgeTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    def read_members_page(
        self,
        *,
        chat: MsGraphTeamsChat,
        continuation: MsGraphKnowledgeContinuation | None,
    ) -> MsGraphTeamsChatMemberPage:
        validated_chat = validate_msgraph_teams_chat(chat)

        read_and_validate_current_teams_chat_observation(
            chat=validated_chat,
            transport=self._transport,
        )

        validated_continuation: MsGraphKnowledgeContinuation | None = None
        if continuation is not None:
            validated_continuation = validate_msgraph_teams_chat_members_continuation(
                continuation,
                mailbox_user_id=validated_chat.mailbox_user_id,
                chat_id=validated_chat.remote_id,
                graph_base_url=self._config.graph_base_url,
            )

        if validated_continuation is None:
            quoted_mailbox = quote(validated_chat.mailbox_user_id, safe="")
            quoted_chat = quote(validated_chat.remote_id, safe="")
            path = f"/users/{quoted_mailbox}/chats/{quoted_chat}/members"
            payload = self._transport.get_initial_json(
                path=path,
                headers=_PREFER_UNKNOWN_ENUM,
            )
        else:
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
            parse_msgraph_teams_chat_member(raw_item, chat=validated_chat)
            for raw_item in collection_page.items
        )
        page = validate_msgraph_teams_chat_member_page(
            _safe_construct_member_page(
                mailbox_user_id=validated_chat.mailbox_user_id,
                chat_remote_id=validated_chat.remote_id,
                chat_revision=validated_chat.last_updated_at,
                items=parsed_items,
                continuation=collection_page.continuation,
            ),
            chat=validated_chat,
            graph_base_url=self._config.graph_base_url,
        )

        read_and_validate_current_teams_chat_observation(
            chat=validated_chat,
            transport=self._transport,
        )
        return page
