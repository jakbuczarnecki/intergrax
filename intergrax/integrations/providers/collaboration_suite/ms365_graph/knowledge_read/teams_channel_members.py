# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Teams Channel knowledge-read: effective member roster snapshots."""

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
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (
    MsGraphTeamsChannel,
    MsGraphTeamsChannelChanged,
    read_and_validate_current_teams_channel_observation,
    validate_msgraph_teams_channel,
    validate_msgraph_teams_channel_id,
    validate_msgraph_teams_channel_member_id,
    validate_msgraph_teams_team_id,
)

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_MEMBERS_RESPONSE = "unexpected Microsoft Graph Teams channel members response"
_INVALID_MEMBERS_CONTINUATION = "invalid Microsoft Graph Teams channel members continuation"
_MAX_DISPLAY_NAME_LEN = 1024
_MAX_EMAIL_LEN = 2048
_MAX_TENANT_ID_LEN = 2048
_MAX_ODATA_TYPE_LEN = 1024
_MAX_MEMBERSHIP_URL_LEN = 8192
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_PREFER_UNKNOWN_ENUM = {"Prefer": "include-unknown-enum-members"}

_ODATA_AAD_USER = "#microsoft.graph.aadUserConversationMember"
_ODATA_ANONYMOUS_GUEST = "#microsoft.graph.anonymousGuestConversationMember"
_ODATA_MICROSOFT_ACCOUNT = "#microsoft.graph.microsoftAccountUserConversationMember"
_ODATA_SKYPE_USER = "#microsoft.graph.skypeUserConversationMember"
_ODATA_SKYPE_FOR_BUSINESS = "#microsoft.graph.skypeForBusinessUserConversationMember"
_ODATA_ACS_USER = "#microsoft.graph.azureCommunicationServicesUserConversationMember"

_ORIGINAL_SOURCE_MEMBERSHIP_URL_KEY = "@microsoft.graph.originalSourceMembershipUrl"
_INDIRECT_MEMBER_KEY = "@microsoft.graph.isIndirectMember"


class MsGraphTeamsChannelMemberKind(StrEnum):
    AAD_USER = "aad_user"
    ANONYMOUS_GUEST = "anonymous_guest"
    MICROSOFT_ACCOUNT = "microsoft_account"
    SKYPE_USER = "skype_user"
    SKYPE_FOR_BUSINESS_USER = "skype_for_business_user"
    AZURE_COMMUNICATION_SERVICES_USER = "azure_communication_services_user"
    UNKNOWN = "unknown"


class MsGraphTeamsChannelMemberRole(StrEnum):
    OWNER = "owner"
    GUEST = "guest"
    UNKNOWN = "unknown"


_MEMBER_KIND_MAP: dict[str, MsGraphTeamsChannelMemberKind] = {
    _ODATA_AAD_USER: MsGraphTeamsChannelMemberKind.AAD_USER,
    _ODATA_ANONYMOUS_GUEST: MsGraphTeamsChannelMemberKind.ANONYMOUS_GUEST,
    _ODATA_MICROSOFT_ACCOUNT: MsGraphTeamsChannelMemberKind.MICROSOFT_ACCOUNT,
    _ODATA_SKYPE_USER: MsGraphTeamsChannelMemberKind.SKYPE_USER,
    _ODATA_SKYPE_FOR_BUSINESS: MsGraphTeamsChannelMemberKind.SKYPE_FOR_BUSINESS_USER,
    _ODATA_ACS_USER: MsGraphTeamsChannelMemberKind.AZURE_COMMUNICATION_SERVICES_USER,
}

_ROLE_MAP: dict[str, MsGraphTeamsChannelMemberRole] = {
    "owner": MsGraphTeamsChannelMemberRole.OWNER,
    "guest": MsGraphTeamsChannelMemberRole.GUEST,
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


def _validate_exact_bool(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    return value


def _validate_https_membership_url(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    if "\x00" in value:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    if len(value) > _MAX_MEMBERSHIP_URL_LEN:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    parsed = urlparse(value)
    if parsed.scheme != "https":
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    if not parsed.hostname:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    if parsed.username or parsed.password:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    return value


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


def _map_member_kind(odata_type: object) -> MsGraphTeamsChannelMemberKind:
    if not isinstance(odata_type, str):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    trimmed = odata_type.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
    mapped = _MEMBER_KIND_MAP.get(trimmed)
    if mapped is not None:
        return mapped
    _validate_unknown_odata_type(trimmed)
    return MsGraphTeamsChannelMemberKind.UNKNOWN


def _map_member_role(value: object) -> MsGraphTeamsChannelMemberRole:
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
    return _ROLE_MAP.get(normalized, MsGraphTeamsChannelMemberRole.UNKNOWN)


def _deduplicate_roles(
    roles: list[MsGraphTeamsChannelMemberRole],
) -> tuple[MsGraphTeamsChannelMemberRole, ...]:
    seen: set[MsGraphTeamsChannelMemberRole] = set()
    result: list[MsGraphTeamsChannelMemberRole] = []
    for role in roles:
        if role in seen:
            continue
        seen.add(role)
        result.append(role)
    return tuple(result)


class MsGraphTeamsChannelMember(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    team_remote_id: str
    channel_remote_id: str

    remote_id: str
    member_kind: MsGraphTeamsChannelMemberKind

    provider_user_id: str | None = None
    tenant_id: str | None = Field(default=None, repr=False)

    display_name: str | None = Field(default=None, repr=False)
    email: str | None = Field(default=None, repr=False)

    roles: tuple[MsGraphTeamsChannelMemberRole, ...] = ()

    visible_history_start_at: datetime | None = None

    is_indirect_member: bool = False
    original_source_membership_url: str | None = Field(default=None, repr=False)

    @field_validator("team_remote_id", mode="before")
    @classmethod
    def _validate_team_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_team_id(value)

    @field_validator("channel_remote_id", mode="before")
    @classmethod
    def _validate_channel_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_id(value)

    @field_validator("remote_id", mode="before")
    @classmethod
    def _validate_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_member_id(value)

    @field_validator("provider_user_id", mode="before")
    @classmethod
    def _validate_provider_user_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return validate_msgraph_teams_channel_member_id(value)

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
    def _validate_roles(cls, value: object) -> tuple[MsGraphTeamsChannelMemberRole, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphTeamsChannelMemberRole):
                raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
        return _deduplicate_roles(list(value))

    @field_validator("visible_history_start_at", mode="before")
    @classmethod
    def _validate_visible_history(cls, value: object) -> datetime | None:
        if value is None:
            return None
        return _normalize_model_datetime(value)

    @field_validator("is_indirect_member", mode="before")
    @classmethod
    def _validate_is_indirect_member(cls, value: object) -> bool:
        return _validate_exact_bool(value)

    @field_validator("original_source_membership_url", mode="before")
    @classmethod
    def _validate_original_source_membership_url(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_https_membership_url(value)


class MsGraphTeamsChannelMemberPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    team_remote_id: str
    channel_remote_id: str

    items: tuple[MsGraphTeamsChannelMember, ...]

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
    def _validate_items(cls, value: object) -> tuple[MsGraphTeamsChannelMember, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphTeamsChannelMember):
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
    def _validate_page_shape(self) -> MsGraphTeamsChannelMemberPage:
        remote_ids = [item.remote_id for item in self.items]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
        for item in self.items:
            if item.team_remote_id != self.team_remote_id:
                raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
            if item.channel_remote_id != self.channel_remote_id:
                raise ValueError(_MALFORMED_MEMBERS_RESPONSE)
        return self


def _safe_construct_member(**kwargs: object) -> MsGraphTeamsChannelMember:
    try:
        return MsGraphTeamsChannelMember(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None


def _safe_construct_member_page(**kwargs: object) -> MsGraphTeamsChannelMemberPage:
    try:
        return MsGraphTeamsChannelMemberPage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None


def _parse_original_source_membership_url(payload: dict[str, object]) -> str | None:
    if _ORIGINAL_SOURCE_MEMBERSHIP_URL_KEY in payload:
        raw = payload.get(_ORIGINAL_SOURCE_MEMBERSHIP_URL_KEY)
        if raw is None:
            return None
        return _validate_https_membership_url(raw)
    if "originalSourceMembershipUrl" in payload:
        raw = payload.get("originalSourceMembershipUrl")
        if raw is None:
            return None
        return _validate_https_membership_url(raw)
    return None


def _parse_is_indirect_member(
    payload: dict[str, object],
    *,
    original_source_membership_url: str | None,
) -> bool:
    if "isIndirectMember" in payload:
        return _validate_exact_bool(payload.get("isIndirectMember"))
    if _INDIRECT_MEMBER_KEY in payload:
        return _validate_exact_bool(payload.get(_INDIRECT_MEMBER_KEY))
    return original_source_membership_url is not None


def parse_msgraph_teams_channel_member(
    payload: object,
    *,
    channel: MsGraphTeamsChannel,
) -> MsGraphTeamsChannelMember:
    validated_channel = validate_msgraph_teams_channel(channel)
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    required_keys = ("@odata.type", "id", "roles")
    for key in required_keys:
        if key not in payload:
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    try:
        member_kind = _map_member_kind(payload.get("@odata.type"))
        remote_id = validate_msgraph_teams_channel_member_id(payload.get("id"))
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
            provider_user_id = validate_msgraph_teams_channel_member_id(payload.get("userId"))
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

    try:
        original_source_membership_url = _parse_original_source_membership_url(payload)
        is_indirect_member = _parse_is_indirect_member(
            payload,
            original_source_membership_url=original_source_membership_url,
        )
    except ValueError:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    return _safe_construct_member(
        team_remote_id=validated_channel.team_remote_id,
        channel_remote_id=validated_channel.remote_id,
        remote_id=remote_id,
        member_kind=member_kind,
        provider_user_id=provider_user_id,
        tenant_id=tenant_id,
        display_name=display_name,
        email=email,
        roles=roles,
        visible_history_start_at=visible_history_start_at,
        is_indirect_member=is_indirect_member,
        original_source_membership_url=original_source_membership_url,
    )


def validate_msgraph_teams_channel_member(value: object) -> MsGraphTeamsChannelMember:
    if not isinstance(value, MsGraphTeamsChannelMember):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None
    try:
        return MsGraphTeamsChannelMember.model_validate(value.model_dump(mode="python"))
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None


def validate_msgraph_teams_channel_member_page(
    value: object,
    *,
    channel: MsGraphTeamsChannel,
    graph_base_url: str,
) -> MsGraphTeamsChannelMemberPage:
    if not isinstance(value, MsGraphTeamsChannelMemberPage):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    validated_channel = validate_msgraph_teams_channel(channel)

    try:
        raw_team_remote_id = value.team_remote_id
        raw_channel_remote_id = value.channel_remote_id
        raw_items = value.items
        raw_continuation = value.continuation
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    if raw_team_remote_id != validated_channel.team_remote_id:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None
    if raw_channel_remote_id != validated_channel.remote_id:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    if type(raw_items) is not tuple:
        raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    validated_items: list[MsGraphTeamsChannelMember] = []
    for item in raw_items:
        if not isinstance(item, MsGraphTeamsChannelMember):
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None
        validated_item = validate_msgraph_teams_channel_member(item)
        if (
            validated_item.team_remote_id != validated_channel.team_remote_id
            or validated_item.channel_remote_id != validated_channel.remote_id
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
            validated_continuation = validate_msgraph_teams_channel_members_continuation(
                raw_continuation,
                team_id=validated_channel.team_remote_id,
                channel_id=validated_channel.remote_id,
                graph_base_url=graph_base_url,
            )
        except IntegrationConfigurationError:
            raise ValueError(_MALFORMED_MEMBERS_RESPONSE) from None

    try:
        return MsGraphTeamsChannelMemberPage(
            team_remote_id=validated_channel.team_remote_id,
            channel_remote_id=validated_channel.remote_id,
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


def _extract_channel_all_members_path(
    path: str,
    *,
    graph_base_path: str,
) -> tuple[str, str] | None:
    normalized = path.rstrip("/") or "/"
    base = graph_base_path.rstrip("/") or "/"

    patterns: list[tuple[str, bool, bool]] = [
        (
            rf"^{re.escape(base)}/teams/([^/]+)/channels/([^/]+)/allMembers$",
            False,
            False,
        ),
        (
            rf"^{re.escape(base)}/teams\('((?:[^']|'')*)'\)/channels\('((?:[^']|'')*)'\)/allMembers$",
            True,
            True,
        ),
        (
            rf"^{re.escape(base)}/teams/([^/]+)/channels\('((?:[^']|'')*)'\)/allMembers$",
            False,
            True,
        ),
        (
            rf"^{re.escape(base)}/teams\('((?:[^']|'')*)'\)/channels/([^/]+)/allMembers$",
            True,
            False,
        ),
        (
            rf"^{re.escape(base)}/teams/([^/]+)/channels/\('((?:[^']|'')*)'\)/allMembers$",
            False,
            True,
        ),
    ]

    for pattern, team_odata, channel_odata in patterns:
        match = re.fullmatch(pattern, normalized, re.IGNORECASE)
        if match is None:
            continue
        team_segment = match.group(1)
        channel_segment = match.group(2)
        if not team_segment or not channel_segment:
            return None
        team_id = _decode_path_segment(team_segment, odata_literal=team_odata)
        channel_id = _decode_path_segment(channel_segment, odata_literal=channel_odata)
        return team_id, channel_id

    return None


def validate_msgraph_teams_channel_members_continuation(
    continuation: object,
    *,
    team_id: str,
    channel_id: str,
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
    path_lower = parsed.path.lower()
    if "/members" in path_lower and "/allmembers" not in path_lower:
        raise IntegrationConfigurationError(_INVALID_MEMBERS_CONTINUATION) from None

    extracted = _extract_channel_all_members_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted is None:
        raise IntegrationConfigurationError(_INVALID_MEMBERS_CONTINUATION) from None

    extracted_team, extracted_channel = extracted
    try:
        validated_team = validate_msgraph_teams_team_id(team_id)
        validated_channel = validate_msgraph_teams_channel_id(channel_id)
        validated_extracted_team = validate_msgraph_teams_team_id(extracted_team)
        validated_extracted_channel = validate_msgraph_teams_channel_id(extracted_channel)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MEMBERS_CONTINUATION) from None

    if (
        validated_extracted_team != validated_team
        or validated_extracted_channel != validated_channel
    ):
        raise IntegrationConfigurationError(_INVALID_MEMBERS_CONTINUATION) from None

    return revalidated


@runtime_checkable
class MsGraphTeamsChannelMembersReadClient(Protocol):
    def read_teams_channel_members_page(
        self,
        *,
        channel: MsGraphTeamsChannel,
        continuation: MsGraphKnowledgeContinuation | None,
    ) -> MsGraphTeamsChannelMemberPage:
        ...


class MsGraphTeamsChannelMembersReader:
    """Complete Teams channel effective member roster reader with channel observation checks."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        transport: MsGraphKnowledgeTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    def read_teams_channel_members_page(
        self,
        *,
        channel: MsGraphTeamsChannel,
        continuation: MsGraphKnowledgeContinuation | None,
    ) -> MsGraphTeamsChannelMemberPage:
        validated_channel = validate_msgraph_teams_channel(channel)

        read_and_validate_current_teams_channel_observation(
            channel=validated_channel,
            transport=self._transport,
        )

        validated_continuation: MsGraphKnowledgeContinuation | None = None
        if continuation is not None:
            validated_continuation = validate_msgraph_teams_channel_members_continuation(
                continuation,
                team_id=validated_channel.team_remote_id,
                channel_id=validated_channel.remote_id,
                graph_base_url=self._config.graph_base_url,
            )

        if validated_continuation is None:
            quoted_team = quote(validated_channel.team_remote_id, safe="")
            quoted_channel = quote(validated_channel.remote_id, safe="")
            path = f"/teams/{quoted_team}/channels/{quoted_channel}/allMembers"
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
            parse_msgraph_teams_channel_member(raw_item, channel=validated_channel)
            for raw_item in collection_page.items
        )
        page = validate_msgraph_teams_channel_member_page(
            _safe_construct_member_page(
                team_remote_id=validated_channel.team_remote_id,
                channel_remote_id=validated_channel.remote_id,
                items=parsed_items,
                continuation=collection_page.continuation,
            ),
            channel=validated_channel,
            graph_base_url=self._config.graph_base_url,
        )

        read_and_validate_current_teams_channel_observation(
            channel=validated_channel,
            transport=self._transport,
        )
        return page


__all__ = [
    "MsGraphTeamsChannelChanged",
    "MsGraphTeamsChannelMember",
    "MsGraphTeamsChannelMemberKind",
    "MsGraphTeamsChannelMemberPage",
    "MsGraphTeamsChannelMemberRole",
    "MsGraphTeamsChannelMembersReadClient",
    "MsGraphTeamsChannelMembersReader",
    "parse_msgraph_teams_channel_member",
    "validate_msgraph_teams_channel_member",
    "validate_msgraph_teams_channel_member_page",
    "validate_msgraph_teams_channel_members_continuation",
]
