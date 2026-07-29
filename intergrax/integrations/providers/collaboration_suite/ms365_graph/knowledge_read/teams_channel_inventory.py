# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Teams Channel knowledge-read: caller-visible channel inventory."""

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

MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND = "teams_channel"

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_CHANNELS_RESPONSE = "unexpected Microsoft Graph Teams channels response"
_INVALID_CHANNELS_REQUEST = "invalid Microsoft Graph Teams channels request"
_INVALID_CHANNELS_CONTINUATION = "invalid Microsoft Graph Teams channels continuation"
_MAX_MSGRAPH_ID_LEN = 4096
_MAX_DISPLAY_NAME_LEN = 1024
_MAX_DESCRIPTION_LEN = 32_768
_MAX_TENANT_ID_LEN = 2048
_MAX_ENUM_STRING_LEN = 1024
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_DESCRIPTION_CONTROL = re.compile(r"[\x00\x01-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PREFER_UNKNOWN_ENUM = {"Prefer": "include-unknown-enum-members"}
_CHANNEL_SELECT = (
    "id,"
    "displayName,"
    "description,"
    "createdDateTime,"
    "membershipType,"
    "isArchived,"
    "tenantId"
)


class MsGraphTeamsChannelMembershipType(StrEnum):
    STANDARD = "standard"
    PRIVATE = "private"
    SHARED = "shared"
    UNKNOWN = "unknown"


_MEMBERSHIP_TYPE_MAP: dict[str, MsGraphTeamsChannelMembershipType] = {
    "standard": MsGraphTeamsChannelMembershipType.STANDARD,
    "private": MsGraphTeamsChannelMembershipType.PRIVATE,
    "shared": MsGraphTeamsChannelMembershipType.SHARED,
}


def _validate_msgraph_opaque_id(value: object, *, error: str = _MALFORMED_CHANNELS_RESPONSE) -> str:
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


def validate_msgraph_teams_team_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def validate_msgraph_teams_channel_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def validate_msgraph_teams_channel_member_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def validate_msgraph_teams_channel_message_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def validate_msgraph_teams_channel_hosted_content_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def _validate_enum_string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    if len(trimmed) > _MAX_ENUM_STRING_LEN:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    return trimmed


def _map_membership_type(value: object) -> MsGraphTeamsChannelMembershipType:
    trimmed = _validate_enum_string(value)
    normalized = trimmed.lower()
    return _MEMBERSHIP_TYPE_MAP.get(normalized, MsGraphTeamsChannelMembershipType.UNKNOWN)


def _validate_exact_bool(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    return value


def _parse_timezone_aware_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    if trimmed.endswith("Z"):
        trimmed = f"{trimmed[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(trimmed)
    except ValueError:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    return parsed.astimezone(timezone.utc)


def _normalize_model_datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    return value.astimezone(timezone.utc)


def _validate_display_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    if len(trimmed) > _MAX_DISPLAY_NAME_LEN:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    return trimmed


def _validate_description(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        return None
    if _DESCRIPTION_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    if len(trimmed) > _MAX_DESCRIPTION_LEN:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    return trimmed


def _validate_optional_tenant_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    if len(trimmed) > _MAX_TENANT_ID_LEN:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
    return trimmed


class MsGraphTeamsChannel(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    team_remote_id: str
    remote_id: str

    display_name: str = Field(repr=False)
    description: str | None = Field(default=None, repr=False)

    created_at: datetime | None = None

    membership_type: MsGraphTeamsChannelMembershipType
    is_archived: bool

    tenant_id: str | None = Field(default=None, repr=False)

    @field_validator("team_remote_id", mode="before")
    @classmethod
    def _validate_team_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_team_id(value)

    @field_validator("remote_id", mode="before")
    @classmethod
    def _validate_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_id(value)

    @field_validator("display_name", mode="before")
    @classmethod
    def _validate_display_name_field(cls, value: object) -> str:
        return _validate_display_name(value)

    @field_validator("description", mode="before")
    @classmethod
    def _validate_description_field(cls, value: object) -> str | None:
        return _validate_description(value)

    @field_validator("created_at", mode="before")
    @classmethod
    def _validate_created_at(cls, value: object) -> datetime | None:
        if value is None:
            return None
        return _normalize_model_datetime(value)

    @field_validator("tenant_id", mode="before")
    @classmethod
    def _validate_tenant_id(cls, value: object) -> str | None:
        return _validate_optional_tenant_id(value)

    @field_validator("is_archived", mode="before")
    @classmethod
    def _validate_is_archived(cls, value: object) -> bool:
        return _validate_exact_bool(value)


class MsGraphTeamsChannelPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    team_remote_id: str
    items: tuple[MsGraphTeamsChannel, ...]

    continuation: MsGraphKnowledgeContinuation | None = Field(default=None, repr=False)

    @field_validator("team_remote_id", mode="before")
    @classmethod
    def _validate_team_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_team_id(value)

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, value: object) -> tuple[MsGraphTeamsChannel, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphTeamsChannel):
                raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
        return value

    @field_validator("continuation", mode="before")
    @classmethod
    def _validate_continuation(cls, value: object) -> MsGraphKnowledgeContinuation | None:
        if value is None:
            return None
        if not isinstance(value, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
        if value.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_page_shape(self) -> MsGraphTeamsChannelPage:
        remote_ids = [item.remote_id for item in self.items]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
        for item in self.items:
            if item.team_remote_id != self.team_remote_id:
                raise ValueError(_MALFORMED_CHANNELS_RESPONSE)
        return self

    @property
    def has_more(self) -> bool:
        return self.continuation is not None


def _safe_construct_channel(**kwargs: object) -> MsGraphTeamsChannel:
    try:
        return MsGraphTeamsChannel(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None


def _safe_construct_channel_page(**kwargs: object) -> MsGraphTeamsChannelPage:
    try:
        return MsGraphTeamsChannelPage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None


class MsGraphTeamsChannelChanged(IntegrationDependencyError):
    def __init__(self) -> None:
        super().__init__("Microsoft Graph Teams channel changed during read")


def parse_msgraph_teams_channel(
    payload: object,
    *,
    expected_team_id: str,
) -> MsGraphTeamsChannel:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None

    required_keys = ("id", "displayName", "membershipType", "isArchived")
    for key in required_keys:
        if key not in payload:
            raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None

    try:
        validated_team_id = validate_msgraph_teams_team_id(expected_team_id)
        remote_id = validate_msgraph_teams_channel_id(payload.get("id"))
        display_name = _validate_display_name(payload.get("displayName"))
        membership_type = _map_membership_type(payload.get("membershipType"))
        is_archived = _validate_exact_bool(payload.get("isArchived"))
    except ValueError:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None

    description: str | None = None
    if "description" in payload:
        try:
            description = _validate_description(payload.get("description"))
        except ValueError:
            raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None

    created_at: datetime | None = None
    if "createdDateTime" in payload and payload.get("createdDateTime") is not None:
        try:
            created_at = _parse_timezone_aware_datetime(payload.get("createdDateTime"))
        except ValueError:
            raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None

    tenant_id: str | None = None
    if "tenantId" in payload:
        try:
            tenant_id = _validate_optional_tenant_id(payload.get("tenantId"))
        except ValueError:
            raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None

    return _safe_construct_channel(
        team_remote_id=validated_team_id,
        remote_id=remote_id,
        display_name=display_name,
        description=description,
        created_at=created_at,
        membership_type=membership_type,
        is_archived=is_archived,
        tenant_id=tenant_id,
    )


def validate_msgraph_teams_channel(value: object) -> MsGraphTeamsChannel:
    if not isinstance(value, MsGraphTeamsChannel):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None
    try:
        return MsGraphTeamsChannel.model_validate(value.model_dump(mode="python"))
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None


def validate_msgraph_teams_channel_page(
    value: object,
    *,
    team_id: str,
    graph_base_url: str,
) -> MsGraphTeamsChannelPage:
    if not isinstance(value, MsGraphTeamsChannelPage):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None

    try:
        validated_team_id = validate_msgraph_teams_team_id(team_id)
    except ValueError:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None

    try:
        raw_team_id = value.team_remote_id
        raw_items = value.items
        raw_continuation = value.continuation
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None

    try:
        validated_page_team = validate_msgraph_teams_team_id(raw_team_id)
    except ValueError:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None

    if validated_page_team != validated_team_id:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None

    if type(raw_items) is not tuple:
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None

    validated_items: list[MsGraphTeamsChannel] = []
    for item in raw_items:
        if not isinstance(item, MsGraphTeamsChannel):
            raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None
        validated_item = validate_msgraph_teams_channel(item)
        if validated_item.team_remote_id != validated_team_id:
            raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None
        validated_items.append(validated_item)

    remote_ids = [item.remote_id for item in validated_items]
    if len(remote_ids) != len(set(remote_ids)):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None

    validated_continuation: MsGraphKnowledgeContinuation | None = None
    if raw_continuation is not None:
        if not isinstance(raw_continuation, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None
        try:
            validated_continuation = validate_msgraph_teams_channels_continuation(
                raw_continuation,
                team_id=validated_team_id,
                graph_base_url=graph_base_url,
            )
        except IntegrationConfigurationError:
            raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None

    try:
        return MsGraphTeamsChannelPage(
            team_remote_id=validated_team_id,
            items=tuple(validated_items),
            continuation=validated_continuation,
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CHANNELS_RESPONSE) from None


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


def _extract_channels_path(
    path: str,
    *,
    graph_base_path: str,
) -> str | None:
    normalized = path.rstrip("/") or "/"
    base = graph_base_path.rstrip("/") or "/"

    slash_match = re.fullmatch(
        rf"^{re.escape(base)}/teams/([^/]+)/channels$",
        normalized,
        re.IGNORECASE,
    )
    if slash_match is not None:
        team_segment = slash_match.group(1)
        if not team_segment:
            return None
        return unquote(team_segment)

    odata_match = re.fullmatch(
        rf"^{re.escape(base)}/teams\('((?:[^']|'')*)'\)/channels$",
        normalized,
        re.IGNORECASE,
    )
    if odata_match is not None:
        team_literal = odata_match.group(1)
        if not team_literal:
            return None
        decoded = unquote(team_literal)
        return _decode_odata_literal(decoded)

    return None


def validate_msgraph_teams_channels_continuation(
    continuation: object,
    *,
    team_id: str,
    graph_base_url: str,
) -> MsGraphKnowledgeContinuation:
    if not isinstance(continuation, MsGraphKnowledgeContinuation):
        raise IntegrationConfigurationError(_INVALID_CHANNELS_CONTINUATION) from None

    try:
        revalidated = MsGraphKnowledgeContinuation.model_validate(
            continuation.model_dump(mode="python")
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise IntegrationConfigurationError(_INVALID_CHANNELS_CONTINUATION) from None

    if revalidated.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
        raise IntegrationConfigurationError(_INVALID_CHANNELS_CONTINUATION) from None

    try:
        validated_url = validate_msgraph_continuation_url(
            revalidated.url,
            graph_base_url=graph_base_url,
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_CHANNELS_CONTINUATION) from None

    parsed = urlparse(validated_url)
    path_lower = parsed.path.lower()
    for forbidden in (
        "/allchannels",
        "/incomingchannels",
        "/members",
        "/allmembers",
        "/messages",
        "/replies",
        "/hostedcontents",
        "/mail",
        "/calendar",
        "/drive",
    ):
        if forbidden in path_lower:
            raise IntegrationConfigurationError(_INVALID_CHANNELS_CONTINUATION) from None

    extracted_team = _extract_channels_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted_team is None:
        raise IntegrationConfigurationError(_INVALID_CHANNELS_CONTINUATION) from None

    try:
        validated_team_id = validate_msgraph_teams_team_id(team_id)
        validated_extracted_team = validate_msgraph_teams_team_id(extracted_team)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_CHANNELS_CONTINUATION) from None

    if validated_extracted_team != validated_team_id:
        raise IntegrationConfigurationError(_INVALID_CHANNELS_CONTINUATION) from None

    return revalidated


def _compare_channel_observation(
    payload: dict[str, object],
    *,
    channel: MsGraphTeamsChannel,
) -> None:
    try:
        response_id = validate_msgraph_teams_channel_id(payload.get("id"))
        response_membership = _map_membership_type(payload.get("membershipType"))
        response_archived = _validate_exact_bool(payload.get("isArchived"))
        response_tenant: str | None = None
        if "tenantId" in payload:
            response_tenant = _validate_optional_tenant_id(payload.get("tenantId"))
        response_created: datetime | None = None
        if "createdDateTime" in payload and payload.get("createdDateTime") is not None:
            response_created = _parse_timezone_aware_datetime(payload.get("createdDateTime"))
    except ValueError:
        raise MsGraphTeamsChannelChanged() from None

    if (
        response_id != channel.remote_id
        or response_membership is not channel.membership_type
        or response_archived != channel.is_archived
        or response_tenant != channel.tenant_id
        or response_created != channel.created_at
    ):
        raise MsGraphTeamsChannelChanged() from None


def read_and_validate_current_teams_channel_observation(
    *,
    channel: MsGraphTeamsChannel,
    transport: MsGraphKnowledgeTransport,
) -> MsGraphTeamsChannel:
    validated_channel = validate_msgraph_teams_channel(channel)
    quoted_team = quote(validated_channel.team_remote_id, safe="")
    quoted_channel = quote(validated_channel.remote_id, safe="")
    path = f"/teams/{quoted_team}/channels/{quoted_channel}"
    payload = transport.get_initial_json(
        path=path,
        params={"$select": _CHANNEL_SELECT},
        headers=_PREFER_UNKNOWN_ENUM,
        not_found_is_dependency=True,
    )
    _compare_channel_observation(payload, channel=validated_channel)
    observed = parse_msgraph_teams_channel(
        payload,
        expected_team_id=validated_channel.team_remote_id,
    )
    if observed.team_remote_id != validated_channel.team_remote_id:
        raise MsGraphTeamsChannelChanged() from None
    _compare_channel_observation(payload, channel=validated_channel)
    return observed


@runtime_checkable
class MsGraphTeamsChannelsReadClient(Protocol):
    def read_teams_channels_page(
        self,
        *,
        team_id: str,
        continuation: MsGraphKnowledgeContinuation | None,
    ) -> MsGraphTeamsChannelPage:
        ...


class MsGraphTeamsChannelsReader:
    """Caller-visible Teams channel inventory reader."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        transport: MsGraphKnowledgeTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    def read_teams_channels_page(
        self,
        *,
        team_id: str,
        continuation: MsGraphKnowledgeContinuation | None,
    ) -> MsGraphTeamsChannelPage:
        try:
            validated_team_id = validate_msgraph_teams_team_id(team_id)
        except ValueError:
            raise IntegrationConfigurationError(_INVALID_CHANNELS_REQUEST) from None

        if continuation is None:
            quoted_team = quote(validated_team_id, safe="")
            path = f"/teams/{quoted_team}/channels"
            payload = self._transport.get_initial_json(
                path=path,
                params={"$select": _CHANNEL_SELECT},
                headers=_PREFER_UNKNOWN_ENUM,
            )
        else:
            validated_continuation = validate_msgraph_teams_channels_continuation(
                continuation,
                team_id=validated_team_id,
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
            parse_msgraph_teams_channel(
                raw_item,
                expected_team_id=validated_team_id,
            )
            for raw_item in collection_page.items
        )
        return validate_msgraph_teams_channel_page(
            _safe_construct_channel_page(
                team_remote_id=validated_team_id,
                items=parsed_items,
                continuation=collection_page.continuation,
            ),
            team_id=validated_team_id,
            graph_base_url=self._config.graph_base_url,
        )
