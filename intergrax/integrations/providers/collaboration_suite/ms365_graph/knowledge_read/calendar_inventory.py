# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Calendar knowledge-read: user calendar inventory for one known mailbox."""

from __future__ import annotations

import re
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

MSGRAPH_CALENDAR_SOURCE_KIND = "calendar"

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_CALENDARS_RESPONSE = "unexpected Microsoft Graph calendars response"
_INVALID_CALENDARS_REQUEST = "invalid Microsoft Graph calendars request"
_INVALID_CALENDARS_CONTINUATION = "invalid Microsoft Graph calendars continuation"
_MAX_MSGRAPH_ID_LEN = 2048
_MAX_NAME_LEN = 1024
_MAX_CHANGE_KEY_LEN = 2048
_MAX_OWNER_ADDRESS_LEN = 2048
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_MIN_CALENDAR_LIMIT = 1
_MAX_CALENDAR_LIMIT = 200

_CALENDARS_SELECT = (
    "id,name,changeKey,isDefaultCalendar,canEdit,canShare,canViewPrivateItems,"
    "isRemovable,owner,allowedOnlineMeetingProviders,defaultOnlineMeetingProvider"
)


class MsGraphCalendarOnlineMeetingProvider(StrEnum):
    UNKNOWN = "unknown"
    TEAMS_FOR_BUSINESS = "teams_for_business"
    SKYPE_FOR_BUSINESS = "skype_for_business"
    SKYPE_FOR_CONSUMER = "skype_for_consumer"


_PROVIDER_MAP: dict[str, MsGraphCalendarOnlineMeetingProvider] = {
    "unknown": MsGraphCalendarOnlineMeetingProvider.UNKNOWN,
    "teamsforbusiness": MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
    "skypeforbusiness": MsGraphCalendarOnlineMeetingProvider.SKYPE_FOR_BUSINESS,
    "skypeforconsumer": MsGraphCalendarOnlineMeetingProvider.SKYPE_FOR_CONSUMER,
}


def validate_msgraph_calendar_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def validate_msgraph_calendar_event_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def validate_msgraph_calendar_attachment_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def _validate_msgraph_opaque_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    if len(trimmed) > _MAX_MSGRAPH_ID_LEN:
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    return trimmed


def _validate_calendar_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    if len(trimmed) > _MAX_NAME_LEN:
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    return trimmed


def _validate_change_key(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    if len(trimmed) > _MAX_CHANGE_KEY_LEN:
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    return trimmed


def _validate_owner_address(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    if len(trimmed) > _MAX_OWNER_ADDRESS_LEN:
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    return trimmed


def _validate_exact_bool(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    return value


def _map_online_meeting_provider(value: object) -> MsGraphCalendarOnlineMeetingProvider:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    normalized = trimmed.lower().replace("_", "")
    return _PROVIDER_MAP.get(normalized, MsGraphCalendarOnlineMeetingProvider.UNKNOWN)


def _parse_owner(payload: object) -> MsGraphCalendarOwner | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    email_address = payload.get("emailAddress")
    if not isinstance(email_address, dict):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    if "address" not in email_address:
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    try:
        address = _validate_owner_address(email_address.get("address"))
    except ValueError:
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None
    display_name: str | None = None
    if "name" in email_address:
        name_value = email_address.get("name")
        if name_value is not None:
            if not isinstance(name_value, str):
                raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
            trimmed = name_value.strip()
            if trimmed and _ASCII_CONTROL.search(trimmed):
                raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
            display_name = trimmed or None
    return _safe_construct_owner(display_name=display_name, address=address)


def _deduplicate_providers(
    providers: list[MsGraphCalendarOnlineMeetingProvider],
) -> tuple[MsGraphCalendarOnlineMeetingProvider, ...]:
    seen: set[MsGraphCalendarOnlineMeetingProvider] = set()
    result: list[MsGraphCalendarOnlineMeetingProvider] = []
    for provider in providers:
        if provider in seen:
            continue
        seen.add(provider)
        result.append(provider)
    return tuple(result)


class MsGraphCalendarOwner(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    display_name: str | None = Field(default=None, repr=False)
    address: str = Field(repr=False)

    @field_validator("display_name", mode="before")
    @classmethod
    def _validate_display_name(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
        trimmed = value.strip()
        if not trimmed:
            return None
        if _ASCII_CONTROL.search(trimmed):
            raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
        return trimmed

    @field_validator("address", mode="before")
    @classmethod
    def _validate_address(cls, value: object) -> str:
        return _validate_owner_address(value)


class MsGraphCalendar(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    remote_id: str

    name: str = Field(repr=False)
    change_key: str = Field(repr=False)

    is_default_calendar: bool
    can_edit: bool
    can_share: bool
    can_view_private_items: bool
    is_removable: bool

    owner: MsGraphCalendarOwner | None = Field(default=None, repr=False)

    allowed_online_meeting_providers: tuple[MsGraphCalendarOnlineMeetingProvider, ...] = ()
    default_online_meeting_provider: MsGraphCalendarOnlineMeetingProvider

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("remote_id", mode="before")
    @classmethod
    def _validate_remote_id(cls, value: object) -> str:
        return validate_msgraph_calendar_id(value)

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name_field(cls, value: object) -> str:
        return _validate_calendar_name(value)

    @field_validator("change_key", mode="before")
    @classmethod
    def _validate_change_key_field(cls, value: object) -> str:
        return _validate_change_key(value)

    @field_validator(
        "is_default_calendar",
        "can_edit",
        "can_share",
        "can_view_private_items",
        "is_removable",
        mode="before",
    )
    @classmethod
    def _validate_bools(cls, value: object) -> bool:
        return _validate_exact_bool(value)

    @field_validator("allowed_online_meeting_providers", mode="before")
    @classmethod
    def _validate_allowed_providers(
        cls,
        value: object,
    ) -> tuple[MsGraphCalendarOnlineMeetingProvider, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphCalendarOnlineMeetingProvider):
                raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
        return value


class MsGraphCalendarPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    items: tuple[MsGraphCalendar, ...]

    continuation: MsGraphKnowledgeContinuation | None = Field(default=None, repr=False)

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, value: object) -> tuple[MsGraphCalendar, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphCalendar):
                raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
        return value

    @field_validator("continuation", mode="before")
    @classmethod
    def _validate_continuation(cls, value: object) -> MsGraphKnowledgeContinuation | None:
        if value is None:
            return None
        if not isinstance(value, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
        if value.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_page_shape(self) -> MsGraphCalendarPage:
        remote_ids = [item.remote_id for item in self.items]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
        for item in self.items:
            if item.mailbox_user_id != self.mailbox_user_id:
                raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
        return self

    @property
    def has_more(self) -> bool:
        return self.continuation is not None


def _safe_construct_owner(**kwargs: object) -> MsGraphCalendarOwner:
    try:
        return MsGraphCalendarOwner(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None


def _safe_construct_calendar(**kwargs: object) -> MsGraphCalendar:
    try:
        return MsGraphCalendar(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None


def _safe_construct_calendar_page(**kwargs: object) -> MsGraphCalendarPage:
    try:
        return MsGraphCalendarPage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None


@runtime_checkable
class MsGraphCalendarsReadClient(Protocol):
    def read_calendars_page(
        self,
        *,
        mailbox_user_id: str,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphCalendarPage:
        ...


def validate_msgraph_calendar(value: object) -> MsGraphCalendar:
    if not isinstance(value, MsGraphCalendar):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None
    try:
        return MsGraphCalendar.model_validate(value.model_dump(mode="python"))
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None


def validate_msgraph_calendar_page(
    value: object,
    *,
    mailbox_user_id: str,
    graph_base_url: str,
) -> MsGraphCalendarPage:
    if not isinstance(value, MsGraphCalendarPage):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None

    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(mailbox_user_id)
    except ValueError:
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None

    try:
        raw_items = value.items
        raw_continuation = value.continuation
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None

    if type(raw_items) is not tuple:
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None

    validated_items: list[MsGraphCalendar] = []
    for item in raw_items:
        if not isinstance(item, MsGraphCalendar):
            raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None
        validated_item = validate_msgraph_calendar(item)
        if validated_item.mailbox_user_id != validated_mailbox_user_id:
            raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None
        validated_items.append(validated_item)

    remote_ids = [item.remote_id for item in validated_items]
    if len(remote_ids) != len(set(remote_ids)):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None

    validated_continuation: MsGraphKnowledgeContinuation | None = None
    if raw_continuation is not None:
        if not isinstance(raw_continuation, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None
        try:
            validated_continuation = validate_msgraph_calendars_continuation(
                raw_continuation,
                mailbox_user_id=validated_mailbox_user_id,
                graph_base_url=graph_base_url,
            )
        except IntegrationConfigurationError:
            raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None

    try:
        return MsGraphCalendarPage(
            mailbox_user_id=validated_mailbox_user_id,
            items=tuple(validated_items),
            continuation=validated_continuation,
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None


def parse_msgraph_calendar(
    payload: object,
    *,
    expected_mailbox_user_id: str,
) -> MsGraphCalendar:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None

    required_keys = (
        "id",
        "name",
        "changeKey",
        "isDefaultCalendar",
        "canEdit",
        "canShare",
        "canViewPrivateItems",
        "isRemovable",
        "allowedOnlineMeetingProviders",
        "defaultOnlineMeetingProvider",
    )
    for key in required_keys:
        if key not in payload:
            raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None

    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(expected_mailbox_user_id)
        remote_id = validate_msgraph_calendar_id(payload.get("id"))
        name = _validate_calendar_name(payload.get("name"))
        change_key = _validate_change_key(payload.get("changeKey"))
        is_default_calendar = _validate_exact_bool(payload.get("isDefaultCalendar"))
        can_edit = _validate_exact_bool(payload.get("canEdit"))
        can_share = _validate_exact_bool(payload.get("canShare"))
        can_view_private_items = _validate_exact_bool(payload.get("canViewPrivateItems"))
        is_removable = _validate_exact_bool(payload.get("isRemovable"))
        default_provider = _map_online_meeting_provider(
            payload.get("defaultOnlineMeetingProvider")
        )
    except ValueError:
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None

    owner: MsGraphCalendarOwner | None
    if "owner" not in payload:
        owner = None
    else:
        try:
            owner = _parse_owner(payload.get("owner"))
        except ValueError:
            raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None

    raw_providers = payload.get("allowedOnlineMeetingProviders")
    if not isinstance(raw_providers, list):
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None
    try:
        mapped_providers = [_map_online_meeting_provider(item) for item in raw_providers]
        for item in raw_providers:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(_MALFORMED_CALENDARS_RESPONSE)
    except ValueError:
        raise ValueError(_MALFORMED_CALENDARS_RESPONSE) from None

    allowed_providers = _deduplicate_providers(mapped_providers)

    return _safe_construct_calendar(
        mailbox_user_id=validated_mailbox_user_id,
        remote_id=remote_id,
        name=name,
        change_key=change_key,
        is_default_calendar=is_default_calendar,
        can_edit=can_edit,
        can_share=can_share,
        can_view_private_items=can_view_private_items,
        is_removable=is_removable,
        owner=owner,
        allowed_online_meeting_providers=allowed_providers,
        default_online_meeting_provider=default_provider,
    )


def _graph_base_path(graph_base_url: str) -> str:
    parsed_base = urlparse(graph_base_url)
    return parsed_base.path.rstrip("/") or "/"


def _decode_odata_literal(literal: str) -> str:
    return literal.replace("''", "'")


def _extract_calendars_path(
    path: str,
    *,
    graph_base_path: str,
) -> str | None:
    normalized = path.rstrip("/") or "/"
    base = graph_base_path.rstrip("/") or "/"

    slash_match = re.fullmatch(
        rf"^{re.escape(base)}/users/([^/]+)/calendars$",
        normalized,
        re.IGNORECASE,
    )
    if slash_match is not None:
        mailbox_segment = slash_match.group(1)
        if not mailbox_segment:
            return None
        return unquote(mailbox_segment)

    odata_match = re.fullmatch(
        rf"^{re.escape(base)}/users\('((?:[^']|'')*)'\)/calendars$",
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


def validate_msgraph_calendars_continuation(
    continuation: object,
    *,
    mailbox_user_id: str,
    graph_base_url: str,
) -> MsGraphKnowledgeContinuation:
    if not isinstance(continuation, MsGraphKnowledgeContinuation):
        raise IntegrationConfigurationError(_INVALID_CALENDARS_CONTINUATION) from None

    try:
        revalidated = MsGraphKnowledgeContinuation.model_validate(
            continuation.model_dump(mode="python")
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise IntegrationConfigurationError(_INVALID_CALENDARS_CONTINUATION) from None

    if revalidated.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
        raise IntegrationConfigurationError(_INVALID_CALENDARS_CONTINUATION) from None

    try:
        validated_url = validate_msgraph_continuation_url(
            revalidated.url,
            graph_base_url=graph_base_url,
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_CALENDARS_CONTINUATION) from None

    parsed = urlparse(validated_url)
    extracted_mailbox = _extract_calendars_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted_mailbox is None:
        raise IntegrationConfigurationError(_INVALID_CALENDARS_CONTINUATION) from None

    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(mailbox_user_id)
        validated_extracted_mailbox = validate_msgraph_mailbox_user_id(extracted_mailbox)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_CALENDARS_CONTINUATION) from None

    if validated_extracted_mailbox != validated_mailbox_user_id:
        raise IntegrationConfigurationError(_INVALID_CALENDARS_CONTINUATION) from None

    return revalidated


def _validate_calendar_limit(limit: object) -> int:
    if type(limit) is not int:
        raise IntegrationConfigurationError(_INVALID_CALENDARS_REQUEST)
    if limit < _MIN_CALENDAR_LIMIT or limit > _MAX_CALENDAR_LIMIT:
        raise IntegrationConfigurationError(_INVALID_CALENDARS_REQUEST)
    return limit


class MsGraphCalendarsReader:
    """User calendar inventory reader over the shared Graph knowledge transport."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        transport: MsGraphKnowledgeTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    def read_calendars_page(
        self,
        *,
        mailbox_user_id: str,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphCalendarPage:
        try:
            validated_mailbox_user_id = validate_msgraph_mailbox_user_id(mailbox_user_id)
        except ValueError:
            raise IntegrationConfigurationError(_INVALID_CALENDARS_REQUEST) from None

        validated_limit = _validate_calendar_limit(limit)

        if continuation is None:
            quoted_mailbox = quote(validated_mailbox_user_id, safe="")
            path = f"/users/{quoted_mailbox}/calendars"
            payload = self._transport.get_initial_json(
                path=path,
                params={
                    "$top": validated_limit,
                    "$select": _CALENDARS_SELECT,
                },
            )
        else:
            validated_continuation = validate_msgraph_calendars_continuation(
                continuation,
                mailbox_user_id=validated_mailbox_user_id,
                graph_base_url=self._config.graph_base_url,
            )
            payload = self._transport.get_continuation_json(
                continuation=validated_continuation,
            )

        collection_page = parse_msgraph_collection_page(
            payload,
            graph_base_url=self._config.graph_base_url,
            delta_mode=False,
        )
        parsed_items = tuple(
            parse_msgraph_calendar(
                raw_item,
                expected_mailbox_user_id=validated_mailbox_user_id,
            )
            for raw_item in collection_page.items
        )
        return validate_msgraph_calendar_page(
            _safe_construct_calendar_page(
                mailbox_user_id=validated_mailbox_user_id,
                items=parsed_items,
                continuation=collection_page.continuation,
            ),
            mailbox_user_id=validated_mailbox_user_id,
            graph_base_url=self._config.graph_base_url,
        )
