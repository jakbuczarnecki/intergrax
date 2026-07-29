# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Calendar knowledge-read: calendar-view event delta for one known calendar."""

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
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_inventory import (
    MsGraphCalendar,
    validate_msgraph_calendar,
    validate_msgraph_calendar_event_id,
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

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_CALENDAR_EVENTS_RESPONSE = "unexpected Microsoft Graph Calendar events response"
_INVALID_CALENDAR_EVENTS_REQUEST = "invalid Microsoft Graph Calendar events request"
_INVALID_CALENDAR_EVENTS_CONTINUATION = "invalid Microsoft Graph Calendar events continuation"
_INVALID_CALENDAR_EVENTS_SNAPSHOT_CONTINUATION = (
    "invalid Microsoft Graph Calendar events snapshot continuation"
)
_MAX_CHANGE_KEY_LEN = 2048
_MAX_REMOVED_REASON_LEN = 256
_MAX_ICal_UID_LEN = 2048
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_MIN_DELTA_LIMIT = 1
_MAX_DELTA_LIMIT = 1000


class MsGraphCalendarEventChangeKind(StrEnum):
    ACTIVE = "active"
    REMOVED = "removed"


class MsGraphCalendarEventType(StrEnum):
    SINGLE_INSTANCE = "single_instance"
    OCCURRENCE = "occurrence"
    EXCEPTION = "exception"
    SERIES_MASTER = "series_master"
    UNKNOWN = "unknown"


_EVENT_TYPE_MAP: dict[str, MsGraphCalendarEventType] = {
    "singleinstance": MsGraphCalendarEventType.SINGLE_INSTANCE,
    "occurrence": MsGraphCalendarEventType.OCCURRENCE,
    "exception": MsGraphCalendarEventType.EXCEPTION,
    "seriesmaster": MsGraphCalendarEventType.SERIES_MASTER,
}


class MsGraphCalendarViewWindow(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    start_at: datetime
    end_at: datetime

    @field_validator("start_at", "end_at", mode="before")
    @classmethod
    def _validate_window_datetime(cls, value: object) -> datetime:
        if not isinstance(value, datetime):
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _validate_window_bounds(self) -> MsGraphCalendarViewWindow:
        if self.start_at >= self.end_at:
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        return self


def format_msgraph_calendar_window_datetime(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    normalized = value.astimezone(timezone.utc)
    iso = normalized.strftime("%Y-%m-%dT%H:%M:%S")
    fractional = normalized.microsecond
    if fractional:
        iso += f".{fractional:06d}".rstrip("0").rstrip(".")
    return f"{iso}Z"


def _normalize_model_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    return value.astimezone(timezone.utc)


def _parse_date_time_time_zone(value: object) -> datetime:
    if not isinstance(value, dict):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    if "dateTime" not in value or "timeZone" not in value:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    time_zone = value.get("timeZone")
    if not isinstance(time_zone, str) or time_zone.upper() != "UTC":
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    date_time = value.get("dateTime")
    if not isinstance(date_time, str) or not date_time.strip():
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    trimmed = date_time.strip()
    if trimmed.endswith("Z"):
        trimmed = f"{trimmed[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(trimmed)
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    elif parsed.utcoffset() is None:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    return parsed.astimezone(timezone.utc)


def _validate_optional_opaque_string(value: object, *, max_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    if len(trimmed) > max_length:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    return trimmed


def _map_event_type(value: object) -> MsGraphCalendarEventType:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    normalized = trimmed.lower().replace("_", "")
    return _EVENT_TYPE_MAP.get(normalized, MsGraphCalendarEventType.UNKNOWN)


def _parse_required_bool(mapping: dict[str, object], key: str) -> bool:
    if key not in mapping:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    value = mapping[key]
    if type(value) is not bool:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    return value


class MsGraphCalendarEventChange(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    calendar_remote_id: str
    remote_id: str

    kind: MsGraphCalendarEventChangeKind

    change_key: str | None = Field(default=None, repr=False)
    event_type: MsGraphCalendarEventType | None = None

    start_at: datetime | None = None
    end_at: datetime | None = None
    original_start_at: datetime | None = None
    last_modified_at: datetime | None = None

    series_master_id: str | None = None
    i_cal_uid: str | None = Field(default=None, repr=False)

    is_all_day: bool | None = None
    is_cancelled: bool | None = None
    is_draft: bool | None = None
    has_attachments: bool | None = None
    is_online_meeting: bool | None = None

    removed_reason: str | None = None

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("calendar_remote_id", mode="before")
    @classmethod
    def _validate_calendar_remote_id(cls, value: object) -> str:
        return validate_msgraph_calendar_id(value)

    @field_validator("remote_id", mode="before")
    @classmethod
    def _validate_remote_id(cls, value: object) -> str:
        return validate_msgraph_calendar_event_id(value)

    @field_validator("change_key", mode="before")
    @classmethod
    def _validate_change_key(cls, value: object) -> str | None:
        return _validate_optional_opaque_string(value, max_length=_MAX_CHANGE_KEY_LEN)

    @field_validator("series_master_id", mode="before")
    @classmethod
    def _validate_series_master_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return validate_msgraph_calendar_event_id(value)

    @field_validator("i_cal_uid", mode="before")
    @classmethod
    def _validate_i_cal_uid(cls, value: object) -> str | None:
        return _validate_optional_opaque_string(value, max_length=_MAX_ICal_UID_LEN)

    @field_validator("removed_reason", mode="before")
    @classmethod
    def _validate_removed_reason(cls, value: object) -> str | None:
        return _validate_optional_opaque_string(value, max_length=_MAX_REMOVED_REASON_LEN)

    @field_validator(
        "start_at",
        "end_at",
        "original_start_at",
        "last_modified_at",
        mode="before",
    )
    @classmethod
    def _validate_datetimes(cls, value: object) -> datetime | None:
        return _normalize_model_datetime(value)

    @field_validator(
        "is_all_day",
        "is_cancelled",
        "is_draft",
        "has_attachments",
        "is_online_meeting",
        mode="before",
    )
    @classmethod
    def _validate_optional_bools(cls, value: object) -> bool | None:
        if value is None:
            return None
        if type(value) is not bool:
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_kind_rules(self) -> MsGraphCalendarEventChange:
        if self.kind == MsGraphCalendarEventChangeKind.ACTIVE:
            if self.change_key is None:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
            if self.event_type is None:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
            if self.start_at is None or self.end_at is None or self.last_modified_at is None:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
            if self.is_all_day is None or self.is_cancelled is None:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
            if self.is_draft is None or self.has_attachments is None:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
            if self.is_online_meeting is None:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
            if self.removed_reason is not None:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
            if self.end_at <= self.start_at:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        elif self.kind == MsGraphCalendarEventChangeKind.REMOVED:
            if self.removed_reason is None:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        return self

    @property
    def is_removed_from_view(self) -> bool:
        return self.kind is MsGraphCalendarEventChangeKind.REMOVED


class MsGraphCalendarEventDeltaPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    calendar_remote_id: str
    window: MsGraphCalendarViewWindow

    items: tuple[MsGraphCalendarEventChange, ...]
    continuation: MsGraphKnowledgeContinuation = Field(repr=False)

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("calendar_remote_id", mode="before")
    @classmethod
    def _validate_calendar_remote_id(cls, value: object) -> str:
        return validate_msgraph_calendar_id(value)

    @field_validator("window", mode="before")
    @classmethod
    def _validate_window(cls, value: object) -> MsGraphCalendarViewWindow:
        if not isinstance(value, MsGraphCalendarViewWindow):
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        return value

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, value: object) -> tuple[MsGraphCalendarEventChange, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphCalendarEventChange):
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        return value

    @field_validator("continuation", mode="before")
    @classmethod
    def _validate_continuation(cls, value: object) -> MsGraphKnowledgeContinuation:
        if not isinstance(value, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        if value.kind not in {
            MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            MsGraphKnowledgeContinuationKind.DELTA,
        }:
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_page_shape(self) -> MsGraphCalendarEventDeltaPage:
        remote_ids = [item.remote_id for item in self.items]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        for item in self.items:
            if item.mailbox_user_id != self.mailbox_user_id:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
            if item.calendar_remote_id != self.calendar_remote_id:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        return self

    @property
    def has_more(self) -> bool:
        return self.continuation.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE

    @property
    def is_complete(self) -> bool:
        return self.continuation.kind is MsGraphKnowledgeContinuationKind.DELTA


class MsGraphCalendarEventSnapshotPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    calendar_remote_id: str
    window: MsGraphCalendarViewWindow

    items: tuple[MsGraphCalendarEventChange, ...]

    continuation: MsGraphKnowledgeContinuation | None = Field(
        default=None,
        repr=False,
    )

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("calendar_remote_id", mode="before")
    @classmethod
    def _validate_calendar_remote_id(cls, value: object) -> str:
        return validate_msgraph_calendar_id(value)

    @field_validator("window", mode="before")
    @classmethod
    def _validate_window(cls, value: object) -> MsGraphCalendarViewWindow:
        if not isinstance(value, MsGraphCalendarViewWindow):
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        return value

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, value: object) -> tuple[MsGraphCalendarEventChange, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphCalendarEventChange):
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        return value

    @field_validator("continuation", mode="before")
    @classmethod
    def _validate_continuation(cls, value: object) -> MsGraphKnowledgeContinuation | None:
        if value is None:
            return None
        if not isinstance(value, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        if value.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_page_shape(self) -> MsGraphCalendarEventSnapshotPage:
        remote_ids = [item.remote_id for item in self.items]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        for item in self.items:
            if item.mailbox_user_id != self.mailbox_user_id:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
            if item.calendar_remote_id != self.calendar_remote_id:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
            if item.kind is not MsGraphCalendarEventChangeKind.ACTIVE:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        return self

    @property
    def has_more(self) -> bool:
        return self.continuation is not None

    @property
    def is_complete(self) -> bool:
        return self.continuation is None


def _safe_construct_event_snapshot_page(**kwargs: object) -> MsGraphCalendarEventSnapshotPage:
    try:
        return MsGraphCalendarEventSnapshotPage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None


def _safe_construct_event_change(**kwargs: object) -> MsGraphCalendarEventChange:
    try:
        return MsGraphCalendarEventChange(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None


def _safe_construct_event_delta_page(**kwargs: object) -> MsGraphCalendarEventDeltaPage:
    try:
        return MsGraphCalendarEventDeltaPage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None


def _parse_timezone_aware_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    if trimmed.endswith("Z"):
        trimmed = f"{trimmed[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(trimmed)
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    return parsed.astimezone(timezone.utc)


def _parse_original_start(value: object) -> datetime:
    if isinstance(value, dict):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
    return _parse_timezone_aware_datetime(value)


def parse_msgraph_calendar_event_change(
    payload: object,
    *,
    expected_mailbox_user_id: str,
    expected_calendar_id: str,
) -> MsGraphCalendarEventChange:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(expected_mailbox_user_id)
        validated_calendar_id = validate_msgraph_calendar_id(expected_calendar_id)
        remote_id = validate_msgraph_calendar_event_id(payload.get("id"))
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    if "id" not in payload:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    if "@removed" in payload:
        removed = payload["@removed"]
        if not isinstance(removed, dict):
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
        reason = removed.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
        normalized_reason = reason.strip()
        if _ASCII_CONTROL.search(normalized_reason):
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
        if len(normalized_reason) > _MAX_REMOVED_REASON_LEN:
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
        return _safe_construct_event_change(
            mailbox_user_id=validated_mailbox_user_id,
            calendar_remote_id=validated_calendar_id,
            remote_id=remote_id,
            kind=MsGraphCalendarEventChangeKind.REMOVED,
            removed_reason=normalized_reason,
        )

    required_active_keys = (
        "changeKey",
        "type",
        "start",
        "end",
        "lastModifiedDateTime",
        "isAllDay",
        "isCancelled",
        "isDraft",
        "hasAttachments",
        "isOnlineMeeting",
    )
    for key in required_active_keys:
        if key not in payload:
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    try:
        change_key = _validate_optional_opaque_string(
            payload.get("changeKey"),
            max_length=_MAX_CHANGE_KEY_LEN,
        )
        if change_key is None:
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE)
        event_type = _map_event_type(payload.get("type"))
        start_at = _parse_date_time_time_zone(payload.get("start"))
        end_at = _parse_date_time_time_zone(payload.get("end"))
        last_modified_at = _parse_timezone_aware_datetime(payload.get("lastModifiedDateTime"))
        is_all_day = _parse_required_bool(payload, "isAllDay")
        is_cancelled = _parse_required_bool(payload, "isCancelled")
        is_draft = _parse_required_bool(payload, "isDraft")
        has_attachments = _parse_required_bool(payload, "hasAttachments")
        is_online_meeting = _parse_required_bool(payload, "isOnlineMeeting")
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    original_start_at: datetime | None = None
    if "originalStart" in payload:
        raw_original_start = payload.get("originalStart")
        if raw_original_start is not None:
            try:
                original_start_at = _parse_original_start(raw_original_start)
            except ValueError:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    if event_type in {
        MsGraphCalendarEventType.OCCURRENCE,
        MsGraphCalendarEventType.EXCEPTION,
    }:
        if original_start_at is None:
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    series_master_id: str | None = None
    if "seriesMasterId" in payload and payload.get("seriesMasterId") is not None:
        try:
            series_master_id = validate_msgraph_calendar_event_id(payload.get("seriesMasterId"))
        except ValueError:
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    i_cal_uid: str | None = None
    if "iCalUId" in payload:
        raw_i_cal = payload.get("iCalUId")
        if raw_i_cal is not None:
            try:
                i_cal_uid = _validate_optional_opaque_string(raw_i_cal, max_length=_MAX_ICal_UID_LEN)
            except ValueError:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    return _safe_construct_event_change(
        mailbox_user_id=validated_mailbox_user_id,
        calendar_remote_id=validated_calendar_id,
        remote_id=remote_id,
        kind=MsGraphCalendarEventChangeKind.ACTIVE,
        change_key=change_key,
        event_type=event_type,
        start_at=start_at,
        end_at=end_at,
        original_start_at=original_start_at,
        last_modified_at=last_modified_at,
        series_master_id=series_master_id,
        i_cal_uid=i_cal_uid,
        is_all_day=is_all_day,
        is_cancelled=is_cancelled,
        is_draft=is_draft,
        has_attachments=has_attachments,
        is_online_meeting=is_online_meeting,
    )


def validate_msgraph_calendar_event_change(value: object) -> MsGraphCalendarEventChange:
    if not isinstance(value, MsGraphCalendarEventChange):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
    try:
        return MsGraphCalendarEventChange.model_validate(value.model_dump(mode="python"))
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None


def validate_msgraph_calendar_event_delta_page(
    value: object,
    *,
    calendar: MsGraphCalendar,
    window: MsGraphCalendarViewWindow,
    graph_base_url: str,
) -> MsGraphCalendarEventDeltaPage:
    if not isinstance(value, MsGraphCalendarEventDeltaPage):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    validated_calendar = validate_msgraph_calendar(calendar)
    if not validated_calendar.is_default_calendar:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    try:
        validated_window = MsGraphCalendarViewWindow.model_validate(
            window.model_dump(mode="python")
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    try:
        raw_mailbox_user_id = value.mailbox_user_id
        raw_calendar_remote_id = value.calendar_remote_id
        raw_items = value.items
        raw_continuation = value.continuation
        raw_window = value.window
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    if raw_mailbox_user_id != validated_calendar.mailbox_user_id:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
    if raw_calendar_remote_id != validated_calendar.remote_id:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    if type(raw_items) is not tuple:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    if not isinstance(raw_window, MsGraphCalendarViewWindow):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
    if (
        raw_window.start_at != validated_window.start_at
        or raw_window.end_at != validated_window.end_at
    ):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    validated_items: list[MsGraphCalendarEventChange] = []
    for item in raw_items:
        if not isinstance(item, MsGraphCalendarEventChange):
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
        validated_change = validate_msgraph_calendar_event_change(item)
        if (
            validated_change.mailbox_user_id != validated_calendar.mailbox_user_id
            or validated_change.calendar_remote_id != validated_calendar.remote_id
        ):
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
        validated_items.append(validated_change)

    remote_ids = [item.remote_id for item in validated_items]
    if len(remote_ids) != len(set(remote_ids)):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    if not isinstance(raw_continuation, MsGraphKnowledgeContinuation):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    try:
        revalidated_continuation = validate_msgraph_calendar_events_continuation(
            raw_continuation,
            mailbox_user_id=validated_calendar.mailbox_user_id,
            calendar_id=validated_calendar.remote_id,
            graph_base_url=graph_base_url,
        )
    except IntegrationConfigurationError:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    try:
        return MsGraphCalendarEventDeltaPage(
            mailbox_user_id=validated_calendar.mailbox_user_id,
            calendar_remote_id=validated_calendar.remote_id,
            window=validated_window,
            items=tuple(validated_items),
            continuation=revalidated_continuation,
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None


def _graph_base_path(graph_base_url: str) -> str:
    parsed_base = urlparse(graph_base_url)
    return parsed_base.path.rstrip("/") or "/"


def _decode_odata_literal(literal: str) -> str:
    return literal.replace("''", "'")


def _extract_primary_calendar_events_delta_path(
    path: str,
    *,
    graph_base_path: str,
) -> str | None:
    normalized = path.rstrip("/") or "/"
    base = graph_base_path.rstrip("/") or "/"

    slash_match = re.fullmatch(
        rf"^{re.escape(base)}/users/([^/]+)/calendarView/delta$",
        normalized,
        re.IGNORECASE,
    )
    if slash_match is not None:
        mailbox_segment = slash_match.group(1)
        if not mailbox_segment:
            return None
        return unquote(mailbox_segment)

    odata_match = re.fullmatch(
        rf"^{re.escape(base)}/users\('((?:[^']|'')*)'\)/calendarView/delta$",
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


def validate_msgraph_calendar_events_continuation(
    continuation: object,
    *,
    mailbox_user_id: str,
    calendar_id: str,
    graph_base_url: str,
) -> MsGraphKnowledgeContinuation:
    if not isinstance(continuation, MsGraphKnowledgeContinuation):
        raise IntegrationConfigurationError(_INVALID_CALENDAR_EVENTS_CONTINUATION) from None

    try:
        revalidated = MsGraphKnowledgeContinuation.model_validate(
            continuation.model_dump(mode="python")
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise IntegrationConfigurationError(_INVALID_CALENDAR_EVENTS_CONTINUATION) from None

    if revalidated.kind not in {
        MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        MsGraphKnowledgeContinuationKind.DELTA,
    }:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_EVENTS_CONTINUATION) from None

    try:
        validated_url = validate_msgraph_continuation_url(
            revalidated.url,
            graph_base_url=graph_base_url,
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_EVENTS_CONTINUATION) from None

    parsed = urlparse(validated_url)
    extracted_mailbox_user_id = _extract_primary_calendar_events_delta_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted_mailbox_user_id is None:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_EVENTS_CONTINUATION) from None

    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(mailbox_user_id)
        validated_calendar_id = validate_msgraph_calendar_id(calendar_id)
        validated_extracted_mailbox = validate_msgraph_mailbox_user_id(
            extracted_mailbox_user_id
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_EVENTS_CONTINUATION) from None

    if validated_extracted_mailbox != validated_mailbox_user_id:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_EVENTS_CONTINUATION) from None

    if not validated_calendar_id:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_EVENTS_CONTINUATION) from None

    return revalidated


def _extract_calendar_events_snapshot_path(
    path: str,
    *,
    graph_base_path: str,
) -> tuple[str, str] | None:
    normalized = path.rstrip("/") or "/"
    base = graph_base_path.rstrip("/") or "/"

    slash_match = re.fullmatch(
        rf"^{re.escape(base)}/users/([^/]+)/calendars/([^/]+)/calendarView$",
        normalized,
        re.IGNORECASE,
    )
    if slash_match is not None:
        mailbox_segment = slash_match.group(1)
        calendar_segment = slash_match.group(2)
        if not mailbox_segment or not calendar_segment:
            return None
        return unquote(mailbox_segment), unquote(calendar_segment)

    odata_match = re.fullmatch(
        rf"^{re.escape(base)}/users\('((?:[^']|'')*)'\)/calendars\('((?:[^']|'')*)'\)/calendarView$",
        normalized,
        re.IGNORECASE,
    )
    if odata_match is not None:
        mailbox_literal = odata_match.group(1)
        calendar_literal = odata_match.group(2)
        if not mailbox_literal or not calendar_literal:
            return None
        mailbox_id = _decode_odata_literal(unquote(mailbox_literal))
        calendar_id = _decode_odata_literal(unquote(calendar_literal))
        return mailbox_id, calendar_id

    odata_calendar_match = re.fullmatch(
        rf"^{re.escape(base)}/users/([^/]+)/calendars\('((?:[^']|'')*)'\)/calendarView$",
        normalized,
        re.IGNORECASE,
    )
    if odata_calendar_match is not None:
        mailbox_segment = odata_calendar_match.group(1)
        calendar_literal = odata_calendar_match.group(2)
        if not mailbox_segment or not calendar_literal:
            return None
        calendar_id = _decode_odata_literal(unquote(calendar_literal))
        return unquote(mailbox_segment), calendar_id

    odata_mailbox_match = re.fullmatch(
        rf"^{re.escape(base)}/users\('((?:[^']|'')*)'\)/calendars/([^/]+)/calendarView$",
        normalized,
        re.IGNORECASE,
    )
    if odata_mailbox_match is not None:
        mailbox_literal = odata_mailbox_match.group(1)
        calendar_segment = odata_mailbox_match.group(2)
        if not mailbox_literal or not calendar_segment:
            return None
        mailbox_id = _decode_odata_literal(unquote(mailbox_literal))
        return mailbox_id, unquote(calendar_segment)

    return None


def validate_msgraph_calendar_events_snapshot_continuation(
    continuation: object,
    *,
    mailbox_user_id: str,
    calendar_id: str,
    graph_base_url: str,
) -> MsGraphKnowledgeContinuation:
    if not isinstance(continuation, MsGraphKnowledgeContinuation):
        raise IntegrationConfigurationError(
            _INVALID_CALENDAR_EVENTS_SNAPSHOT_CONTINUATION
        ) from None

    try:
        revalidated = MsGraphKnowledgeContinuation.model_validate(
            continuation.model_dump(mode="python")
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise IntegrationConfigurationError(
            _INVALID_CALENDAR_EVENTS_SNAPSHOT_CONTINUATION
        ) from None

    if revalidated.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
        raise IntegrationConfigurationError(
            _INVALID_CALENDAR_EVENTS_SNAPSHOT_CONTINUATION
        ) from None

    try:
        validated_url = validate_msgraph_continuation_url(
            revalidated.url,
            graph_base_url=graph_base_url,
        )
    except ValueError:
        raise IntegrationConfigurationError(
            _INVALID_CALENDAR_EVENTS_SNAPSHOT_CONTINUATION
        ) from None

    parsed = urlparse(validated_url)
    extracted = _extract_calendar_events_snapshot_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted is None:
        raise IntegrationConfigurationError(
            _INVALID_CALENDAR_EVENTS_SNAPSHOT_CONTINUATION
        ) from None

    extracted_mailbox_user_id, extracted_calendar_id = extracted
    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(mailbox_user_id)
        validated_calendar_id = validate_msgraph_calendar_id(calendar_id)
        validated_extracted_mailbox = validate_msgraph_mailbox_user_id(
            extracted_mailbox_user_id
        )
        validated_extracted_calendar = validate_msgraph_calendar_id(extracted_calendar_id)
    except ValueError:
        raise IntegrationConfigurationError(
            _INVALID_CALENDAR_EVENTS_SNAPSHOT_CONTINUATION
        ) from None

    if (
        validated_extracted_mailbox != validated_mailbox_user_id
        or validated_extracted_calendar != validated_calendar_id
    ):
        raise IntegrationConfigurationError(
            _INVALID_CALENDAR_EVENTS_SNAPSHOT_CONTINUATION
        ) from None

    return revalidated


def validate_msgraph_calendar_event_snapshot_page(
    value: object,
    *,
    calendar: MsGraphCalendar,
    window: MsGraphCalendarViewWindow,
    graph_base_url: str,
) -> MsGraphCalendarEventSnapshotPage:
    if not isinstance(value, MsGraphCalendarEventSnapshotPage):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    validated_calendar = validate_msgraph_calendar(calendar)
    try:
        validated_window = MsGraphCalendarViewWindow.model_validate(
            window.model_dump(mode="python")
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    try:
        raw_mailbox_user_id = value.mailbox_user_id
        raw_calendar_remote_id = value.calendar_remote_id
        raw_window = value.window
        raw_items = value.items
        raw_continuation = value.continuation
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    if raw_mailbox_user_id != validated_calendar.mailbox_user_id:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
    if raw_calendar_remote_id != validated_calendar.remote_id:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    if not isinstance(raw_window, MsGraphCalendarViewWindow):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
    if (
        raw_window.start_at != validated_window.start_at
        or raw_window.end_at != validated_window.end_at
    ):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    if type(raw_items) is not tuple:
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    validated_items: list[MsGraphCalendarEventChange] = []
    for item in raw_items:
        if not isinstance(item, MsGraphCalendarEventChange):
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
        validated_change = validate_msgraph_calendar_event_change(item)
        if (
            validated_change.mailbox_user_id != validated_calendar.mailbox_user_id
            or validated_change.calendar_remote_id != validated_calendar.remote_id
        ):
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
        if validated_change.kind is not MsGraphCalendarEventChangeKind.ACTIVE:
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
        validated_items.append(validated_change)

    remote_ids = [item.remote_id for item in validated_items]
    if len(remote_ids) != len(set(remote_ids)):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    validated_continuation: MsGraphKnowledgeContinuation | None = None
    if raw_continuation is not None:
        if not isinstance(raw_continuation, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
        try:
            validated_continuation = validate_msgraph_calendar_events_snapshot_continuation(
                raw_continuation,
                mailbox_user_id=validated_calendar.mailbox_user_id,
                calendar_id=validated_calendar.remote_id,
                graph_base_url=graph_base_url,
            )
        except IntegrationConfigurationError:
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

    try:
        return MsGraphCalendarEventSnapshotPage(
            mailbox_user_id=validated_calendar.mailbox_user_id,
            calendar_remote_id=validated_calendar.remote_id,
            window=validated_window,
            items=tuple(validated_items),
            continuation=validated_continuation,
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None


def _deduplicate_event_changes(
    items: tuple[MsGraphCalendarEventChange, ...],
) -> tuple[MsGraphCalendarEventChange, ...]:
    last_by_id: dict[str, MsGraphCalendarEventChange] = {}
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
        raise IntegrationConfigurationError(_INVALID_CALENDAR_EVENTS_REQUEST)
    if limit < _MIN_DELTA_LIMIT or limit > _MAX_DELTA_LIMIT:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_EVENTS_REQUEST)
    return limit


def _delta_headers(limit: int) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Prefer": (
            'IdType="ImmutableId", '
            'outlook.timezone="UTC", '
            'outlook.body-content-type="text", '
            f"odata.maxpagesize={limit}"
        ),
    }


def _snapshot_headers() -> dict[str, str]:
    return {
        "Prefer": (
            'IdType="ImmutableId", '
            'outlook.timezone="UTC", '
            'outlook.body-content-type="text"'
        ),
    }


@runtime_checkable
class MsGraphCalendarEventsReadClient(Protocol):
    def read_calendar_events_delta_page(
        self,
        *,
        calendar: MsGraphCalendar,
        window: MsGraphCalendarViewWindow,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphCalendarEventDeltaPage:
        ...


@runtime_checkable
class MsGraphCalendarEventSnapshotsReadClient(Protocol):
    def read_calendar_events_snapshot_page(
        self,
        *,
        calendar: MsGraphCalendar,
        window: MsGraphCalendarViewWindow,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphCalendarEventSnapshotPage:
        ...


class MsGraphCalendarEventsReader:
    """Calendar-view event delta reader over the shared Graph knowledge transport."""

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
        calendar: MsGraphCalendar,
        window: MsGraphCalendarViewWindow,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphCalendarEventDeltaPage:
        validated_calendar = validate_msgraph_calendar(calendar)
        if not validated_calendar.is_default_calendar:
            raise IntegrationConfigurationError(
                "Microsoft Graph v1.0 Calendar delta supports only the primary calendar"
            ) from None

        try:
            validated_window = MsGraphCalendarViewWindow.model_validate(
                window.model_dump(mode="python")
            )
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise IntegrationConfigurationError(_INVALID_CALENDAR_EVENTS_REQUEST) from None

        validated_limit = _validate_delta_limit(limit)
        headers = _delta_headers(validated_limit)

        if continuation is None:
            quoted_mailbox = quote(validated_calendar.mailbox_user_id, safe="")
            path = f"/users/{quoted_mailbox}/calendarView/delta"
            payload = self._transport.get_initial_json(
                path=path,
                params={
                    "startDateTime": format_msgraph_calendar_window_datetime(
                        validated_window.start_at
                    ),
                    "endDateTime": format_msgraph_calendar_window_datetime(
                        validated_window.end_at
                    ),
                },
                headers=headers,
                not_found_is_dependency=True,
            )
        else:
            validated_continuation = validate_msgraph_calendar_events_continuation(
                continuation,
                mailbox_user_id=validated_calendar.mailbox_user_id,
                calendar_id=validated_calendar.remote_id,
                graph_base_url=self._config.graph_base_url,
            )
            payload = self._transport.get_continuation_json(
                continuation=validated_continuation,
                headers=headers,
                not_found_is_dependency=True,
            )

        collection_page = parse_msgraph_collection_page(
            payload,
            graph_base_url=self._config.graph_base_url,
            delta_mode=True,
        )
        if collection_page.continuation is None:
            raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None

        parsed_items = tuple(
            parse_msgraph_calendar_event_change(
                raw_item,
                expected_mailbox_user_id=validated_calendar.mailbox_user_id,
                expected_calendar_id=validated_calendar.remote_id,
            )
            for raw_item in collection_page.items
        )
        deduplicated = _deduplicate_event_changes(parsed_items)
        page = _safe_construct_event_delta_page(
            mailbox_user_id=validated_calendar.mailbox_user_id,
            calendar_remote_id=validated_calendar.remote_id,
            window=validated_window,
            items=deduplicated,
            continuation=collection_page.continuation,
        )
        return validate_msgraph_calendar_event_delta_page(
            page,
            calendar=validated_calendar,
            window=validated_window,
            graph_base_url=self._config.graph_base_url,
        )

    def read_snapshot_page(
        self,
        *,
        calendar: MsGraphCalendar,
        window: MsGraphCalendarViewWindow,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphCalendarEventSnapshotPage:
        validated_calendar = validate_msgraph_calendar(calendar)
        try:
            validated_window = MsGraphCalendarViewWindow.model_validate(
                window.model_dump(mode="python")
            )
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise IntegrationConfigurationError(_INVALID_CALENDAR_EVENTS_REQUEST) from None

        validated_limit = _validate_delta_limit(limit)
        headers = _snapshot_headers()

        if continuation is None:
            quoted_mailbox = quote(validated_calendar.mailbox_user_id, safe="")
            quoted_calendar = quote(validated_calendar.remote_id, safe="")
            path = (
                f"/users/{quoted_mailbox}/calendars/{quoted_calendar}/calendarView"
            )
            payload = self._transport.get_initial_json(
                path=path,
                params={
                    "startDateTime": format_msgraph_calendar_window_datetime(
                        validated_window.start_at
                    ),
                    "endDateTime": format_msgraph_calendar_window_datetime(
                        validated_window.end_at
                    ),
                    "$top": validated_limit,
                },
                headers=headers,
                not_found_is_dependency=True,
            )
        else:
            validated_continuation = validate_msgraph_calendar_events_snapshot_continuation(
                continuation,
                mailbox_user_id=validated_calendar.mailbox_user_id,
                calendar_id=validated_calendar.remote_id,
                graph_base_url=self._config.graph_base_url,
            )
            payload = self._transport.get_continuation_json(
                continuation=validated_continuation,
                headers=headers,
                not_found_is_dependency=True,
            )

        collection_page = parse_msgraph_collection_page(
            payload,
            graph_base_url=self._config.graph_base_url,
            delta_mode=False,
        )

        parsed_items: list[MsGraphCalendarEventChange] = []
        for raw_item in collection_page.items:
            change = parse_msgraph_calendar_event_change(
                raw_item,
                expected_mailbox_user_id=validated_calendar.mailbox_user_id,
                expected_calendar_id=validated_calendar.remote_id,
            )
            if change.kind is not MsGraphCalendarEventChangeKind.ACTIVE:
                raise ValueError(_MALFORMED_CALENDAR_EVENTS_RESPONSE) from None
            parsed_items.append(change)

        deduplicated = _deduplicate_event_changes(tuple(parsed_items))
        page = _safe_construct_event_snapshot_page(
            mailbox_user_id=validated_calendar.mailbox_user_id,
            calendar_remote_id=validated_calendar.remote_id,
            window=validated_window,
            items=deduplicated,
            continuation=collection_page.continuation,
        )
        return validate_msgraph_calendar_event_snapshot_page(
            page,
            calendar=validated_calendar,
            window=validated_window,
            graph_base_url=self._config.graph_base_url,
        )
