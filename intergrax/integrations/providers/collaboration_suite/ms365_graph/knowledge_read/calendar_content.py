# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Calendar knowledge-read: event text content and participants."""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Protocol, runtime_checkable
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_events import (
    MsGraphCalendarEventChange,
    MsGraphCalendarEventChangeKind,
    MsGraphCalendarEventType,
    _parse_date_time_time_zone,
    _parse_timezone_aware_datetime,
    parse_msgraph_calendar_event_change,
    validate_msgraph_calendar_event_change,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_inventory import (
    MsGraphCalendarOnlineMeetingProvider,
    validate_msgraph_calendar_event_id,
    validate_msgraph_calendar_id,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.common import (
    MsGraphKnowledgeTransport,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_folders import (
    validate_msgraph_mailbox_user_id,
)

DEFAULT_CALENDAR_EVENT_CONTENT_MAX_CHARS = 2_000_000
ABSOLUTE_CALENDAR_EVENT_CONTENT_MAX_CHARS = 8_000_000

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE = (
    "unexpected Microsoft Graph Calendar event content response"
)
_INVALID_CALENDAR_EVENT_CONTENT_REQUEST = "invalid Microsoft Graph Calendar content request"
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_MAX_PARTICIPANT_ADDRESS_LEN = 2048
_MAX_PARTICIPANT_DISPLAY_NAME_LEN = 1024
_MAX_CHANGE_KEY_LEN = 2048
_MAX_SUBJECT_LEN = 4096
_MAX_ICal_UID_LEN = 2048
_MAX_LOCATION_DISPLAY_NAME_LEN = 1024
_MAX_LOCATION_FIELD_LEN = 1024
_MAX_TIME_ZONE_LEN = 256
_MAX_BODY_PREVIEW_LEN = 4096
_MAX_CATEGORY_LEN = 256

_OBSERVATION_SELECT = (
    "id,changeKey,type,start,end,originalStart,seriesMasterId,iCalUId,"
    "lastModifiedDateTime,isAllDay,isCancelled,isDraft,hasAttachments,isOnlineMeeting"
)

_CONTENT_SELECT = (
    "id,changeKey,type,subject,body,bodyPreview,start,end,originalStart,"
    "originalStartTimeZone,originalEndTimeZone,createdDateTime,lastModifiedDateTime,"
    "organizer,attendees,location,locations,recurrence,seriesMasterId,"
    "cancelledOccurrences,categories,iCalUId,importance,sensitivity,showAs,"
    "responseStatus,isAllDay,isCancelled,isDraft,isOrganizer,isOnlineMeeting,"
    "onlineMeetingProvider,hasAttachments,hideAttendees,allowNewTimeProposals,"
    "responseRequested,isReminderOn,reminderMinutesBeforeStart"
)

_IMMUTABLE_TEXT_BODY_HEADERS = {
    "Prefer": (
        'IdType="ImmutableId", '
        'outlook.timezone="UTC", '
        'outlook.body-content-type="text"'
    ),
}

_OBSERVATION_HEADERS = {
    "Prefer": 'IdType="ImmutableId", outlook.timezone="UTC"',
}


class MsGraphCalendarEventChanged(IntegrationDependencyError):
    """Calendar event identity or revision changed during read."""

    def __init__(self) -> None:
        super().__init__("Microsoft Graph Calendar event changed during read")


class MsGraphCalendarEventContentTooLarge(IntegrationConfigurationError):
    """Calendar event text exceeds the configured character limit."""

    def __init__(self) -> None:
        super().__init__(
            "Microsoft Graph Calendar event exceeds the configured content limit"
        )


class MsGraphCalendarBodyKind(StrEnum):
    TEXT = "text"
    HTML = "html"


class MsGraphCalendarImportance(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    UNKNOWN = "unknown"


class MsGraphCalendarSensitivity(StrEnum):
    NORMAL = "normal"
    PERSONAL = "personal"
    PRIVATE = "private"
    CONFIDENTIAL = "confidential"
    UNKNOWN = "unknown"


class MsGraphCalendarShowAs(StrEnum):
    FREE = "free"
    TENTATIVE = "tentative"
    BUSY = "busy"
    OOF = "oof"
    WORKING_ELSEWHERE = "workingElsewhere"
    UNKNOWN = "unknown"


class MsGraphCalendarAttendeeType(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


class MsGraphCalendarResponseType(StrEnum):
    NONE = "none"
    ORGANIZER = "organizer"
    TENTATIVELY_ACCEPTED = "tentatively_accepted"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    NOT_RESPONDED = "not_responded"
    UNKNOWN = "unknown"


class MsGraphCalendarLocationType(StrEnum):
    DEFAULT = "default"
    CONFERENCE_ROOM = "conference_room"
    HOME_ADDRESS = "home_address"
    BUSINESS_ADDRESS = "business_address"
    GEO_COORDINATES = "geo_coordinates"
    STREET_ADDRESS = "street_address"
    HOTEL = "hotel"
    RESTAURANT = "restaurant"
    LOCAL_BUSINESS = "local_business"
    POSTAL_ADDRESS = "postal_address"
    UNKNOWN = "unknown"


class MsGraphCalendarRecurrencePatternType(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    ABSOLUTE_MONTHLY = "absoluteMonthly"
    RELATIVE_MONTHLY = "relativeMonthly"
    ABSOLUTE_YEARLY = "absoluteYearly"
    RELATIVE_YEARLY = "relativeYearly"
    UNKNOWN = "unknown"


class MsGraphCalendarRecurrenceRangeType(StrEnum):
    END_DATE = "endDate"
    NO_END = "noEnd"
    NUMBERED = "numbered"
    UNKNOWN = "unknown"


class MsGraphCalendarDayOfWeek(StrEnum):
    SUNDAY = "sunday"
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    UNKNOWN = "unknown"


class MsGraphCalendarWeekIndex(StrEnum):
    FIRST = "first"
    SECOND = "second"
    THIRD = "third"
    FOURTH = "fourth"
    LAST = "last"
    UNKNOWN = "unknown"


_IMPORTANCE_MAP: dict[str, MsGraphCalendarImportance] = {
    "low": MsGraphCalendarImportance.LOW,
    "normal": MsGraphCalendarImportance.NORMAL,
    "high": MsGraphCalendarImportance.HIGH,
}

_SENSITIVITY_MAP: dict[str, MsGraphCalendarSensitivity] = {
    "normal": MsGraphCalendarSensitivity.NORMAL,
    "personal": MsGraphCalendarSensitivity.PERSONAL,
    "private": MsGraphCalendarSensitivity.PRIVATE,
    "confidential": MsGraphCalendarSensitivity.CONFIDENTIAL,
}

_SHOW_AS_MAP: dict[str, MsGraphCalendarShowAs] = {
    "free": MsGraphCalendarShowAs.FREE,
    "tentative": MsGraphCalendarShowAs.TENTATIVE,
    "busy": MsGraphCalendarShowAs.BUSY,
    "oof": MsGraphCalendarShowAs.OOF,
    "workingelsewhere": MsGraphCalendarShowAs.WORKING_ELSEWHERE,
}

_ATTENDEE_TYPE_MAP: dict[str, MsGraphCalendarAttendeeType] = {
    "required": MsGraphCalendarAttendeeType.REQUIRED,
    "optional": MsGraphCalendarAttendeeType.OPTIONAL,
    "resource": MsGraphCalendarAttendeeType.RESOURCE,
}

_RESPONSE_TYPE_MAP: dict[str, MsGraphCalendarResponseType] = {
    "none": MsGraphCalendarResponseType.NONE,
    "organizer": MsGraphCalendarResponseType.ORGANIZER,
    "tentativelyaccepted": MsGraphCalendarResponseType.TENTATIVELY_ACCEPTED,
    "accepted": MsGraphCalendarResponseType.ACCEPTED,
    "declined": MsGraphCalendarResponseType.DECLINED,
    "notresponded": MsGraphCalendarResponseType.NOT_RESPONDED,
}

_LOCATION_TYPE_MAP: dict[str, MsGraphCalendarLocationType] = {
    "default": MsGraphCalendarLocationType.DEFAULT,
    "conferenceroom": MsGraphCalendarLocationType.CONFERENCE_ROOM,
    "homeaddress": MsGraphCalendarLocationType.HOME_ADDRESS,
    "businessaddress": MsGraphCalendarLocationType.BUSINESS_ADDRESS,
    "geocoordinates": MsGraphCalendarLocationType.GEO_COORDINATES,
    "streetaddress": MsGraphCalendarLocationType.STREET_ADDRESS,
    "hotel": MsGraphCalendarLocationType.HOTEL,
    "restaurant": MsGraphCalendarLocationType.RESTAURANT,
    "localbusiness": MsGraphCalendarLocationType.LOCAL_BUSINESS,
    "postaladdress": MsGraphCalendarLocationType.POSTAL_ADDRESS,
}

_RECURRENCE_PATTERN_TYPE_MAP: dict[str, MsGraphCalendarRecurrencePatternType] = {
    "daily": MsGraphCalendarRecurrencePatternType.DAILY,
    "weekly": MsGraphCalendarRecurrencePatternType.WEEKLY,
    "absolutemonthly": MsGraphCalendarRecurrencePatternType.ABSOLUTE_MONTHLY,
    "relativemonthly": MsGraphCalendarRecurrencePatternType.RELATIVE_MONTHLY,
    "absoluteyearly": MsGraphCalendarRecurrencePatternType.ABSOLUTE_YEARLY,
    "relativeyearly": MsGraphCalendarRecurrencePatternType.RELATIVE_YEARLY,
}

_RECURRENCE_RANGE_TYPE_MAP: dict[str, MsGraphCalendarRecurrenceRangeType] = {
    "enddate": MsGraphCalendarRecurrenceRangeType.END_DATE,
    "noend": MsGraphCalendarRecurrenceRangeType.NO_END,
    "numbered": MsGraphCalendarRecurrenceRangeType.NUMBERED,
}

_DAY_OF_WEEK_MAP: dict[str, MsGraphCalendarDayOfWeek] = {
    "sunday": MsGraphCalendarDayOfWeek.SUNDAY,
    "monday": MsGraphCalendarDayOfWeek.MONDAY,
    "tuesday": MsGraphCalendarDayOfWeek.TUESDAY,
    "wednesday": MsGraphCalendarDayOfWeek.WEDNESDAY,
    "thursday": MsGraphCalendarDayOfWeek.THURSDAY,
    "friday": MsGraphCalendarDayOfWeek.FRIDAY,
    "saturday": MsGraphCalendarDayOfWeek.SATURDAY,
}

_WEEK_INDEX_MAP: dict[str, MsGraphCalendarWeekIndex] = {
    "first": MsGraphCalendarWeekIndex.FIRST,
    "second": MsGraphCalendarWeekIndex.SECOND,
    "third": MsGraphCalendarWeekIndex.THIRD,
    "fourth": MsGraphCalendarWeekIndex.FOURTH,
    "last": MsGraphCalendarWeekIndex.LAST,
}

_ONLINE_MEETING_PROVIDER_MAP: dict[str, MsGraphCalendarOnlineMeetingProvider] = {
    "unknown": MsGraphCalendarOnlineMeetingProvider.UNKNOWN,
    "teamsforbusiness": MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
    "skypeforbusiness": MsGraphCalendarOnlineMeetingProvider.SKYPE_FOR_BUSINESS,
    "skypeforconsumer": MsGraphCalendarOnlineMeetingProvider.SKYPE_FOR_CONSUMER,
}


def _validate_participant_address(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if len(trimmed) > _MAX_PARTICIPANT_ADDRESS_LEN:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    return trimmed


def _validate_participant_display_name(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        return None
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if len(trimmed) > _MAX_PARTICIPANT_DISPLAY_NAME_LEN:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    return trimmed


def _validate_optional_opaque_string(value: object, *, max_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if len(trimmed) > max_length:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    return trimmed


def _validate_content_revision(value: object) -> str:
    result = _validate_optional_opaque_string(value, max_length=_MAX_CHANGE_KEY_LEN)
    if result is None:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    return result


def _validate_subject(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if "\x00" in value:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if len(value) > _MAX_SUBJECT_LEN:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    return value


def _validate_body_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if "\x00" in value:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    return value


def _validate_body_preview(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if "\x00" in value:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if len(value) > _MAX_BODY_PREVIEW_LEN:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    return value


def _validate_location_display_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if "\x00" in value:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if len(value) > _MAX_LOCATION_DISPLAY_NAME_LEN:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    return value


def _validate_location_field(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if "\x00" in value:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if _ASCII_CONTROL.search(value):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if len(value) > _MAX_LOCATION_FIELD_LEN:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        return None
    return trimmed


def _validate_category(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    if len(trimmed) > _MAX_CATEGORY_LEN:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    return trimmed


def _normalize_enum_key(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    return trimmed.lower().replace("_", "")


def _map_importance(value: object) -> MsGraphCalendarImportance:
    return _IMPORTANCE_MAP.get(_normalize_enum_key(value), MsGraphCalendarImportance.UNKNOWN)


def _map_sensitivity(value: object) -> MsGraphCalendarSensitivity:
    return _SENSITIVITY_MAP.get(_normalize_enum_key(value), MsGraphCalendarSensitivity.UNKNOWN)


def _map_show_as(value: object) -> MsGraphCalendarShowAs:
    return _SHOW_AS_MAP.get(_normalize_enum_key(value), MsGraphCalendarShowAs.UNKNOWN)


def _map_attendee_type(value: object) -> MsGraphCalendarAttendeeType:
    return _ATTENDEE_TYPE_MAP.get(_normalize_enum_key(value), MsGraphCalendarAttendeeType.UNKNOWN)


def _map_response_type(value: object) -> MsGraphCalendarResponseType:
    return _RESPONSE_TYPE_MAP.get(_normalize_enum_key(value), MsGraphCalendarResponseType.UNKNOWN)


def _map_location_type(value: object) -> MsGraphCalendarLocationType:
    return _LOCATION_TYPE_MAP.get(_normalize_enum_key(value), MsGraphCalendarLocationType.UNKNOWN)


def _map_recurrence_pattern_type(value: object) -> MsGraphCalendarRecurrencePatternType:
    return _RECURRENCE_PATTERN_TYPE_MAP.get(
        _normalize_enum_key(value),
        MsGraphCalendarRecurrencePatternType.UNKNOWN,
    )


def _map_recurrence_range_type(value: object) -> MsGraphCalendarRecurrenceRangeType:
    return _RECURRENCE_RANGE_TYPE_MAP.get(
        _normalize_enum_key(value),
        MsGraphCalendarRecurrenceRangeType.UNKNOWN,
    )


def _map_day_of_week(value: object) -> MsGraphCalendarDayOfWeek:
    return _DAY_OF_WEEK_MAP.get(_normalize_enum_key(value), MsGraphCalendarDayOfWeek.UNKNOWN)


def _map_week_index(value: object) -> MsGraphCalendarWeekIndex:
    return _WEEK_INDEX_MAP.get(_normalize_enum_key(value), MsGraphCalendarWeekIndex.UNKNOWN)


def _map_online_meeting_provider(value: object) -> MsGraphCalendarOnlineMeetingProvider:
    return _ONLINE_MEETING_PROVIDER_MAP.get(
        _normalize_enum_key(value),
        MsGraphCalendarOnlineMeetingProvider.UNKNOWN,
    )


def _parse_content_date_time_time_zone(value: object) -> datetime:
    try:
        return _parse_date_time_time_zone(value)
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None


def _parse_content_timezone_aware_datetime(value: object) -> datetime:
    try:
        return _parse_timezone_aware_datetime(value)
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None


def _parse_date_value(value: object) -> date:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    try:
        return date.fromisoformat(trimmed)
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None


def _parse_required_bool(mapping: dict[str, object], key: str) -> bool:
    if key not in mapping:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    value = mapping[key]
    if type(value) is not bool:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    return value


def _parse_required_int(mapping: dict[str, object], key: str) -> int:
    if key not in mapping:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    value = mapping[key]
    if type(value) is not int:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    return value


class MsGraphCalendarParticipant(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    display_name: str | None = Field(default=None, repr=False)
    address: str = Field(repr=False)

    @field_validator("display_name", mode="before")
    @classmethod
    def _validate_display_name_field(cls, value: object) -> str | None:
        return _validate_participant_display_name(value)

    @field_validator("address", mode="before")
    @classmethod
    def _validate_address_field(cls, value: object) -> str:
        return _validate_participant_address(value)


class MsGraphCalendarResponseStatus(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    response: MsGraphCalendarResponseType
    responded_at: datetime | None = None

    @field_validator("responded_at", mode="before")
    @classmethod
    def _validate_responded_at(cls, value: object) -> datetime | None:
        if value is None:
            return None
        return _parse_content_timezone_aware_datetime(value)


class MsGraphCalendarAttendee(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    participant: MsGraphCalendarParticipant = Field(repr=False)
    attendee_type: MsGraphCalendarAttendeeType
    status: MsGraphCalendarResponseStatus


class MsGraphCalendarLocation(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    display_name: str = Field(repr=False)
    location_type: MsGraphCalendarLocationType

    street: str | None = Field(default=None, repr=False)
    city: str | None = Field(default=None, repr=False)
    state: str | None = Field(default=None, repr=False)
    country_or_region: str | None = Field(default=None, repr=False)
    postal_code: str | None = Field(default=None, repr=False)

    @field_validator("display_name", mode="before")
    @classmethod
    def _validate_display_name_field(cls, value: object) -> str:
        return _validate_location_display_name(value)

    @field_validator(
        "street",
        "city",
        "state",
        "country_or_region",
        "postal_code",
        mode="before",
    )
    @classmethod
    def _validate_location_fields(cls, value: object) -> str | None:
        return _validate_location_field(value)


class MsGraphCalendarRecurrencePattern(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    pattern_type: MsGraphCalendarRecurrencePatternType
    interval: int

    month: int | None = None
    day_of_month: int | None = None
    days_of_week: tuple[MsGraphCalendarDayOfWeek, ...] = ()
    first_day_of_week: MsGraphCalendarDayOfWeek | None = None
    index: MsGraphCalendarWeekIndex | None = None

    @field_validator("interval", mode="before")
    @classmethod
    def _validate_interval(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        if value < 1:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        return value

    @field_validator("month", mode="before")
    @classmethod
    def _validate_month(cls, value: object) -> int | None:
        if value is None:
            return None
        if type(value) is not int:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        if value < 1 or value > 12:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        return value

    @field_validator("day_of_month", mode="before")
    @classmethod
    def _validate_day_of_month(cls, value: object) -> int | None:
        if value is None:
            return None
        if type(value) is not int:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        if value < 1 or value > 31:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        return value

    @field_validator("days_of_week", mode="before")
    @classmethod
    def _validate_days_of_week(
        cls,
        value: object,
    ) -> tuple[MsGraphCalendarDayOfWeek, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphCalendarDayOfWeek):
                raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        return value


class MsGraphCalendarRecurrenceRange(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    range_type: MsGraphCalendarRecurrenceRangeType
    start_date: date
    end_date: date | None = None
    number_of_occurrences: int | None = None
    recurrence_time_zone: str | None = None

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _validate_dates(cls, value: object) -> date | None:
        if value is None:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)

    @field_validator("number_of_occurrences", mode="before")
    @classmethod
    def _validate_number_of_occurrences(cls, value: object) -> int | None:
        if value is None:
            return None
        if type(value) is not int:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        if value < 1:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        return value

    @field_validator("recurrence_time_zone", mode="before")
    @classmethod
    def _validate_recurrence_time_zone(cls, value: object) -> str | None:
        return _validate_optional_opaque_string(value, max_length=_MAX_TIME_ZONE_LEN)

    @model_validator(mode="after")
    def _validate_range_bounds(self) -> MsGraphCalendarRecurrenceRange:
        if self.end_date is not None and self.end_date < self.start_date:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        return self


class MsGraphCalendarRecurrence(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    pattern: MsGraphCalendarRecurrencePattern
    range: MsGraphCalendarRecurrenceRange


class MsGraphCalendarEventContent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    calendar_remote_id: str
    remote_id: str
    content_revision: str = Field(repr=False)

    event_type: MsGraphCalendarEventType

    subject: str | None = Field(default=None, repr=False)
    body_kind: MsGraphCalendarBodyKind
    body_content: str = Field(repr=False)
    body_preview: str | None = Field(default=None, repr=False)

    start_at: datetime
    end_at: datetime
    original_start_at: datetime | None = None

    original_start_time_zone: str | None = None
    original_end_time_zone: str | None = None

    created_at: datetime
    last_modified_at: datetime

    organizer: MsGraphCalendarParticipant | None = Field(default=None, repr=False)
    attendees: tuple[MsGraphCalendarAttendee, ...] = Field(default=(), repr=False)

    location: MsGraphCalendarLocation | None = Field(default=None, repr=False)
    locations: tuple[MsGraphCalendarLocation, ...] = Field(default=(), repr=False)

    recurrence: MsGraphCalendarRecurrence | None = None
    series_master_id: str | None = None
    cancelled_occurrence_ids: tuple[str, ...] = ()

    categories: tuple[str, ...] = ()

    i_cal_uid: str | None = Field(default=None, repr=False)

    importance: MsGraphCalendarImportance
    sensitivity: MsGraphCalendarSensitivity
    show_as: MsGraphCalendarShowAs

    response_status: MsGraphCalendarResponseStatus

    is_all_day: bool
    is_cancelled: bool
    is_draft: bool
    is_organizer: bool
    is_online_meeting: bool
    has_attachments: bool
    hide_attendees: bool
    allow_new_time_proposals: bool
    response_requested: bool
    is_reminder_on: bool

    reminder_minutes_before_start: int

    online_meeting_provider: MsGraphCalendarOnlineMeetingProvider

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("calendar_remote_id", mode="before")
    @classmethod
    def _validate_calendar_remote_id(cls, value: object) -> str:
        return validate_msgraph_calendar_id(value)

    @field_validator("remote_id", "series_master_id", mode="before")
    @classmethod
    def _validate_remote_ids(cls, value: object) -> str | None:
        if value is None:
            return None
        return validate_msgraph_calendar_event_id(value)

    @field_validator("content_revision", mode="before")
    @classmethod
    def _validate_content_revision_field(cls, value: object) -> str:
        return _validate_content_revision(value)

    @field_validator("subject", mode="before")
    @classmethod
    def _validate_subject_field(cls, value: object) -> str | None:
        return _validate_subject(value)

    @field_validator("body_content", mode="before")
    @classmethod
    def _validate_body_content_field(cls, value: object) -> str:
        return _validate_body_text(value)

    @field_validator("body_preview", mode="before")
    @classmethod
    def _validate_body_preview_field(cls, value: object) -> str | None:
        return _validate_body_preview(value)

    @field_validator(
        "original_start_time_zone",
        "original_end_time_zone",
        mode="before",
    )
    @classmethod
    def _validate_time_zone_fields(cls, value: object) -> str | None:
        return _validate_optional_opaque_string(value, max_length=_MAX_TIME_ZONE_LEN)

    @field_validator("i_cal_uid", mode="before")
    @classmethod
    def _validate_i_cal_uid(cls, value: object) -> str | None:
        return _validate_optional_opaque_string(value, max_length=_MAX_ICal_UID_LEN)

    @field_validator(
        "start_at",
        "end_at",
        "original_start_at",
        "created_at",
        "last_modified_at",
        mode="before",
    )
    @classmethod
    def _validate_datetimes(cls, value: object) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        return value.astimezone(timezone.utc)

    @field_validator(
        "is_all_day",
        "is_cancelled",
        "is_draft",
        "is_organizer",
        "is_online_meeting",
        "has_attachments",
        "hide_attendees",
        "allow_new_time_proposals",
        "response_requested",
        "is_reminder_on",
        mode="before",
    )
    @classmethod
    def _validate_bools(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        return value

    @field_validator("reminder_minutes_before_start", mode="before")
    @classmethod
    def _validate_reminder_minutes(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        return value

    @field_validator("attendees", mode="before")
    @classmethod
    def _validate_attendees(cls, value: object) -> tuple[MsGraphCalendarAttendee, ...]:
        return _validate_attendee_tuple(value)

    @field_validator("locations", mode="before")
    @classmethod
    def _validate_locations(cls, value: object) -> tuple[MsGraphCalendarLocation, ...]:
        return _validate_location_tuple(value)

    @field_validator("cancelled_occurrence_ids", mode="before")
    @classmethod
    def _validate_cancelled_occurrence_ids(cls, value: object) -> tuple[str, ...]:
        return _validate_event_id_tuple(value)

    @field_validator("categories", mode="before")
    @classmethod
    def _validate_categories(cls, value: object) -> tuple[str, ...]:
        return _validate_category_tuple(value)

    @model_validator(mode="after")
    def _validate_time_bounds(self) -> MsGraphCalendarEventContent:
        if self.end_at <= self.start_at:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        return self


def _validate_attendee_tuple(value: object) -> tuple[MsGraphCalendarAttendee, ...]:
    if type(value) is not tuple:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    validated: list[MsGraphCalendarAttendee] = []
    for item in value:
        validated.append(validate_msgraph_calendar_attendee(item))
    return tuple(validated)


def _validate_location_tuple(value: object) -> tuple[MsGraphCalendarLocation, ...]:
    if type(value) is not tuple:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    validated: list[MsGraphCalendarLocation] = []
    for item in value:
        validated.append(validate_msgraph_calendar_location(item))
    return tuple(validated)


def _validate_event_id_tuple(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    validated: list[str] = []
    for item in value:
        validated.append(validate_msgraph_calendar_event_id(item))
    return tuple(validated)


def _validate_category_tuple(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    validated: list[str] = []
    for item in value:
        validated.append(_validate_category(item))
    return tuple(validated)


def _safe_construct_participant(**kwargs: object) -> MsGraphCalendarParticipant:
    try:
        return MsGraphCalendarParticipant(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None


def _safe_construct_response_status(**kwargs: object) -> MsGraphCalendarResponseStatus:
    try:
        return MsGraphCalendarResponseStatus(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None


def _safe_construct_attendee(**kwargs: object) -> MsGraphCalendarAttendee:
    try:
        return MsGraphCalendarAttendee(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None


def _safe_construct_location(**kwargs: object) -> MsGraphCalendarLocation:
    try:
        return MsGraphCalendarLocation(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None


def _safe_construct_recurrence_pattern(**kwargs: object) -> MsGraphCalendarRecurrencePattern:
    try:
        return MsGraphCalendarRecurrencePattern(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None


def _safe_construct_recurrence_range(**kwargs: object) -> MsGraphCalendarRecurrenceRange:
    try:
        return MsGraphCalendarRecurrenceRange(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None


def _safe_construct_recurrence(**kwargs: object) -> MsGraphCalendarRecurrence:
    try:
        return MsGraphCalendarRecurrence(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None


def _safe_construct_event_content(**kwargs: object) -> MsGraphCalendarEventContent:
    try:
        return MsGraphCalendarEventContent(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None


def parse_msgraph_calendar_participant(payload: object) -> MsGraphCalendarParticipant:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    email_address = payload.get("emailAddress")
    if not isinstance(email_address, dict):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    if "address" not in email_address:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    try:
        address = _validate_participant_address(email_address.get("address"))
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    if "name" not in email_address:
        display_name = None
    else:
        name_value = email_address.get("name")
        try:
            display_name = _validate_participant_display_name(name_value)
        except ValueError:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    return _safe_construct_participant(display_name=display_name, address=address)


def _parse_response_status(payload: object) -> MsGraphCalendarResponseStatus:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    if "response" not in payload:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    response = _map_response_type(payload.get("response"))
    responded_at: datetime | None = None
    if "time" in payload:
        time_value = payload.get("time")
        if time_value is not None:
            responded_at = _parse_content_timezone_aware_datetime(time_value)
    return _safe_construct_response_status(response=response, responded_at=responded_at)


def parse_msgraph_calendar_attendee(payload: object) -> MsGraphCalendarAttendee:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    if "emailAddress" not in payload or "type" not in payload or "status" not in payload:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    try:
        participant = parse_msgraph_calendar_participant(payload)
        attendee_type = _map_attendee_type(payload.get("type"))
        status = _parse_response_status(payload.get("status"))
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    return _safe_construct_attendee(
        participant=participant,
        attendee_type=attendee_type,
        status=status,
    )


def parse_msgraph_calendar_location(payload: object) -> MsGraphCalendarLocation:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    if "displayName" not in payload or "locationType" not in payload:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    try:
        display_name = _validate_location_display_name(payload.get("displayName"))
        location_type = _map_location_type(payload.get("locationType"))
        street: str | None = None
        city: str | None = None
        state: str | None = None
        country_or_region: str | None = None
        postal_code: str | None = None
        if "address" in payload:
            address = payload.get("address")
            if address is not None:
                if not isinstance(address, dict):
                    raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
                street = _validate_location_field(address.get("street"))
                city = _validate_location_field(address.get("city"))
                state = _validate_location_field(address.get("state"))
                country_or_region = _validate_location_field(address.get("countryOrRegion"))
                postal_code = _validate_location_field(address.get("postalCode"))
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    return _safe_construct_location(
        display_name=display_name,
        location_type=location_type,
        street=street,
        city=city,
        state=state,
        country_or_region=country_or_region,
        postal_code=postal_code,
    )


def _parse_days_of_week_list(value: object) -> tuple[MsGraphCalendarDayOfWeek, ...]:
    if not isinstance(value, list):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
    return tuple(_map_day_of_week(item) for item in value)


def _parse_recurrence_pattern(payload: object) -> MsGraphCalendarRecurrencePattern:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    if "type" not in payload or "interval" not in payload:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    try:
        pattern_type = _map_recurrence_pattern_type(payload.get("type"))
        interval = payload.get("interval")
        if type(interval) is not int:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        month: int | None = None
        if "month" in payload and payload.get("month") is not None:
            month = payload.get("month")
        day_of_month: int | None = None
        if "dayOfMonth" in payload and payload.get("dayOfMonth") is not None:
            day_of_month = payload.get("dayOfMonth")
        days_of_week: tuple[MsGraphCalendarDayOfWeek, ...] = ()
        if "daysOfWeek" in payload:
            days_of_week = _parse_days_of_week_list(payload.get("daysOfWeek"))
        first_day_of_week: MsGraphCalendarDayOfWeek | None = None
        if "firstDayOfWeek" in payload and payload.get("firstDayOfWeek") is not None:
            first_day_of_week = _map_day_of_week(payload.get("firstDayOfWeek"))
        index: MsGraphCalendarWeekIndex | None = None
        if "index" in payload and payload.get("index") is not None:
            index = _map_week_index(payload.get("index"))
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    return _safe_construct_recurrence_pattern(
        pattern_type=pattern_type,
        interval=interval,
        month=month,
        day_of_month=day_of_month,
        days_of_week=days_of_week,
        first_day_of_week=first_day_of_week,
        index=index,
    )


def _parse_recurrence_range(payload: object) -> MsGraphCalendarRecurrenceRange:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    if "type" not in payload or "startDate" not in payload:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    try:
        range_type = _map_recurrence_range_type(payload.get("type"))
        start_date = _parse_date_value(payload.get("startDate"))
        end_date: date | None = None
        if "endDate" in payload and payload.get("endDate") is not None:
            end_date = _parse_date_value(payload.get("endDate"))
        number_of_occurrences: int | None = None
        if "numberOfOccurrences" in payload and payload.get("numberOfOccurrences") is not None:
            number_of_occurrences = payload.get("numberOfOccurrences")
        recurrence_time_zone: str | None = None
        if "recurrenceTimeZone" in payload and payload.get("recurrenceTimeZone") is not None:
            recurrence_time_zone = _validate_optional_opaque_string(
                payload.get("recurrenceTimeZone"),
                max_length=_MAX_TIME_ZONE_LEN,
            )
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    return _safe_construct_recurrence_range(
        range_type=range_type,
        start_date=start_date,
        end_date=end_date,
        number_of_occurrences=number_of_occurrences,
        recurrence_time_zone=recurrence_time_zone,
    )


def parse_msgraph_calendar_recurrence(payload: object) -> MsGraphCalendarRecurrence:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    if "pattern" not in payload or "range" not in payload:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    try:
        pattern = _parse_recurrence_pattern(payload.get("pattern"))
        recurrence_range = _parse_recurrence_range(payload.get("range"))
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    return _safe_construct_recurrence(pattern=pattern, range=recurrence_range)


def _parse_body_field(payload: object) -> tuple[MsGraphCalendarBodyKind, str]:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    if "contentType" not in payload or "content" not in payload:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    content_type = payload.get("contentType")
    content = payload.get("content")
    if not isinstance(content_type, str) or not content_type.strip():
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    normalized_type = content_type.strip().lower()
    if normalized_type == "text":
        body_kind = MsGraphCalendarBodyKind.TEXT
    elif normalized_type == "html":
        body_kind = MsGraphCalendarBodyKind.HTML
    else:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    if not isinstance(content, str):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    if "\x00" in content:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    return body_kind, content


def _parse_optional_location(payload: dict[str, object], key: str) -> MsGraphCalendarLocation | None:
    if key not in payload:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    value = payload.get(key)
    if value is None:
        return None
    return parse_msgraph_calendar_location(value)


def _parse_location_list(payload: dict[str, object], key: str) -> tuple[MsGraphCalendarLocation, ...]:
    if key not in payload:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    return tuple(parse_msgraph_calendar_location(item) for item in value)


def _parse_attendee_list(payload: dict[str, object], key: str) -> tuple[MsGraphCalendarAttendee, ...]:
    if key not in payload:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    return tuple(parse_msgraph_calendar_attendee(item) for item in value)


def _parse_event_id_list(payload: dict[str, object], key: str) -> tuple[str, ...]:
    if key not in payload:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    return tuple(validate_msgraph_calendar_event_id(item) for item in value)


def _parse_category_list(payload: dict[str, object], key: str) -> tuple[str, ...]:
    if key not in payload:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    return tuple(_validate_category(item) for item in value)


def _enforce_content_limit(body_content: str, *, max_chars: int) -> None:
    if len(body_content) > max_chars:
        raise MsGraphCalendarEventContentTooLarge() from None


def _require_active_event(event: object) -> MsGraphCalendarEventChange:
    validated = validate_msgraph_calendar_event_change(event)
    if validated.kind is not MsGraphCalendarEventChangeKind.ACTIVE:
        raise MsGraphCalendarEventChanged() from None
    return validated


def _compare_event_content_observation(
    payload: dict[str, object],
    *,
    event: MsGraphCalendarEventChange,
) -> None:
    try:
        response_id = validate_msgraph_calendar_event_id(payload.get("id"))
        response_change_key = _validate_content_revision(payload.get("changeKey"))
    except ValueError:
        raise MsGraphCalendarEventChanged() from None
    if response_id != event.remote_id or response_change_key != event.change_key:
        raise MsGraphCalendarEventChanged() from None


def parse_msgraph_calendar_event_content(
    payload: object,
    *,
    expected_mailbox_user_id: str,
    expected_calendar_id: str,
    max_chars: int,
) -> MsGraphCalendarEventContent:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None

    validated_max_chars = _validate_max_chars(max_chars)

    try:
        canonical = parse_msgraph_calendar_event_change(
            payload,
            expected_mailbox_user_id=expected_mailbox_user_id,
            expected_calendar_id=expected_calendar_id,
        )
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None

    if canonical.kind is not MsGraphCalendarEventChangeKind.ACTIVE:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    if canonical.event_type is None or canonical.change_key is None:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    if canonical.start_at is None or canonical.end_at is None or canonical.last_modified_at is None:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None

    required_content_keys = (
        "subject",
        "body",
        "bodyPreview",
        "originalStartTimeZone",
        "originalEndTimeZone",
        "createdDateTime",
        "organizer",
        "attendees",
        "location",
        "locations",
        "recurrence",
        "cancelledOccurrences",
        "categories",
        "importance",
        "sensitivity",
        "showAs",
        "responseStatus",
        "isOrganizer",
        "onlineMeetingProvider",
        "hideAttendees",
        "allowNewTimeProposals",
        "responseRequested",
        "isReminderOn",
        "reminderMinutesBeforeStart",
    )
    for key in required_content_keys:
        if key not in payload:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None

    try:
        if "body" not in payload:
            raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE)
        body_kind, body_content = _parse_body_field(payload.get("body"))
        _enforce_content_limit(body_content, max_chars=validated_max_chars)

        subject = _validate_subject(payload.get("subject"))
        body_preview = _validate_body_preview(payload.get("bodyPreview"))

        original_start_time_zone = _validate_optional_opaque_string(
            payload.get("originalStartTimeZone"),
            max_length=_MAX_TIME_ZONE_LEN,
        )
        original_end_time_zone = _validate_optional_opaque_string(
            payload.get("originalEndTimeZone"),
            max_length=_MAX_TIME_ZONE_LEN,
        )
        created_at = _parse_content_timezone_aware_datetime(payload.get("createdDateTime"))

        organizer_value = payload.get("organizer")
        organizer = (
            parse_msgraph_calendar_participant(organizer_value)
            if organizer_value is not None
            else None
        )
        attendees = _parse_attendee_list(payload, "attendees")
        location = _parse_optional_location(payload, "location")
        locations = _parse_location_list(payload, "locations")

        recurrence: MsGraphCalendarRecurrence | None = None
        recurrence_value = payload.get("recurrence")
        if recurrence_value is not None:
            recurrence = parse_msgraph_calendar_recurrence(recurrence_value)

        cancelled_occurrence_ids = _parse_event_id_list(payload, "cancelledOccurrences")
        categories = _parse_category_list(payload, "categories")

        importance = _map_importance(payload.get("importance"))
        sensitivity = _map_sensitivity(payload.get("sensitivity"))
        show_as = _map_show_as(payload.get("showAs"))
        response_status = _parse_response_status(payload.get("responseStatus"))

        is_organizer = _parse_required_bool(payload, "isOrganizer")
        online_meeting_provider = _map_online_meeting_provider(
            payload.get("onlineMeetingProvider")
        )
        hide_attendees = _parse_required_bool(payload, "hideAttendees")
        allow_new_time_proposals = _parse_required_bool(payload, "allowNewTimeProposals")
        response_requested = _parse_required_bool(payload, "responseRequested")
        is_reminder_on = _parse_required_bool(payload, "isReminderOn")
        reminder_minutes_before_start = _parse_required_int(
            payload,
            "reminderMinutesBeforeStart",
        )
    except MsGraphCalendarEventContentTooLarge:
        raise
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None

    if canonical.is_all_day is None or canonical.is_cancelled is None:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    if canonical.is_draft is None or canonical.has_attachments is None:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    if canonical.is_online_meeting is None:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None

    return _safe_construct_event_content(
        mailbox_user_id=canonical.mailbox_user_id,
        calendar_remote_id=canonical.calendar_remote_id,
        remote_id=canonical.remote_id,
        content_revision=canonical.change_key,
        event_type=canonical.event_type,
        subject=subject,
        body_kind=body_kind,
        body_content=body_content,
        body_preview=body_preview,
        start_at=canonical.start_at,
        end_at=canonical.end_at,
        original_start_at=canonical.original_start_at,
        original_start_time_zone=original_start_time_zone,
        original_end_time_zone=original_end_time_zone,
        created_at=created_at,
        last_modified_at=canonical.last_modified_at,
        organizer=organizer,
        attendees=attendees,
        location=location,
        locations=locations,
        recurrence=recurrence,
        series_master_id=canonical.series_master_id,
        cancelled_occurrence_ids=cancelled_occurrence_ids,
        categories=categories,
        i_cal_uid=canonical.i_cal_uid,
        importance=importance,
        sensitivity=sensitivity,
        show_as=show_as,
        response_status=response_status,
        is_all_day=canonical.is_all_day,
        is_cancelled=canonical.is_cancelled,
        is_draft=canonical.is_draft,
        is_organizer=is_organizer,
        is_online_meeting=canonical.is_online_meeting,
        has_attachments=canonical.has_attachments,
        hide_attendees=hide_attendees,
        allow_new_time_proposals=allow_new_time_proposals,
        response_requested=response_requested,
        is_reminder_on=is_reminder_on,
        reminder_minutes_before_start=reminder_minutes_before_start,
        online_meeting_provider=online_meeting_provider,
    )


def read_and_validate_current_calendar_event_observation(
    *,
    event: MsGraphCalendarEventChange,
    transport: MsGraphKnowledgeTransport,
) -> MsGraphCalendarEventChange:
    validated_event = _require_active_event(event)
    quoted_mailbox = quote(validated_event.mailbox_user_id, safe="")
    quoted_calendar = quote(validated_event.calendar_remote_id, safe="")
    quoted_event_id = quote(validated_event.remote_id, safe="")
    path = (
        f"/users/{quoted_mailbox}/calendars/{quoted_calendar}/events/{quoted_event_id}"
    )
    payload = transport.get_initial_json(
        path=path,
        params={"$select": _OBSERVATION_SELECT},
        headers=_OBSERVATION_HEADERS,
        not_found_is_dependency=True,
    )
    _compare_event_content_observation(payload, event=validated_event)
    observed = parse_msgraph_calendar_event_change(
        payload,
        expected_mailbox_user_id=validated_event.mailbox_user_id,
        expected_calendar_id=validated_event.calendar_remote_id,
    )
    if observed.kind is not MsGraphCalendarEventChangeKind.ACTIVE:
        raise MsGraphCalendarEventChanged() from None
    if (
        observed.mailbox_user_id != validated_event.mailbox_user_id
        or observed.calendar_remote_id != validated_event.calendar_remote_id
        or observed.remote_id != validated_event.remote_id
        or observed.change_key != validated_event.change_key
    ):
        raise MsGraphCalendarEventChanged() from None
    return observed


def validate_msgraph_calendar_participant(value: object) -> MsGraphCalendarParticipant:
    if isinstance(value, MsGraphCalendarParticipant):
        source: object = value.model_dump(mode="python")
    elif isinstance(value, dict):
        source = value
    else:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    try:
        return MsGraphCalendarParticipant.model_validate(source)
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None


def validate_msgraph_calendar_attendee(value: object) -> MsGraphCalendarAttendee:
    if isinstance(value, MsGraphCalendarAttendee):
        source: object = value.model_dump(mode="python")
    elif isinstance(value, dict):
        source = value
    else:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    try:
        validated = MsGraphCalendarAttendee.model_validate(source)
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    return MsGraphCalendarAttendee(
        participant=validate_msgraph_calendar_participant(validated.participant),
        attendee_type=validated.attendee_type,
        status=MsGraphCalendarResponseStatus.model_validate(
            validated.status.model_dump(mode="python")
        ),
    )


def validate_msgraph_calendar_location(value: object) -> MsGraphCalendarLocation:
    if isinstance(value, MsGraphCalendarLocation):
        source: object = value.model_dump(mode="python")
    elif isinstance(value, dict):
        source = value
    else:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    try:
        return MsGraphCalendarLocation.model_validate(source)
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None


def validate_msgraph_calendar_recurrence(value: object) -> MsGraphCalendarRecurrence:
    if isinstance(value, MsGraphCalendarRecurrence):
        source: object = value.model_dump(mode="python")
    elif isinstance(value, dict):
        source = value
    else:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    try:
        validated = MsGraphCalendarRecurrence.model_validate(source)
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    return MsGraphCalendarRecurrence(
        pattern=MsGraphCalendarRecurrencePattern.model_validate(
            validated.pattern.model_dump(mode="python")
        ),
        range=MsGraphCalendarRecurrenceRange.model_validate(
            validated.range.model_dump(mode="python")
        ),
    )


def validate_msgraph_calendar_event_content(
    value: object,
    *,
    event: MsGraphCalendarEventChange,
    max_chars: int,
) -> MsGraphCalendarEventContent:
    validated_max_chars = _validate_max_chars(max_chars)

    if not isinstance(value, MsGraphCalendarEventContent):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None

    validated_event = _require_active_event(event)

    try:
        raw = value.model_dump(mode="python")
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None

    try:
        mailbox_user_id = validate_msgraph_mailbox_user_id(raw["mailbox_user_id"])
        calendar_remote_id = validate_msgraph_calendar_id(raw["calendar_remote_id"])
        remote_id = validate_msgraph_calendar_event_id(raw["remote_id"])
        content_revision = _validate_content_revision(raw["content_revision"])
        body_content = _validate_body_text(raw["body_content"])
        organizer_value = raw["organizer"]
        organizer = (
            validate_msgraph_calendar_participant(organizer_value)
            if organizer_value is not None
            else None
        )
        attendees = _validate_attendee_tuple(raw["attendees"])
        location_value = raw["location"]
        location = (
            validate_msgraph_calendar_location(location_value)
            if location_value is not None
            else None
        )
        locations = _validate_location_tuple(raw["locations"])
        recurrence_value = raw["recurrence"]
        recurrence = (
            validate_msgraph_calendar_recurrence(recurrence_value)
            if recurrence_value is not None
            else None
        )
        series_master_id = raw["series_master_id"]
        if series_master_id is not None:
            series_master_id = validate_msgraph_calendar_event_id(series_master_id)
        cancelled_occurrence_ids = _validate_event_id_tuple(raw["cancelled_occurrence_ids"])
        categories = _validate_category_tuple(raw["categories"])
        response_status = MsGraphCalendarResponseStatus.model_validate(
            raw["response_status"]
        )
    except KeyError:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None

    if (
        mailbox_user_id != validated_event.mailbox_user_id
        or calendar_remote_id != validated_event.calendar_remote_id
        or remote_id != validated_event.remote_id
        or content_revision != validated_event.change_key
    ):
        raise MsGraphCalendarEventChanged() from None

    _enforce_content_limit(body_content, max_chars=validated_max_chars)

    try:
        return _safe_construct_event_content(
            mailbox_user_id=validated_event.mailbox_user_id,
            calendar_remote_id=validated_event.calendar_remote_id,
            remote_id=validated_event.remote_id,
            content_revision=validated_event.change_key,
            event_type=raw["event_type"],
            subject=raw.get("subject"),
            body_kind=raw["body_kind"],
            body_content=body_content,
            body_preview=raw.get("body_preview"),
            start_at=raw["start_at"],
            end_at=raw["end_at"],
            original_start_at=raw.get("original_start_at"),
            original_start_time_zone=raw.get("original_start_time_zone"),
            original_end_time_zone=raw.get("original_end_time_zone"),
            created_at=raw["created_at"],
            last_modified_at=raw["last_modified_at"],
            organizer=organizer,
            attendees=attendees,
            location=location,
            locations=locations,
            recurrence=recurrence,
            series_master_id=series_master_id,
            cancelled_occurrence_ids=cancelled_occurrence_ids,
            categories=categories,
            i_cal_uid=raw.get("i_cal_uid"),
            importance=raw["importance"],
            sensitivity=raw["sensitivity"],
            show_as=raw["show_as"],
            response_status=response_status,
            is_all_day=raw["is_all_day"],
            is_cancelled=raw["is_cancelled"],
            is_draft=raw["is_draft"],
            is_organizer=raw["is_organizer"],
            is_online_meeting=raw["is_online_meeting"],
            has_attachments=raw["has_attachments"],
            hide_attendees=raw["hide_attendees"],
            allow_new_time_proposals=raw["allow_new_time_proposals"],
            response_requested=raw["response_requested"],
            is_reminder_on=raw["is_reminder_on"],
            reminder_minutes_before_start=raw["reminder_minutes_before_start"],
            online_meeting_provider=raw["online_meeting_provider"],
        )
    except MsGraphCalendarEventContentTooLarge:
        raise
    except ValueError:
        raise ValueError(_MALFORMED_CALENDAR_EVENT_CONTENT_RESPONSE) from None


def _validate_max_chars(max_chars: object) -> int:
    if type(max_chars) is not int:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_EVENT_CONTENT_REQUEST) from None
    if max_chars < 1 or max_chars > ABSOLUTE_CALENDAR_EVENT_CONTENT_MAX_CHARS:
        raise IntegrationConfigurationError(_INVALID_CALENDAR_EVENT_CONTENT_REQUEST) from None
    return max_chars


@runtime_checkable
class MsGraphCalendarContentReadClient(Protocol):
    def read_calendar_event_content(
        self,
        *,
        event: MsGraphCalendarEventChange,
        max_chars: int,
    ) -> MsGraphCalendarEventContent:
        ...


class MsGraphCalendarContentReader:
    """Calendar event text content and participants reader."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        transport: MsGraphKnowledgeTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    def read_calendar_event_content(
        self,
        *,
        event: MsGraphCalendarEventChange,
        max_chars: int,
    ) -> MsGraphCalendarEventContent:
        validated_event = _require_active_event(event)
        validated_max_chars = _validate_max_chars(max_chars)

        quoted_mailbox = quote(validated_event.mailbox_user_id, safe="")
        quoted_calendar = quote(validated_event.calendar_remote_id, safe="")
        quoted_event_id = quote(validated_event.remote_id, safe="")
        path = (
            f"/users/{quoted_mailbox}/calendars/{quoted_calendar}/events/{quoted_event_id}"
        )
        payload = self._transport.get_initial_json(
            path=path,
            params={"$select": _CONTENT_SELECT},
            headers=_IMMUTABLE_TEXT_BODY_HEADERS,
            not_found_is_dependency=True,
        )

        _compare_event_content_observation(payload, event=validated_event)

        content = parse_msgraph_calendar_event_content(
            payload,
            expected_mailbox_user_id=validated_event.mailbox_user_id,
            expected_calendar_id=validated_event.calendar_remote_id,
            max_chars=validated_max_chars,
        )

        return validate_msgraph_calendar_event_content(
            content,
            event=validated_event,
            max_chars=validated_max_chars,
        )
