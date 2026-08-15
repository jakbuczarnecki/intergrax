# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Calendar knowledge-read: one-calendar structured event pages."""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import StrEnum
from typing import Protocol, TypeVar, runtime_checkable
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceSourceKind,
    GoogleWorkspaceTransport,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
    GoogleWorkspacePageToken,
)

GOOGLE_CALENDAR_SOURCE_KIND = "calendar"

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)

_INVALID_CALENDAR_ID_MESSAGE = "invalid Google Calendar identifier"
_INVALID_EVENT_ID_MESSAGE = "invalid Google Calendar event identifier"
_INVALID_PAGE_TOKEN_MESSAGE = "invalid Google Calendar page token"
_INVALID_SYNC_TOKEN_MESSAGE = "invalid Google Calendar sync token"
_INVALID_PAGE_LIMIT_MESSAGE = "invalid Google Calendar page limit"
_REQUEST_FAILED_MESSAGE = "Google Calendar provider request failed"
_UNEXPECTED_RESPONSE_MESSAGE = "unexpected Google Calendar provider response"

_MAX_CALENDAR_ID_LENGTH = 1024
_MAX_PAGE_TOKEN_LENGTH = 4096
_MAX_SYNC_TOKEN_LENGTH = 4096
_MAX_EVENT_PAGE_SIZE = 2500
_MAX_TEXT_LENGTH = 16_384
_MAX_DESCRIPTION_LENGTH = 100_000
_MAX_TOTAL_TEXT_CHARS = 4_000_000
_MAX_EVENTS = 2500
_MAX_ATTENDEES = 5000
_MAX_RECURRENCE_LINES = 128
_MAX_RECURRENCE_LINE_LENGTH = 4096
_MAX_REMINDER_OVERRIDES = 32
_MAX_CONFERENCE_ENTRY_POINTS = 32

_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_RFC3339_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})$"
)
_CALENDAR_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_GOOGLE_CALENDAR_EVENT_FIELDS = (
    "id,iCalUID,etag,status,eventType,summary,description,location,htmlLink,"
    "created,updated,colorId,visibility,transparency,sequence,start,end,"
    "endTimeUnspecified,recurrence,recurringEventId,originalStartTime,"
    "creator,organizer,attendees,attendeesOmitted,guestsCanInviteOthers,"
    "guestsCanModify,guestsCanSeeOtherGuests,privateCopy,locked,hangoutLink,"
    "conferenceData(conferenceId,signature,notes,"
    "entryPoints(entryPointType,uri,label,pin,accessCode,meetingCode,passcode,password),"
    "conferenceSolution(key(type),name,iconUri)),"
    "reminders(useDefault,overrides(method,minutes))"
)
_GOOGLE_CALENDAR_EVENTS_FIELDS = (
    "summary,description,updated,timeZone,accessRole,nextPageToken,nextSyncToken,"
    f"items({_GOOGLE_CALENDAR_EVENT_FIELDS})"
)

_EVENT_ALLOWED_KEYS = frozenset(
    {
        "id",
        "iCalUID",
        "etag",
        "status",
        "eventType",
        "summary",
        "description",
        "location",
        "htmlLink",
        "created",
        "updated",
        "colorId",
        "visibility",
        "transparency",
        "sequence",
        "start",
        "end",
        "endTimeUnspecified",
        "recurrence",
        "recurringEventId",
        "originalStartTime",
        "creator",
        "organizer",
        "attendees",
        "attendeesOmitted",
        "guestsCanInviteOthers",
        "guestsCanModify",
        "guestsCanSeeOtherGuests",
        "privateCopy",
        "locked",
        "hangoutLink",
        "conferenceData",
        "reminders",
    }
)
_PERSON_ALLOWED_KEYS = frozenset({"id", "email", "displayName", "self"})
_ATTENDEE_ALLOWED_KEYS = frozenset(
    {
        "id",
        "email",
        "displayName",
        "organizer",
        "self",
        "resource",
        "optional",
        "responseStatus",
        "comment",
        "additionalGuests",
    }
)
_DATE_TIME_ALLOWED_KEYS = frozenset({"date", "dateTime", "timeZone"})
_REMINDER_ALLOWED_KEYS = frozenset({"method", "minutes"})
_REMINDERS_ALLOWED_KEYS = frozenset({"useDefault", "overrides"})
_ENTRY_POINT_ALLOWED_KEYS = frozenset(
    {
        "entryPointType",
        "uri",
        "label",
        "pin",
        "accessCode",
        "meetingCode",
        "passcode",
        "password",
    }
)
_CONFERENCE_SOLUTION_ALLOWED_KEYS = frozenset({"key", "name", "iconUri"})
_CONFERENCE_KEY_ALLOWED_KEYS = frozenset({"type"})
_CONFERENCE_ALLOWED_KEYS = frozenset(
    {"conferenceId", "signature", "notes", "entryPoints", "conferenceSolution"}
)
_PAGE_ALLOWED_KEYS = frozenset(
    {
        "summary",
        "description",
        "updated",
        "timeZone",
        "accessRole",
        "nextPageToken",
        "nextSyncToken",
        "items",
    }
)


class GoogleCalendarAccessRole(StrEnum):
    NONE = "none"
    FREE_BUSY_READER = "freeBusyReader"
    READER = "reader"
    WRITER = "writer"
    OWNER = "owner"


class GoogleCalendarEventStatus(StrEnum):
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class GoogleCalendarEventType(StrEnum):
    DEFAULT = "default"
    OUT_OF_OFFICE = "outOfOffice"
    FOCUS_TIME = "focusTime"
    WORKING_LOCATION = "workingLocation"
    BIRTHDAY = "birthday"
    FROM_GMAIL = "fromGmail"


class GoogleCalendarVisibility(StrEnum):
    DEFAULT = "default"
    PUBLIC = "public"
    PRIVATE = "private"
    CONFIDENTIAL = "confidential"


class GoogleCalendarTransparency(StrEnum):
    OPAQUE = "opaque"
    TRANSPARENT = "transparent"


class GoogleCalendarAttendeeResponseStatus(StrEnum):
    NEEDS_ACTION = "needsAction"
    DECLINED = "declined"
    TENTATIVE = "tentative"
    ACCEPTED = "accepted"


class GoogleCalendarConferenceSolutionType(StrEnum):
    EVENT_HANGOUT = "eventHangout"
    EVENT_NAMED_HANGOUT = "eventNamedHangout"
    HANGOUTS_MEET = "hangoutsMeet"
    ADD_ON = "addOn"


class GoogleCalendarReminderMethod(StrEnum):
    EMAIL = "email"
    POPUP = "popup"


def _validate_nonblank(value: object, *, max_length: int, message: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(message)
    if len(value) > max_length or _ASCII_CONTROL.search(value):
        raise ValueError(message)
    return value


def _validate_text(
    value: object,
    *,
    max_length: int = _MAX_TEXT_LENGTH,
    message: str = _UNEXPECTED_RESPONSE_MESSAGE,
) -> str:
    if type(value) is not str or len(value) > max_length or _ASCII_CONTROL.search(value):
        raise ValueError(message)
    return value


def _validate_optional_text(value: object) -> str | None:
    if value is None:
        return None
    return _validate_text(value)


def _validate_exact_bool(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


def _validate_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    return _validate_exact_bool(value)


def _validate_exact_int(value: object, *, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


def _validate_rfc3339(value: object) -> str:
    if type(value) is not str or not _RFC3339_TIMESTAMP.fullmatch(value):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


def _validate_date(value: object) -> str:
    if type(value) is not str or not _CALENDAR_DATE.fullmatch(value):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE) from None
    return value


def _validate_enum(value: object, enum_cls: type[StrEnum]) -> StrEnum:
    if type(value) is enum_cls:
        return value
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    try:
        return enum_cls(value)
    except ValueError:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE) from None


def _exact_tuple(value: object, item_type: type[BaseModel], *, field_name: str) -> tuple[BaseModel, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{field_name} must be a tuple")
    if any(type(item) is not item_type for item in value):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return tuple(_rebuild_model(item_type, item) for item in value)


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _rebuild_model(model_type: type[_ModelT], value: object) -> _ModelT:
    if type(value) is not model_type:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return model_type(**value.__dict__)


def _provider_dict(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


def _provider_list(value: object) -> list[object]:
    if type(value) is not list:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


def _reject_unknown(mapping: dict[str, object], allowed: frozenset[str]) -> None:
    if not set(mapping).issubset(allowed):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)


def _retained_text_size(value: object) -> int:
    if type(value) is str:
        return len(value)
    if type(value) is dict:
        return sum(_retained_text_size(item) for item in value.values())
    if type(value) is list:
        return sum(_retained_text_size(item) for item in value)
    return 0


def _optional_provider(
    mapping: dict[str, object],
    key: str,
    parser,
    default=None,
):
    if key not in mapping:
        return default
    value = mapping[key]
    if value is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return parser(value)


class GoogleCalendarPerson(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    id: str | None = Field(default=None, repr=False)
    email: str | None = Field(default=None, repr=False)
    display_name: str | None = Field(default=None, repr=False)
    self: bool | None = Field(default=None, repr=False)

    _validate_id = field_validator("id", "email", "display_name", mode="before")(
        _validate_optional_text
    )
    _validate_self = field_validator("self", mode="before")(_validate_optional_bool)


class GoogleCalendarAttendee(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    id: str | None = Field(default=None, repr=False)
    email: str | None = Field(default=None, repr=False)
    display_name: str | None = Field(default=None, repr=False)
    organizer: bool | None = Field(default=None, repr=False)
    self: bool | None = Field(default=None, repr=False)
    resource: bool | None = Field(default=None, repr=False)
    optional: bool | None = Field(default=None, repr=False)
    response_status: GoogleCalendarAttendeeResponseStatus | None = Field(
        default=None, repr=False
    )
    comment: str | None = Field(default=None, repr=False)
    additional_guests: int | None = Field(default=None, repr=False)

    _validate_text_fields = field_validator(
        "id", "email", "display_name", "comment", mode="before"
    )(_validate_optional_text)
    _validate_bool_fields = field_validator(
        "organizer", "self", "resource", "optional", mode="before"
    )(_validate_optional_bool)
    _validate_response = field_validator("response_status", mode="before")(
        lambda value: None if value is None else _validate_enum(
            value, GoogleCalendarAttendeeResponseStatus
        )
    )
    _validate_guests = field_validator("additional_guests", mode="before")(
        lambda value: None if value is None else _validate_exact_int(value)
    )


class GoogleCalendarEventDateTime(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    date_time: str | None = Field(default=None, repr=False)
    date: str | None = Field(default=None, repr=False)
    time_zone: str | None = Field(default=None, repr=False)

    _validate_date_time = field_validator("date_time", mode="before")(
        lambda value: None if value is None else _validate_rfc3339(value)
    )
    _validate_date = field_validator("date", mode="before")(
        lambda value: None if value is None else _validate_date(value)
    )
    _validate_zone = field_validator("time_zone", mode="before")(
        lambda value: None if value is None else _validate_text(value, max_length=256)
    )

    @model_validator(mode="after")
    def _validate_union(self) -> GoogleCalendarEventDateTime:
        if (self.date_time is None) == (self.date is None):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return self


class GoogleCalendarReminder(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    method: GoogleCalendarReminderMethod = Field(repr=False)
    minutes: int = Field(repr=False)

    _validate_method = field_validator("method", mode="before")(
        lambda value: _validate_enum(value, GoogleCalendarReminderMethod)
    )
    _validate_minutes = field_validator("minutes", mode="before")(
        lambda value: _validate_exact_int(value, maximum=40_320)
    )


class GoogleCalendarReminders(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    use_default: bool = Field(repr=False)
    overrides: tuple[GoogleCalendarReminder, ...] = Field(repr=False)

    _validate_default = field_validator("use_default", mode="before")(_validate_exact_bool)
    _validate_overrides = field_validator("overrides", mode="before")(
        lambda value: _exact_tuple(value, GoogleCalendarReminder, field_name="overrides")
    )

    @model_validator(mode="after")
    def _validate_override_pairs(self) -> GoogleCalendarReminders:
        pairs = {(item.method, item.minutes) for item in self.overrides}
        if len(pairs) != len(self.overrides) or len(self.overrides) > _MAX_REMINDER_OVERRIDES:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return self


class GoogleCalendarConferenceEntryPoint(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    entry_point_type: str = Field(repr=False)
    uri: str = Field(repr=False)
    label: str | None = Field(default=None, repr=False)
    pin: str | None = Field(default=None, repr=False)
    access_code: str | None = Field(default=None, repr=False)
    meeting_code: str | None = Field(default=None, repr=False)
    passcode: str | None = Field(default=None, repr=False)
    password: str | None = Field(default=None, repr=False)

    _validate_required = field_validator("entry_point_type", "uri", mode="before")(
        lambda value: _validate_nonblank(value, max_length=_MAX_TEXT_LENGTH, message=_UNEXPECTED_RESPONSE_MESSAGE)
    )
    _validate_optional = field_validator(
        "label", "pin", "access_code", "meeting_code", "passcode", "password", mode="before"
    )(_validate_optional_text)


class GoogleCalendarConferenceSolution(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    type: GoogleCalendarConferenceSolutionType = Field(repr=False)
    name: str = Field(repr=False)
    icon_uri: str | None = Field(default=None, repr=False)

    _validate_type = field_validator("type", mode="before")(
        lambda value: _validate_enum(value, GoogleCalendarConferenceSolutionType)
    )
    _validate_name = field_validator("name", mode="before")(
        lambda value: _validate_nonblank(value, max_length=_MAX_TEXT_LENGTH, message=_UNEXPECTED_RESPONSE_MESSAGE)
    )
    _validate_icon = field_validator("icon_uri", mode="before")(_validate_optional_text)


class GoogleCalendarConferenceData(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    conference_id: str | None = Field(default=None, repr=False)
    signature: str | None = Field(default=None, repr=False)
    notes: str | None = Field(default=None, repr=False)
    entry_points: tuple[GoogleCalendarConferenceEntryPoint, ...] = Field(repr=False)
    conference_solution: GoogleCalendarConferenceSolution | None = Field(
        default=None, repr=False
    )

    _validate_text_fields = field_validator(
        "conference_id", "signature", "notes", mode="before"
    )(_validate_optional_text)
    _validate_entry_points = field_validator("entry_points", mode="before")(
        lambda value: _exact_tuple(
            value, GoogleCalendarConferenceEntryPoint, field_name="entry_points"
        )
    )
    _validate_solution = field_validator("conference_solution", mode="before")(
        lambda value: None
        if value is None
        else _rebuild_model(GoogleCalendarConferenceSolution, value)
    )

    @model_validator(mode="after")
    def _validate_bounds(self) -> GoogleCalendarConferenceData:
        if len(self.entry_points) > _MAX_CONFERENCE_ENTRY_POINTS:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return self


class GoogleCalendarSyncToken(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    value: str = Field(repr=False)

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: object) -> str:
        return _validate_nonblank(
            value, max_length=_MAX_SYNC_TOKEN_LENGTH, message=_INVALID_SYNC_TOKEN_MESSAGE
        )


class GoogleCalendarEvent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    id: str = Field(repr=False)
    i_cal_uid: str | None = Field(default=None, repr=False)
    etag: str | None = Field(default=None, repr=False)
    status: GoogleCalendarEventStatus = Field(repr=False)
    event_type: GoogleCalendarEventType | None = Field(default=None, repr=False)
    summary: str | None = Field(default=None, repr=False)
    description: str | None = Field(default=None, repr=False)
    location: str | None = Field(default=None, repr=False)
    html_link: str | None = Field(default=None, repr=False)
    created: str | None = Field(default=None, repr=False)
    updated: str | None = Field(default=None, repr=False)
    color_id: str | None = Field(default=None, repr=False)
    visibility: GoogleCalendarVisibility | None = Field(default=None, repr=False)
    transparency: GoogleCalendarTransparency | None = Field(default=None, repr=False)
    sequence: int | None = Field(default=None, repr=False)
    start: GoogleCalendarEventDateTime | None = Field(default=None, repr=False)
    end: GoogleCalendarEventDateTime | None = Field(default=None, repr=False)
    end_time_unspecified: bool | None = Field(default=None, repr=False)
    recurrence: tuple[str, ...] = Field(default=(), repr=False)
    recurring_event_id: str | None = Field(default=None, repr=False)
    original_start_time: GoogleCalendarEventDateTime | None = Field(default=None, repr=False)
    creator: GoogleCalendarPerson | None = Field(default=None, repr=False)
    organizer: GoogleCalendarPerson | None = Field(default=None, repr=False)
    attendees: tuple[GoogleCalendarAttendee, ...] = Field(default=(), repr=False)
    attendees_omitted: bool | None = Field(default=None, repr=False)
    guests_can_invite_others: bool | None = Field(default=None, repr=False)
    guests_can_modify: bool | None = Field(default=None, repr=False)
    guests_can_see_other_guests: bool | None = Field(default=None, repr=False)
    private_copy: bool | None = Field(default=None, repr=False)
    locked: bool | None = Field(default=None, repr=False)
    hangout_link: str | None = Field(default=None, repr=False)
    conference_data: GoogleCalendarConferenceData | None = Field(default=None, repr=False)
    reminders: GoogleCalendarReminders | None = Field(default=None, repr=False)

    _validate_id = field_validator("id", mode="before")(
        lambda value: _validate_nonblank(value, max_length=_MAX_TEXT_LENGTH, message=_UNEXPECTED_RESPONSE_MESSAGE)
    )
    _validate_optional_text_fields = field_validator(
        "i_cal_uid", "etag", "summary", "location", "html_link", "color_id", "recurring_event_id", "hangout_link",
        mode="before",
    )(_validate_optional_text)
    _validate_description = field_validator("description", mode="before")(
        lambda value: None
        if value is None
        else _validate_text(value, max_length=_MAX_DESCRIPTION_LENGTH)
    )
    _validate_timestamps = field_validator("created", "updated", mode="before")(
        lambda value: None if value is None else _validate_rfc3339(value)
    )
    _validate_status = field_validator("status", mode="before")(
        lambda value: _validate_enum(value, GoogleCalendarEventStatus)
    )
    _validate_event_type = field_validator("event_type", mode="before")(
        lambda value: None if value is None else _validate_enum(value, GoogleCalendarEventType)
    )
    _validate_visibility = field_validator("visibility", mode="before")(
        lambda value: None if value is None else _validate_enum(value, GoogleCalendarVisibility)
    )
    _validate_transparency = field_validator("transparency", mode="before")(
        lambda value: None if value is None else _validate_enum(value, GoogleCalendarTransparency)
    )
    _validate_sequence = field_validator("sequence", mode="before")(
        lambda value: None if value is None else _validate_exact_int(value)
    )
    _validate_dates = field_validator(
        "start", "end", "original_start_time", mode="before"
    )(
        lambda value: None
        if value is None
        else _rebuild_model(GoogleCalendarEventDateTime, value)
    )
    _validate_bool_fields = field_validator(
        "end_time_unspecified",
        "attendees_omitted",
        "guests_can_invite_others",
        "guests_can_modify",
        "guests_can_see_other_guests",
        "private_copy",
        "locked",
        mode="before",
    )(_validate_optional_bool)
    _validate_recurrence = field_validator("recurrence", mode="before")(
        lambda value: _validate_recurrence_tuple(value)
    )
    _validate_people = field_validator("creator", "organizer", mode="before")(
        lambda value: None if value is None else _rebuild_model(GoogleCalendarPerson, value)
    )
    _validate_attendees = field_validator("attendees", mode="before")(
        lambda value: _exact_tuple(value, GoogleCalendarAttendee, field_name="attendees")
    )
    _validate_conference = field_validator("conference_data", mode="before")(
        lambda value: None if value is None else _rebuild_model(GoogleCalendarConferenceData, value)
    )
    _validate_reminders = field_validator("reminders", mode="before")(
        lambda value: None if value is None else _rebuild_model(GoogleCalendarReminders, value)
    )

    @model_validator(mode="after")
    def _validate_event_invariants(self) -> GoogleCalendarEvent:
        if len(self.attendees) > _MAX_ATTENDEES or len(self.recurrence) > _MAX_RECURRENCE_LINES:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if self.status is not GoogleCalendarEventStatus.CANCELLED:
            if self.start is None or self.end is None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if self.start is not None and self.end is not None:
            if (self.start.date_time is None) != (self.end.date_time is None):
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            if self.start.date_time is not None:
                if _parse_comparable_datetime(self.end.date_time) <= _parse_comparable_datetime(
                    self.start.date_time
                ):
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            elif self.end.date <= self.start.date:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return self


def _validate_recurrence_tuple(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError("recurrence must be a tuple")
    if len(value) > _MAX_RECURRENCE_LINES:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    result: list[str] = []
    for line in value:
        if type(line) is not str or not line or len(line) > _MAX_RECURRENCE_LINE_LENGTH:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if _ASCII_CONTROL.search(line):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        result.append(line)
    return tuple(result)


def _parse_comparable_datetime(value: str | None) -> datetime:
    if value is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)


class GoogleCalendarEventPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    calendar_id: str = Field(repr=False)
    summary: str | None = Field(default=None, repr=False)
    description: str | None = Field(default=None, repr=False)
    updated: str | None = Field(default=None, repr=False)
    time_zone: str | None = Field(default=None, repr=False)
    access_role: GoogleCalendarAccessRole | None = Field(default=None, repr=False)
    events: tuple[GoogleCalendarEvent, ...] = Field(repr=False)
    next_page_token: GoogleWorkspacePageToken | None = Field(default=None, repr=False)
    next_sync_token: GoogleCalendarSyncToken | None = Field(default=None, repr=False)

    _validate_calendar_id = field_validator("calendar_id", mode="before")(
        lambda value: _validate_page_calendar_id(value)
    )
    _validate_metadata = field_validator("summary", "time_zone", mode="before")(
        _validate_optional_text
    )
    _validate_description = field_validator("description", mode="before")(
        lambda value: None
        if value is None
        else _validate_text(value, max_length=_MAX_DESCRIPTION_LENGTH)
    )
    _validate_updated = field_validator("updated", mode="before")(
        lambda value: None if value is None else _validate_rfc3339(value)
    )
    _validate_role = field_validator("access_role", mode="before")(
        lambda value: None if value is None else _validate_enum(value, GoogleCalendarAccessRole)
    )
    _validate_events = field_validator("events", mode="before")(
        lambda value: _exact_tuple(value, GoogleCalendarEvent, field_name="events")
    )
    _validate_page_token = field_validator("next_page_token", mode="before")(
        lambda value: _rebuild_page_token(value)
    )
    _validate_sync_token = field_validator("next_sync_token", mode="before")(
        lambda value: None if value is None else _rebuild_sync_token(value)
    )

    @model_validator(mode="after")
    def _validate_page_invariants(self) -> GoogleCalendarEventPage:
        if len(self.events) > _MAX_EVENTS:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if (self.next_page_token is None) == (self.next_sync_token is None):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        ids = [event.id for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return self


def _rebuild_page_token(value: object) -> GoogleWorkspacePageToken | None:
    if value is None:
        return None
    if type(value) is not GoogleWorkspacePageToken or type(value.value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if (
        not value.value
        or value.value != value.value.strip()
        or len(value.value) > _MAX_PAGE_TOKEN_LENGTH
        or _ASCII_CONTROL.search(value.value)
    ):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return GoogleWorkspacePageToken(value=value.value)


def _rebuild_sync_token(value: object) -> GoogleCalendarSyncToken:
    if type(value) is not GoogleCalendarSyncToken:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return GoogleCalendarSyncToken(value=value.value)


@runtime_checkable
class GoogleCalendarKnowledgeReadClient(Protocol):
    def list_events_page(
        self,
        *,
        calendar_id: str,
        page_token: GoogleWorkspacePageToken | None = None,
        sync_token: GoogleCalendarSyncToken | None = None,
        max_results: int = 250,
    ) -> GoogleCalendarEventPage:
        ...

    def read_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
    ) -> GoogleCalendarEvent:
        ...


def _validate_calendar_id_for_request(value: object) -> str:
    validated = _validate_nonblank(
        value, max_length=_MAX_CALENDAR_ID_LENGTH, message=_INVALID_CALENDAR_ID_MESSAGE
    )
    if "/" in validated or "\\" in validated:
        raise ValueError(_INVALID_CALENDAR_ID_MESSAGE)
    return validated


def _validate_event_id_for_request(value: object) -> str:
    validated = _validate_nonblank(
        value, max_length=_MAX_CALENDAR_ID_LENGTH, message=_INVALID_EVENT_ID_MESSAGE
    )
    if "/" in validated or "\\" in validated:
        raise ValueError(_INVALID_EVENT_ID_MESSAGE)
    return validated


def _validate_page_calendar_id(value: object) -> str:
    return _validate_calendar_id_for_request(value)


def _validate_page_token_for_request(value: object) -> GoogleWorkspacePageToken | None:
    if value is None:
        return None
    try:
        return _rebuild_page_token(value)
    except Exception:
        raise ValueError(_INVALID_PAGE_TOKEN_MESSAGE) from None


def _validate_sync_token_for_request(value: object) -> GoogleCalendarSyncToken | None:
    if value is None:
        return None
    try:
        return _rebuild_sync_token(value)
    except Exception:
        raise ValueError(_INVALID_SYNC_TOKEN_MESSAGE) from None


def _parse_enum(mapping: dict[str, object], key: str, enum_cls: type[StrEnum]):
    return _optional_provider(mapping, key, lambda value: _validate_enum(value, enum_cls))


def _parse_person(value: object) -> GoogleCalendarPerson:
    mapping = _provider_dict(value)
    _reject_unknown(mapping, _PERSON_ALLOWED_KEYS)
    return GoogleCalendarPerson(
        id=_optional_provider(mapping, "id", lambda item: _validate_text(item)),
        email=_optional_provider(mapping, "email", lambda item: _validate_text(item)),
        display_name=_optional_provider(mapping, "displayName", lambda item: _validate_text(item)),
        self=_optional_provider(mapping, "self", _validate_exact_bool),
    )


def _parse_attendee(value: object) -> GoogleCalendarAttendee:
    mapping = _provider_dict(value)
    _reject_unknown(mapping, _ATTENDEE_ALLOWED_KEYS)
    return GoogleCalendarAttendee(
        id=_optional_provider(mapping, "id", lambda item: _validate_text(item)),
        email=_optional_provider(mapping, "email", lambda item: _validate_text(item)),
        display_name=_optional_provider(mapping, "displayName", lambda item: _validate_text(item)),
        organizer=_optional_provider(mapping, "organizer", _validate_exact_bool),
        self=_optional_provider(mapping, "self", _validate_exact_bool),
        resource=_optional_provider(mapping, "resource", _validate_exact_bool),
        optional=_optional_provider(mapping, "optional", _validate_exact_bool),
        response_status=_parse_enum(
            mapping, "responseStatus", GoogleCalendarAttendeeResponseStatus
        ),
        comment=_optional_provider(mapping, "comment", lambda item: _validate_text(item)),
        additional_guests=_optional_provider(
            mapping, "additionalGuests", lambda item: _validate_exact_int(item)
        ),
    )


def _parse_event_date_time(value: object) -> GoogleCalendarEventDateTime:
    mapping = _provider_dict(value)
    _reject_unknown(mapping, _DATE_TIME_ALLOWED_KEYS)
    return GoogleCalendarEventDateTime(
        date_time=_optional_provider(mapping, "dateTime", _validate_rfc3339),
        date=_optional_provider(mapping, "date", _validate_date),
        time_zone=_optional_provider(mapping, "timeZone", lambda item: _validate_text(item, max_length=256)),
    )


def _parse_reminder(value: object) -> GoogleCalendarReminder:
    mapping = _provider_dict(value)
    _reject_unknown(mapping, _REMINDER_ALLOWED_KEYS)
    if "method" not in mapping or "minutes" not in mapping:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if mapping["method"] is None or mapping["minutes"] is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return GoogleCalendarReminder(
        method=_validate_enum(mapping["method"], GoogleCalendarReminderMethod),
        minutes=_validate_exact_int(mapping["minutes"], maximum=40_320),
    )


def _parse_reminders(value: object) -> GoogleCalendarReminders:
    mapping = _provider_dict(value)
    _reject_unknown(mapping, _REMINDERS_ALLOWED_KEYS)
    if "useDefault" not in mapping or mapping["useDefault"] is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    overrides_value = mapping.get("overrides", [])
    if overrides_value is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    raw_overrides = _provider_list(overrides_value)
    if len(raw_overrides) > _MAX_REMINDER_OVERRIDES:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return GoogleCalendarReminders(
        use_default=_validate_exact_bool(mapping["useDefault"]),
        overrides=tuple(_parse_reminder(item) for item in raw_overrides),
    )


def _parse_conference_entry_point(value: object) -> GoogleCalendarConferenceEntryPoint:
    mapping = _provider_dict(value)
    _reject_unknown(mapping, _ENTRY_POINT_ALLOWED_KEYS)
    if "entryPointType" not in mapping or "uri" not in mapping:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return GoogleCalendarConferenceEntryPoint(
        entry_point_type=_validate_nonblank(
            mapping["entryPointType"],
            max_length=_MAX_TEXT_LENGTH,
            message=_UNEXPECTED_RESPONSE_MESSAGE,
        ),
        uri=_validate_nonblank(
            mapping["uri"], max_length=_MAX_TEXT_LENGTH, message=_UNEXPECTED_RESPONSE_MESSAGE
        ),
        label=_optional_provider(mapping, "label", _validate_text),
        pin=_optional_provider(mapping, "pin", _validate_text),
        access_code=_optional_provider(mapping, "accessCode", _validate_text),
        meeting_code=_optional_provider(mapping, "meetingCode", _validate_text),
        passcode=_optional_provider(mapping, "passcode", _validate_text),
        password=_optional_provider(mapping, "password", _validate_text),
    )


def _parse_conference(value: object) -> GoogleCalendarConferenceData:
    mapping = _provider_dict(value)
    _reject_unknown(mapping, _CONFERENCE_ALLOWED_KEYS)
    raw_entry_points = mapping.get("entryPoints", [])
    if raw_entry_points is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    entry_points = _provider_list(raw_entry_points)
    if len(entry_points) > _MAX_CONFERENCE_ENTRY_POINTS:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    solution = None
    if "conferenceSolution" in mapping:
        raw_solution = mapping["conferenceSolution"]
        if raw_solution is None:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        solution_mapping = _provider_dict(raw_solution)
        _reject_unknown(solution_mapping, _CONFERENCE_SOLUTION_ALLOWED_KEYS)
        if "key" not in solution_mapping or "name" not in solution_mapping:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        key_mapping = _provider_dict(solution_mapping["key"])
        _reject_unknown(key_mapping, _CONFERENCE_KEY_ALLOWED_KEYS)
        if "type" not in key_mapping or key_mapping["type"] is None:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        solution = GoogleCalendarConferenceSolution(
            type=_validate_enum(key_mapping["type"], GoogleCalendarConferenceSolutionType),
            name=_validate_nonblank(
                solution_mapping["name"],
                max_length=_MAX_TEXT_LENGTH,
                message=_UNEXPECTED_RESPONSE_MESSAGE,
            ),
            icon_uri=_optional_provider(solution_mapping, "iconUri", _validate_text),
        )
    return GoogleCalendarConferenceData(
        conference_id=_optional_provider(mapping, "conferenceId", _validate_text),
        signature=_optional_provider(mapping, "signature", _validate_text),
        notes=_optional_provider(mapping, "notes", _validate_text),
        entry_points=tuple(_parse_conference_entry_point(item) for item in entry_points),
        conference_solution=solution,
    )


def _parse_recurrence(value: object) -> tuple[str, ...]:
    raw_lines = _provider_list(value)
    if len(raw_lines) > _MAX_RECURRENCE_LINES:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    lines: list[str] = []
    for line in raw_lines:
        if type(line) is not str or not line or len(line) > _MAX_RECURRENCE_LINE_LENGTH:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if _ASCII_CONTROL.search(line):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        lines.append(line)
    return tuple(lines)


def _parse_event(value: object) -> GoogleCalendarEvent:
    mapping = _provider_dict(value)
    _reject_unknown(mapping, _EVENT_ALLOWED_KEYS)
    if "id" not in mapping or "status" not in mapping:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if mapping["id"] is None or mapping["status"] is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    status = _validate_enum(mapping["status"], GoogleCalendarEventStatus)
    attendees_value = mapping.get("attendees", [])
    if attendees_value is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    attendees = _provider_list(attendees_value)
    if len(attendees) > _MAX_ATTENDEES:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    recurrence = _optional_provider(mapping, "recurrence", _parse_recurrence, ())
    return GoogleCalendarEvent(
        id=_validate_nonblank(
            mapping["id"], max_length=_MAX_TEXT_LENGTH, message=_UNEXPECTED_RESPONSE_MESSAGE
        ),
        i_cal_uid=_optional_provider(mapping, "iCalUID", _validate_text),
        etag=_optional_provider(mapping, "etag", _validate_text),
        status=status,
        event_type=_parse_enum(mapping, "eventType", GoogleCalendarEventType),
        summary=_optional_provider(mapping, "summary", _validate_text),
        description=_optional_provider(
            mapping, "description", lambda item: _validate_text(item, max_length=_MAX_DESCRIPTION_LENGTH)
        ),
        location=_optional_provider(mapping, "location", _validate_text),
        html_link=_optional_provider(mapping, "htmlLink", _validate_text),
        created=_optional_provider(mapping, "created", _validate_rfc3339),
        updated=_optional_provider(mapping, "updated", _validate_rfc3339),
        color_id=_optional_provider(mapping, "colorId", _validate_text),
        visibility=_parse_enum(mapping, "visibility", GoogleCalendarVisibility),
        transparency=_parse_enum(mapping, "transparency", GoogleCalendarTransparency),
        sequence=_optional_provider(mapping, "sequence", _validate_exact_int),
        start=_optional_provider(mapping, "start", _parse_event_date_time),
        end=_optional_provider(mapping, "end", _parse_event_date_time),
        end_time_unspecified=_optional_provider(mapping, "endTimeUnspecified", _validate_exact_bool),
        recurrence=recurrence,
        recurring_event_id=_optional_provider(mapping, "recurringEventId", _validate_text),
        original_start_time=_optional_provider(
            mapping, "originalStartTime", _parse_event_date_time
        ),
        creator=_optional_provider(mapping, "creator", _parse_person),
        organizer=_optional_provider(mapping, "organizer", _parse_person),
        attendees=tuple(_parse_attendee(item) for item in attendees),
        attendees_omitted=_optional_provider(mapping, "attendeesOmitted", _validate_exact_bool),
        guests_can_invite_others=_optional_provider(
            mapping, "guestsCanInviteOthers", _validate_exact_bool
        ),
        guests_can_modify=_optional_provider(mapping, "guestsCanModify", _validate_exact_bool),
        guests_can_see_other_guests=_optional_provider(
            mapping, "guestsCanSeeOtherGuests", _validate_exact_bool
        ),
        private_copy=_optional_provider(mapping, "privateCopy", _validate_exact_bool),
        locked=_optional_provider(mapping, "locked", _validate_exact_bool),
        hangout_link=_optional_provider(mapping, "hangoutLink", _validate_text),
        conference_data=_optional_provider(mapping, "conferenceData", _parse_conference),
        reminders=_optional_provider(mapping, "reminders", _parse_reminders),
    )


def _parse_sync_token_value(value: object) -> GoogleCalendarSyncToken:
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    try:
        return GoogleCalendarSyncToken(value=value)
    except Exception:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE) from None


def _parse_page_token_value(value: object) -> GoogleWorkspacePageToken:
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if (
        not value
        or value != value.strip()
        or len(value) > _MAX_PAGE_TOKEN_LENGTH
        or _ASCII_CONTROL.search(value)
    ):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return GoogleWorkspacePageToken(value=value)


def _parse_event_page(
    payload: object,
    *,
    calendar_id: str,
    max_results: int,
) -> GoogleCalendarEventPage:
    mapping = _provider_dict(payload)
    _reject_unknown(mapping, _PAGE_ALLOWED_KEYS)
    if _retained_text_size(mapping) > _MAX_TOTAL_TEXT_CHARS:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if "items" not in mapping or mapping["items"] is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    raw_items = _provider_list(mapping["items"])
    if len(raw_items) > max_results or len(raw_items) > _MAX_EVENTS:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    has_page = "nextPageToken" in mapping
    has_sync = "nextSyncToken" in mapping
    if has_page == has_sync:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    page_token = (
        _parse_page_token_value(mapping["nextPageToken"]) if has_page else None
    )
    sync_token = _parse_sync_token_value(mapping["nextSyncToken"]) if has_sync else None
    return GoogleCalendarEventPage(
        calendar_id=calendar_id,
        summary=_optional_provider(mapping, "summary", _validate_text),
        description=_optional_provider(
            mapping, "description", lambda item: _validate_text(item, max_length=_MAX_DESCRIPTION_LENGTH)
        ),
        updated=_optional_provider(mapping, "updated", _validate_rfc3339),
        time_zone=_optional_provider(mapping, "timeZone", lambda item: _validate_text(item, max_length=256)),
        access_role=_parse_enum(mapping, "accessRole", GoogleCalendarAccessRole),
        events=tuple(_parse_event(item) for item in raw_items),
        next_page_token=page_token,
        next_sync_token=sync_token,
    )


class GoogleCalendarKnowledgeReader:
    """Stateless Google Calendar knowledge reader using one shared transport."""

    def __init__(self, *, transport: GoogleWorkspaceTransport) -> None:
        if not isinstance(transport, GoogleWorkspaceTransport):
            raise IntegrationConfigurationError(_UNEXPECTED_RESPONSE_MESSAGE)
        self._transport = transport

    def list_events_page(
        self,
        *,
        calendar_id: str,
        page_token: GoogleWorkspacePageToken | None = None,
        sync_token: GoogleCalendarSyncToken | None = None,
        max_results: int = 250,
    ) -> GoogleCalendarEventPage:
        try:
            validated_calendar_id = _validate_calendar_id_for_request(calendar_id)
            validated_page_token = _validate_page_token_for_request(page_token)
            validated_sync_token = _validate_sync_token_for_request(sync_token)
            if type(max_results) is not int or not 1 <= max_results <= _MAX_EVENT_PAGE_SIZE:
                raise ValueError(_INVALID_PAGE_LIMIT_MESSAGE)
        except ValueError as exc:
            message = str(exc)
            if message == _INVALID_CALENDAR_ID_MESSAGE:
                raise IntegrationConfigurationError(message) from None
            if message == _INVALID_PAGE_TOKEN_MESSAGE:
                raise IntegrationConfigurationError(message) from None
            if message == _INVALID_PAGE_LIMIT_MESSAGE:
                raise IntegrationConfigurationError(message) from None
            raise IntegrationConfigurationError(_INVALID_SYNC_TOKEN_MESSAGE) from None

        encoded_id = quote(validated_calendar_id, safe="")
        params: dict[str, object] = {
            "maxResults": max_results,
            "showDeleted": True,
            "singleEvents": False,
            "fields": _GOOGLE_CALENDAR_EVENTS_FIELDS,
        }
        if validated_page_token is not None:
            params["pageToken"] = validated_page_token.value
        if validated_sync_token is not None:
            params["syncToken"] = validated_sync_token.value
        try:
            payload = self._transport.get_json(
                source_kind=GoogleWorkspaceSourceKind.CALENDAR,
                relative_path=f"/calendars/{encoded_id}/events",
                params=params,
            )
        except GoogleWorkspaceApiError:
            raise
        except Exception:
            raise IntegrationDependencyError(_REQUEST_FAILED_MESSAGE) from None
        try:
            return _parse_event_page(
                payload,
                calendar_id=validated_calendar_id,
                max_results=max_results,
            )
        except Exception:
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE) from None

    def read_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
    ) -> GoogleCalendarEvent:
        try:
            validated_calendar_id = _validate_calendar_id_for_request(calendar_id)
            validated_event_id = _validate_event_id_for_request(event_id)
        except ValueError as exc:
            raise IntegrationConfigurationError(str(exc)) from None

        encoded_calendar_id = quote(validated_calendar_id, safe="")
        encoded_event_id = quote(validated_event_id, safe="")
        try:
            payload = self._transport.get_json(
                source_kind=GoogleWorkspaceSourceKind.CALENDAR,
                relative_path=(
                    f"/calendars/{encoded_calendar_id}/events/{encoded_event_id}"
                ),
                params={"fields": _GOOGLE_CALENDAR_EVENT_FIELDS},
            )
        except GoogleWorkspaceApiError:
            raise
        except Exception:
            raise IntegrationDependencyError(_REQUEST_FAILED_MESSAGE) from None
        try:
            return _parse_event(payload)
        except Exception:
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE) from None
