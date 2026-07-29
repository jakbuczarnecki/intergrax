# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph Calendar knowledge-read event content surface."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers.collaboration_suite.ms365_graph.adapter import (
    _Ms365GraphCollaborationSuite,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.client import GraphRestClient
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    DEFAULT_GRAPH_BASE_URL,
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    ABSOLUTE_CALENDAR_EVENT_CONTENT_MAX_CHARS,
    DEFAULT_CALENDAR_EVENT_CONTENT_MAX_CHARS,
    MsGraphCalendarAttendeeType,
    MsGraphCalendarBodyKind,
    MsGraphCalendarContentReadClient,
    MsGraphCalendarContentReader,
    MsGraphCalendarEventChange,
    MsGraphCalendarEventChangeKind,
    MsGraphCalendarEventChanged,
    MsGraphCalendarEventContent,
    MsGraphCalendarEventContentTooLarge,
    MsGraphCalendarEventType,
    MsGraphCalendarImportance,
    MsGraphCalendarLocationType,
    MsGraphCalendarOnlineMeetingProvider,
    MsGraphCalendarRecurrencePatternType,
    MsGraphCalendarRecurrenceRangeType,
    MsGraphCalendarResponseStatus,
    MsGraphCalendarResponseType,
    MsGraphCalendarSensitivity,
    MsGraphCalendarShowAs,
    MsGraphKnowledgeTransport,
    parse_msgraph_calendar_attendee,
    parse_msgraph_calendar_event_content,
    parse_msgraph_calendar_location,
    parse_msgraph_calendar_participant,
    parse_msgraph_calendar_recurrence,
    validate_msgraph_calendar_attendee,
    validate_msgraph_calendar_event_content,
    validate_msgraph_calendar_location,
    validate_msgraph_calendar_participant,
    validate_msgraph_calendar_recurrence,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_MAILBOX_USER_ID = "user@contoso.com"
_OTHER_MAILBOX_USER_ID = "other@contoso.com"
_CALENDAR_ID = "AAMkAGI2THSAAA-calendar-id"
_OTHER_CALENDAR_ID = "other-calendar-id"
_EVENT_ID = "AAMkAGI2THSAAA-event-id"
_OTHER_EVENT_ID = "AAMkAGI2THSBBB"
_CHANGE_KEY = "change-key-secret-value"
_OTHER_CHANGE_KEY = "other-change-key"
_I_CAL_UID = "00000000-0000-0000-0000-000000000001@contoso.com"
_SUBJECT = "Quarterly sync"
_SECRET_SUBJECT = "secret-subject-value"
_SECRET_BODY = "secret-body-text-value"
_SECRET_ADDRESS = "secret-participant@example.com"
_SECRET_DISPLAY_NAME = "Secret Participant Name"
_SECRET_LOCATION = "Secret Conference Room"
_START_AT = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
_END_AT = datetime(2024, 6, 1, 11, 0, tzinfo=timezone.utc)
_CREATED_AT = datetime(2024, 5, 31, 9, 0, tzinfo=timezone.utc)
_LAST_MODIFIED_AT = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
_QUOTED_MAILBOX = quote(_MAILBOX_USER_ID, safe="")
_QUOTED_CALENDAR = quote(_CALENDAR_ID, safe="")
_QUOTED_EVENT_ID = quote(_EVENT_ID, safe="")
_CONTENT_PATH = f"/users/{_QUOTED_MAILBOX}/calendars/{_QUOTED_CALENDAR}/events/{_QUOTED_EVENT_ID}"
_SELECT = (
    "id,changeKey,type,subject,body,bodyPreview,start,end,originalStart,"
    "originalStartTimeZone,originalEndTimeZone,createdDateTime,lastModifiedDateTime,"
    "organizer,attendees,location,locations,recurrence,seriesMasterId,"
    "cancelledOccurrences,categories,iCalUId,importance,sensitivity,showAs,"
    "responseStatus,isAllDay,isCancelled,isDraft,isOrganizer,isOnlineMeeting,"
    "onlineMeetingProvider,hasAttachments,hideAttendees,allowNewTimeProposals,"
    "responseRequested,isReminderOn,reminderMinutesBeforeStart"
)
_TEXT_BODY_HEADERS = {
    "Prefer": (
        'IdType="ImmutableId", '
        'outlook.timezone="UTC", '
        'outlook.body-content-type="text"'
    ),
}
_SAFE_ERROR = "unexpected Microsoft Graph Calendar event content response"
_KNOWLEDGE_SAFE_ERROR = "unexpected Microsoft Graph knowledge response"
_EVENTS_SAFE_ERROR = "unexpected Microsoft Graph Calendar events response"
_REQUEST_ERROR = "invalid Microsoft Graph Calendar content request"
_CHANGED_ERROR = "Microsoft Graph Calendar event changed during read"
_TOO_LARGE_ERROR = "Microsoft Graph Calendar event exceeds the configured content limit"
_CAPABILITY_ERROR = (
    "Microsoft Graph integration does not expose Calendar content capability"
)


def _config() -> Ms365GraphIntegrationConfig:
    return Ms365GraphIntegrationConfig(
        tenant_id="tenant-123",
        client_id="client-456",
        client_secret="secret",
        graph_base_url=_GRAPH_BASE,
    )


def _json_response(*, status_code: int = 200, payload: object | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload if payload is not None else {}
    response.raise_for_status = MagicMock()
    return response


def _utc_dt(value: datetime) -> dict[str, str]:
    return {"dateTime": value.strftime("%Y-%m-%dT%H:%M:%S"), "timeZone": "UTC"}


def _text_body(content: str) -> dict[str, str]:
    return {"contentType": "text", "content": content}


def _html_body(content: str) -> dict[str, str]:
    return {"contentType": "html", "content": content}


def _participant(
    address: str,
    *,
    display_name: str | None = None,
    include_name_key: bool = True,
) -> dict[str, Any]:
    email_address: dict[str, Any] = {"address": address}
    if include_name_key:
        if display_name is not None:
            email_address["name"] = display_name
    return {"emailAddress": email_address}


def _attendee_payload(
    address: str,
    *,
    display_name: str | None = None,
    attendee_type: str = "required",
    response: str = "accepted",
) -> dict[str, Any]:
    participant = _participant(address, display_name=display_name)
    return {
        **participant,
        "type": attendee_type,
        "status": {"response": response},
    }


def _location_payload(
    display_name: str = "Room A",
    *,
    location_type: str = "default",
    with_address: bool = False,
) -> dict[str, Any]:
    location: dict[str, Any] = {
        "displayName": display_name,
        "locationType": location_type,
    }
    if with_address:
        location["address"] = {
            "street": "1 Main St",
            "city": "Seattle",
            "state": "WA",
            "countryOrRegion": "US",
            "postalCode": "98101",
        }
    return location


def _recurrence_payload() -> dict[str, Any]:
    return {
        "pattern": {
            "type": "weekly",
            "interval": 1,
            "daysOfWeek": ["monday"],
            "firstDayOfWeek": "sunday",
        },
        "range": {
            "type": "noEnd",
            "startDate": "2024-06-01",
        },
    }


def _content_payload(
    *,
    event_id: str = _EVENT_ID,
    change_key: str = _CHANGE_KEY,
    subject: str | None = None,
    include_subject_key: bool = True,
    subject_null: bool = False,
    body_text: str = "Hello, world.",
    body_kind: str = "text",
    include_body: bool = True,
    body_preview: str | None = "Hello, world.",
    include_body_preview_key: bool = True,
    organizer: dict[str, Any] | None = None,
    include_organizer_key: bool = True,
    attendees: list[dict[str, Any]] | None = None,
    location: dict[str, Any] | None = None,
    include_location_key: bool = True,
    locations: list[dict[str, Any]] | None = None,
    recurrence: dict[str, Any] | None = None,
    include_recurrence_key: bool = True,
    i_cal_uid: str | None = None,
    include_i_cal_uid_key: bool = False,
    removed: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": event_id,
        "changeKey": change_key,
        "type": "singleInstance",
        "start": _utc_dt(_START_AT),
        "end": _utc_dt(_END_AT),
        "lastModifiedDateTime": _LAST_MODIFIED_AT.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "isAllDay": False,
        "isCancelled": False,
        "isDraft": False,
        "hasAttachments": False,
        "isOnlineMeeting": False,
        "originalStartTimeZone": "UTC",
        "originalEndTimeZone": "UTC",
        "createdDateTime": _CREATED_AT.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "attendees": [] if attendees is None else attendees,
        "locations": [] if locations is None else locations,
        "cancelledOccurrences": [],
        "categories": [],
        "importance": "normal",
        "sensitivity": "normal",
        "showAs": "busy",
        "responseStatus": {"response": "organizer"},
        "isOrganizer": True,
        "onlineMeetingProvider": "unknown",
        "hideAttendees": False,
        "allowNewTimeProposals": True,
        "responseRequested": True,
        "isReminderOn": True,
        "reminderMinutesBeforeStart": 15,
    }
    if include_subject_key:
        payload["subject"] = None if subject_null else (subject if subject is not None else "")
    elif subject is not None:
        payload["subject"] = subject
    if include_body:
        payload["body"] = _text_body(body_text) if body_kind == "text" else _html_body(body_text)
    if include_body_preview_key:
        payload["bodyPreview"] = body_preview
    if include_organizer_key:
        payload["organizer"] = organizer
    if include_location_key:
        payload["location"] = location
    if include_recurrence_key:
        payload["recurrence"] = recurrence
    if include_i_cal_uid_key:
        payload["iCalUId"] = i_cal_uid
    if removed:
        payload["@removed"] = {"reason": "deleted"}
    return payload


def _valid_active_change_kwargs(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "calendar_remote_id": _CALENDAR_ID,
        "remote_id": _EVENT_ID,
        "kind": MsGraphCalendarEventChangeKind.ACTIVE,
        "change_key": _CHANGE_KEY,
        "event_type": MsGraphCalendarEventType.SINGLE_INSTANCE,
        "start_at": _START_AT,
        "end_at": _END_AT,
        "last_modified_at": _LAST_MODIFIED_AT,
        "is_all_day": False,
        "is_cancelled": False,
        "is_draft": False,
        "has_attachments": False,
        "is_online_meeting": False,
    }
    defaults.update(overrides)
    return defaults


def _valid_active_change(**overrides: object) -> MsGraphCalendarEventChange:
    return MsGraphCalendarEventChange(**_valid_active_change_kwargs(**overrides))


def _valid_removed_change(**overrides: object) -> MsGraphCalendarEventChange:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "calendar_remote_id": _CALENDAR_ID,
        "remote_id": _EVENT_ID,
        "kind": MsGraphCalendarEventChangeKind.REMOVED,
        "removed_reason": "deleted",
    }
    defaults.update(overrides)
    return MsGraphCalendarEventChange(**defaults)


def _default_response_status() -> MsGraphCalendarResponseStatus:
    return MsGraphCalendarResponseStatus(response=MsGraphCalendarResponseType.ORGANIZER)


def _valid_event_content(**overrides: object) -> MsGraphCalendarEventContent:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "calendar_remote_id": _CALENDAR_ID,
        "remote_id": _EVENT_ID,
        "content_revision": _CHANGE_KEY,
        "event_type": MsGraphCalendarEventType.SINGLE_INSTANCE,
        "body_kind": MsGraphCalendarBodyKind.TEXT,
        "body_content": "Hello, world.",
        "start_at": _START_AT,
        "end_at": _END_AT,
        "created_at": _CREATED_AT,
        "last_modified_at": _LAST_MODIFIED_AT,
        "importance": MsGraphCalendarImportance.NORMAL,
        "sensitivity": MsGraphCalendarSensitivity.NORMAL,
        "show_as": MsGraphCalendarShowAs.BUSY,
        "response_status": _default_response_status(),
        "is_all_day": False,
        "is_cancelled": False,
        "is_draft": False,
        "is_organizer": True,
        "is_online_meeting": False,
        "has_attachments": False,
        "hide_attendees": False,
        "allow_new_time_proposals": True,
        "response_requested": True,
        "is_reminder_on": True,
        "reminder_minutes_before_start": 15,
        "online_meeting_provider": MsGraphCalendarOnlineMeetingProvider.UNKNOWN,
    }
    defaults.update(overrides)
    return MsGraphCalendarEventContent(**defaults)


def _reader(http: MagicMock) -> MsGraphCalendarContentReader:
    return MsGraphCalendarContentReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
    )


def _graph_client(http: MagicMock) -> GraphRestClient:
    return GraphRestClient(_config(), http_client=http)


def _setup_happy_path(
    *,
    payload: dict[str, Any] | None = None,
) -> tuple[MagicMock, MsGraphCalendarContentReader]:
    http = MagicMock()
    http.get.return_value = _json_response(payload=payload or _content_payload())
    return http, _reader(http)


def _require_calendar_content_client(client: object) -> MsGraphCalendarContentReadClient:
    if not isinstance(client, MsGraphCalendarContentReadClient):
        raise IntegrationConfigurationError(_CAPABILITY_ERROR)
    return client


def _consume_calendar_event_content(
    client: MsGraphCalendarContentReadClient,
    *,
    event: MsGraphCalendarEventChange,
    max_chars: int,
) -> MsGraphCalendarEventContent:
    result = client.read_calendar_event_content(event=event, max_chars=max_chars)
    return validate_msgraph_calendar_event_content(
        result,
        event=event,
        max_chars=max_chars,
    )


# --- constants ---


def test_default_max_chars_constants() -> None:
    assert DEFAULT_CALENDAR_EVENT_CONTENT_MAX_CHARS == 2_000_000
    assert ABSOLUTE_CALENDAR_EVENT_CONTENT_MAX_CHARS == 8_000_000


# --- participant parser ---


def test_parse_participant_with_display_name() -> None:
    participant = parse_msgraph_calendar_participant(
        _participant(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
    )
    assert participant.address == _SECRET_ADDRESS
    assert participant.display_name == _SECRET_DISPLAY_NAME


def test_parse_participant_without_name_key() -> None:
    participant = parse_msgraph_calendar_participant(
        _participant(_SECRET_ADDRESS, include_name_key=False)
    )
    assert participant.display_name is None


def test_parse_participant_empty_name_becomes_none() -> None:
    participant = parse_msgraph_calendar_participant(
        _participant(_SECRET_ADDRESS, display_name="   ")
    )
    assert participant.display_name is None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"emailAddress": None},
        {"emailAddress": {}},
        {"emailAddress": {"name": "Only Name"}},
        {"emailAddress": {"address": ""}},
        {"emailAddress": {"address": "a@b.com", "name": 123}},
        {"emailAddress": {"address": "a\x00@b.com"}},
    ],
)
def test_parse_participant_malformed(payload: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        parse_msgraph_calendar_participant(payload)
    assert exc.value.__cause__ is None
    assert _SECRET_ADDRESS not in str(exc.value)


def test_validate_participant_returns_new_instance() -> None:
    original = parse_msgraph_calendar_participant(
        _participant(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
    )
    validated = validate_msgraph_calendar_participant(original)
    assert validated == original
    assert validated is not original


# --- success: text/html body ---


def test_read_text_body() -> None:
    http, reader = _setup_happy_path(
        payload=_content_payload(body_text=_SECRET_BODY),
    )
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert result.body_kind is MsGraphCalendarBodyKind.TEXT
    assert result.body_content == _SECRET_BODY
    http.get.assert_called_once()


def test_read_html_body() -> None:
    html = "<p>Hello, <b>world</b>.</p>"
    _, reader = _setup_happy_path(
        payload=_content_payload(body_text=html, body_kind="html"),
    )
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert result.body_kind is MsGraphCalendarBodyKind.HTML
    assert result.body_content == html


def test_read_empty_body() -> None:
    _, reader = _setup_happy_path(payload=_content_payload(body_text=""))
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert result.body_content == ""


def test_read_body_preview_present() -> None:
    _, reader = _setup_happy_path(
        payload=_content_payload(body_preview="Preview text"),
    )
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert result.body_preview == "Preview text"


def test_read_body_preview_null() -> None:
    _, reader = _setup_happy_path(
        payload=_content_payload(body_preview=None),
    )
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert result.body_preview is None


# --- success: organizer/attendees/locations/recurrence ---


def test_read_organizer_present() -> None:
    organizer = _participant(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
    _, reader = _setup_happy_path(
        payload=_content_payload(organizer=organizer),
    )
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert result.organizer is not None
    assert result.organizer.address == _SECRET_ADDRESS
    assert result.organizer.display_name == _SECRET_DISPLAY_NAME


def test_read_organizer_null() -> None:
    _, reader = _setup_happy_path(payload=_content_payload(organizer=None))
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert result.organizer is None


def test_read_attendees() -> None:
    attendee = _attendee_payload(
        "attendee@contoso.com",
        display_name="Attendee",
        attendee_type="optional",
        response="tentativelyAccepted",
    )
    _, reader = _setup_happy_path(payload=_content_payload(attendees=[attendee]))
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert len(result.attendees) == 1
    assert result.attendees[0].participant.address == "attendee@contoso.com"
    assert result.attendees[0].attendee_type is MsGraphCalendarAttendeeType.OPTIONAL
    assert result.attendees[0].status.response is MsGraphCalendarResponseType.TENTATIVELY_ACCEPTED


def test_read_empty_attendees() -> None:
    _, reader = _setup_happy_path(payload=_content_payload())
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert result.attendees == ()


def test_read_location_and_locations() -> None:
    primary = _location_payload(_SECRET_LOCATION, with_address=True)
    secondary = _location_payload("Lobby", location_type="conferenceRoom")
    _, reader = _setup_happy_path(
        payload=_content_payload(location=primary, locations=[secondary]),
    )
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert result.location is not None
    assert result.location.display_name == _SECRET_LOCATION
    assert result.location.location_type is MsGraphCalendarLocationType.DEFAULT
    assert result.location.city == "Seattle"
    assert len(result.locations) == 1
    assert result.locations[0].location_type is MsGraphCalendarLocationType.CONFERENCE_ROOM


def test_read_null_location() -> None:
    _, reader = _setup_happy_path(payload=_content_payload(location=None))
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert result.location is None
    assert result.locations == ()


def test_read_recurrence() -> None:
    _, reader = _setup_happy_path(
        payload=_content_payload(recurrence=_recurrence_payload()),
    )
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert result.recurrence is not None
    assert result.recurrence.pattern.pattern_type is MsGraphCalendarRecurrencePatternType.WEEKLY
    assert result.recurrence.pattern.interval == 1
    assert result.recurrence.range.range_type is MsGraphCalendarRecurrenceRangeType.NO_END
    assert result.recurrence.range.start_date == date(2024, 6, 1)


def test_read_null_recurrence() -> None:
    _, reader = _setup_happy_path(payload=_content_payload(recurrence=None))
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert result.recurrence is None


def test_parse_attendee_and_location_and_recurrence_round_trip() -> None:
    attendee = parse_msgraph_calendar_attendee(
        _attendee_payload(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
    )
    validated_attendee = validate_msgraph_calendar_attendee(attendee)
    assert validated_attendee.participant.address == _SECRET_ADDRESS

    location = parse_msgraph_calendar_location(
        _location_payload(_SECRET_LOCATION, with_address=True)
    )
    validated_location = validate_msgraph_calendar_location(location)
    assert validated_location.display_name == _SECRET_LOCATION

    recurrence = parse_msgraph_calendar_recurrence(_recurrence_payload())
    validated_recurrence = validate_msgraph_calendar_recurrence(recurrence)
    assert validated_recurrence.pattern.pattern_type is MsGraphCalendarRecurrencePatternType.WEEKLY


# --- success: subject variants ---


def test_read_subject_present() -> None:
    _, reader = _setup_happy_path(
        payload=_content_payload(
            subject=_SECRET_SUBJECT,
            include_i_cal_uid_key=True,
            i_cal_uid=_I_CAL_UID,
        ),
    )
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert result.subject == _SECRET_SUBJECT
    assert result.i_cal_uid == _I_CAL_UID


def test_read_empty_subject() -> None:
    _, reader = _setup_happy_path(payload=_content_payload(subject=""))
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert result.subject == ""


def test_read_null_subject() -> None:
    _, reader = _setup_happy_path(
        payload=_content_payload(include_subject_key=True, subject_null=True),
    )
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert result.subject is None


# --- success: body at limits ---


def test_read_body_16001_chars_within_max_chars_succeeds() -> None:
    body = "x" * 16_001
    _, reader = _setup_happy_path(payload=_content_payload(body_text=body))
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=20_000)
    assert len(result.body_content) == 16_001


def test_read_body_at_max_chars_succeeds() -> None:
    body = "x" * 100_000
    _, reader = _setup_happy_path(payload=_content_payload(body_text=body))
    result = reader.read_calendar_event_content(event=_valid_active_change(), max_chars=100_000)
    assert len(result.body_content) == 100_000


def test_read_body_one_over_max_chars_rejected() -> None:
    body = "x" * 100_001
    http = MagicMock()
    http.get.return_value = _json_response(payload=_content_payload(body_text=body))
    with pytest.raises(MsGraphCalendarEventContentTooLarge, match=_TOO_LARGE_ERROR) as exc:
        _reader(http).read_calendar_event_content(event=_valid_active_change(), max_chars=100_000)
    assert exc.value.__cause__ is None


# --- cancelledOccurrences event-type rules ---


def test_single_instance_without_cancelled_occurrences_succeeds() -> None:
    payload = _content_payload()
    del payload["cancelledOccurrences"]
    content = parse_msgraph_calendar_event_content(
        payload,
        expected_mailbox_user_id=_MAILBOX_USER_ID,
        expected_calendar_id=_CALENDAR_ID,
        max_chars=10_000,
    )
    assert content.cancelled_occurrence_ids == ()


def test_occurrence_without_cancelled_occurrences_succeeds() -> None:
    payload = _content_payload()
    payload["type"] = "occurrence"
    payload["originalStart"] = "2024-06-01T09:00:00Z"
    payload["seriesMasterId"] = "series-master-id"
    del payload["cancelledOccurrences"]
    content = parse_msgraph_calendar_event_content(
        payload,
        expected_mailbox_user_id=_MAILBOX_USER_ID,
        expected_calendar_id=_CALENDAR_ID,
        max_chars=10_000,
    )
    assert content.cancelled_occurrence_ids == ()


def test_exception_without_cancelled_occurrences_succeeds() -> None:
    payload = _content_payload()
    payload["type"] = "exception"
    payload["originalStart"] = "2024-06-01T09:00:00Z"
    payload["seriesMasterId"] = "series-master-id"
    del payload["cancelledOccurrences"]
    content = parse_msgraph_calendar_event_content(
        payload,
        expected_mailbox_user_id=_MAILBOX_USER_ID,
        expected_calendar_id=_CALENDAR_ID,
        max_chars=10_000,
    )
    assert content.cancelled_occurrence_ids == ()


def test_series_master_with_empty_cancelled_occurrences_succeeds() -> None:
    payload = _content_payload()
    payload["type"] = "seriesMaster"
    payload["cancelledOccurrences"] = []
    content = parse_msgraph_calendar_event_content(
        payload,
        expected_mailbox_user_id=_MAILBOX_USER_ID,
        expected_calendar_id=_CALENDAR_ID,
        max_chars=10_000,
    )
    assert content.cancelled_occurrence_ids == ()


def test_series_master_with_cancelled_ids_succeeds() -> None:
    payload = _content_payload()
    payload["type"] = "seriesMaster"
    payload["cancelledOccurrences"] = ["occ-id-1", "occ-id-2"]
    content = parse_msgraph_calendar_event_content(
        payload,
        expected_mailbox_user_id=_MAILBOX_USER_ID,
        expected_calendar_id=_CALENDAR_ID,
        max_chars=10_000,
    )
    assert content.cancelled_occurrence_ids == ("occ-id-1", "occ-id-2")


def test_series_master_missing_cancelled_occurrences_rejected() -> None:
    payload = _content_payload()
    payload["type"] = "seriesMaster"
    del payload["cancelledOccurrences"]
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        parse_msgraph_calendar_event_content(
            payload,
            expected_mailbox_user_id=_MAILBOX_USER_ID,
            expected_calendar_id=_CALENDAR_ID,
            max_chars=10_000,
        )


def test_present_null_cancelled_occurrences_rejected() -> None:
    payload = _content_payload() | {"cancelledOccurrences": None}
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        parse_msgraph_calendar_event_content(
            payload,
            expected_mailbox_user_id=_MAILBOX_USER_ID,
            expected_calendar_id=_CALENDAR_ID,
            max_chars=10_000,
        )


def test_present_malformed_cancelled_occurrences_collection_rejected() -> None:
    payload = _content_payload() | {"cancelledOccurrences": "not-a-list"}
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        parse_msgraph_calendar_event_content(
            payload,
            expected_mailbox_user_id=_MAILBOX_USER_ID,
            expected_calendar_id=_CALENDAR_ID,
            max_chars=10_000,
        )


def test_read_occurrence_with_datetimeoffset_original_start() -> None:
    http = MagicMock()
    payload = _content_payload()
    payload["type"] = "occurrence"
    payload["originalStart"] = "2024-06-01T09:00:00Z"
    payload["seriesMasterId"] = "series-master-id"
    del payload["cancelledOccurrences"]
    http.get.return_value = _json_response(payload=payload)
    content = _reader(http).read_calendar_event_content(
        event=_valid_active_change(
            event_type=MsGraphCalendarEventType.OCCURRENCE,
            original_start_at=datetime(2024, 6, 1, 9, 0, tzinfo=timezone.utc),
            series_master_id="series-master-id",
        ),
        max_chars=10_000,
    )
    assert content.original_start_at == datetime(2024, 6, 1, 9, 0, tzinfo=timezone.utc)


# --- malformed provider responses ---


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        _content_payload(include_body=False),
        _content_payload() | {"body": None},
        _content_payload() | {"body": {"contentType": "text"}},
        _content_payload() | {"body": {"contentType": "text", "content": 1}},
        _content_payload() | {"body": {"contentType": "", "content": "x"}},
        _content_payload() | {"body": {"contentType": "rtf", "content": "x"}},
        _content_payload(body_text="bad\x00body"),
        _content_payload(include_subject_key=False),
        _content_payload(include_body_preview_key=False),
        _content_payload(include_organizer_key=False),
        _content_payload(include_location_key=False),
        _content_payload(include_recurrence_key=False),
        _content_payload() | {"attendees": None},
        _content_payload() | {"locations": "not-a-list"},
        _content_payload() | {"cancelledOccurrences": [None]},
        _content_payload(subject=123),
        _content_payload() | {"id": _OTHER_EVENT_ID},
        _content_payload() | {"changeKey": _OTHER_CHANGE_KEY},
        _content_payload() | {"lastModifiedDateTime": "2024-06-01T12:00:00"},
        _content_payload() | {"isAllDay": 1},
        _content_payload() | {"organizer": {"emailAddress": {"address": "bad\x00@x.com"}}},
        _content_payload(attendees=[{"type": "required"}]),
        _content_payload(location={"displayName": "x"}),
        _content_payload(recurrence={"pattern": {"type": "weekly", "interval": 1}}),
        _content_payload(removed=True),
    ],
)
def test_read_malformed_provider_payload(payload: object) -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=payload)
    with pytest.raises(
        (ValueError, MsGraphCalendarEventChanged),
        match=f"{_SAFE_ERROR}|{_KNOWLEDGE_SAFE_ERROR}|{_EVENTS_SAFE_ERROR}|{_CHANGED_ERROR}",
    ) as exc:
        _reader(http).read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    if isinstance(exc.value, ValueError):
        assert exc.value.__cause__ is None
    assert _SECRET_BODY not in str(exc.value)
    assert _SECRET_SUBJECT not in str(exc.value)


# --- consistency: identity mismatch and REMOVED ---


@pytest.mark.parametrize(
    "field_override",
    [
        {"event_id": _OTHER_EVENT_ID},
        {"change_key": _OTHER_CHANGE_KEY},
    ],
)
def test_read_identity_mismatch_raises_changed(field_override: dict[str, str]) -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_content_payload(**field_override))
    with pytest.raises(MsGraphCalendarEventChanged, match=_CHANGED_ERROR) as exc:
        _reader(http).read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert exc.value.__cause__ is None
    assert _CHANGE_KEY not in str(exc.value)
    assert _EVENT_ID not in str(exc.value)


def test_removed_input_rejected_before_http() -> None:
    http = MagicMock()
    with pytest.raises(MsGraphCalendarEventChanged, match=_CHANGED_ERROR):
        _reader(http).read_calendar_event_content(event=_valid_removed_change(), max_chars=10_000)
    http.get.assert_not_called()


def test_validate_event_content_returns_new_instance() -> None:
    original = _valid_event_content(
        subject=_SECRET_SUBJECT,
        organizer=parse_msgraph_calendar_participant(
            _participant(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
        ),
    )
    validated = validate_msgraph_calendar_event_content(
        original,
        event=_valid_active_change(),
        max_chars=10_000,
    )
    assert validated == original
    assert validated is not original


@pytest.mark.parametrize(
    "kwargs",
    [
        {"remote_id": _OTHER_EVENT_ID},
        {"calendar_remote_id": _OTHER_CALENDAR_ID},
        {"content_revision": _OTHER_CHANGE_KEY},
        {"mailbox_user_id": _OTHER_MAILBOX_USER_ID},
        {"body_content": 123},
        {"body_content": None},
        {"remote_id": None},
        {"calendar_remote_id": None},
        {"content_revision": None},
        {"attendees": []},
        {"organizer": {"emailAddress": {"address": "bad"}}},
        {"body_kind": "rtf"},
    ],
    ids=[
        "remote_id_mismatch",
        "calendar_mismatch",
        "revision_mismatch",
        "mailbox_mismatch",
        "body_content_int",
        "missing_body_content",
        "missing_remote_id",
        "missing_calendar",
        "missing_revision",
        "attendees_list_not_tuple",
        "malformed_organizer",
        "invalid_body_kind",
    ],
)
def test_model_construct_malformed_content_rejected(kwargs: dict[str, object]) -> None:
    base = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "calendar_remote_id": _CALENDAR_ID,
        "remote_id": _EVENT_ID,
        "content_revision": _CHANGE_KEY,
        "event_type": MsGraphCalendarEventType.SINGLE_INSTANCE,
        "body_kind": MsGraphCalendarBodyKind.TEXT,
        "body_content": "hello",
        "start_at": _START_AT,
        "end_at": _END_AT,
        "created_at": _CREATED_AT,
        "last_modified_at": _LAST_MODIFIED_AT,
        "importance": MsGraphCalendarImportance.NORMAL,
        "sensitivity": MsGraphCalendarSensitivity.NORMAL,
        "show_as": MsGraphCalendarShowAs.BUSY,
        "response_status": _default_response_status(),
        "is_all_day": False,
        "is_cancelled": False,
        "is_draft": False,
        "is_organizer": True,
        "is_online_meeting": False,
        "has_attachments": False,
        "hide_attendees": False,
        "allow_new_time_proposals": True,
        "response_requested": True,
        "is_reminder_on": True,
        "reminder_minutes_before_start": 15,
        "online_meeting_provider": MsGraphCalendarOnlineMeetingProvider.UNKNOWN,
        "organizer": None,
        "attendees": (),
        "locations": (),
    }
    malformed = MsGraphCalendarEventContent.model_construct(**{**base, **kwargs})
    with pytest.raises(
        (ValueError, MsGraphCalendarEventChanged),
        match=f"{_SAFE_ERROR}|{_CHANGED_ERROR}",
    ) as exc:
        validate_msgraph_calendar_event_content(
            malformed,
            event=_valid_active_change(),
            max_chars=10_000,
        )
    if isinstance(exc.value, ValueError):
        assert exc.value.__cause__ is None


def test_model_construct_missing_body_content_rejected() -> None:
    malformed = MsGraphCalendarEventContent.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=_CALENDAR_ID,
        remote_id=_EVENT_ID,
        content_revision=_CHANGE_KEY,
        event_type=MsGraphCalendarEventType.SINGLE_INSTANCE,
        body_kind=MsGraphCalendarBodyKind.TEXT,
        start_at=_START_AT,
        end_at=_END_AT,
        created_at=_CREATED_AT,
        last_modified_at=_LAST_MODIFIED_AT,
        importance=MsGraphCalendarImportance.NORMAL,
        sensitivity=MsGraphCalendarSensitivity.NORMAL,
        show_as=MsGraphCalendarShowAs.BUSY,
        response_status=_default_response_status(),
        is_all_day=False,
        is_cancelled=False,
        is_draft=False,
        is_organizer=True,
        is_online_meeting=False,
        has_attachments=False,
        hide_attendees=False,
        allow_new_time_proposals=True,
        response_requested=True,
        is_reminder_on=True,
        reminder_minutes_before_start=15,
        online_meeting_provider=MsGraphCalendarOnlineMeetingProvider.UNKNOWN,
        organizer=None,
        attendees=(),
        locations=(),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_calendar_event_content(
            malformed,
            event=_valid_active_change(),
            max_chars=10_000,
        )
    assert exc.value.__cause__ is None


def test_model_construct_wrong_type_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        validate_msgraph_calendar_event_content(
            "not-content",
            event=_valid_active_change(),
            max_chars=10_000,
        )


# --- limits ---


@pytest.mark.parametrize(
    "max_chars",
    [0, ABSOLUTE_CALENDAR_EVENT_CONTENT_MAX_CHARS + 1, True, "1000", None],
)
def test_invalid_max_chars_rejected_before_http(max_chars: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_calendar_event_content(
            event=_valid_active_change(),
            max_chars=max_chars,  # type: ignore[arg-type]
        )
    http.get.assert_not_called()


def test_content_over_limit_raises_too_large() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_content_payload(body_text="a" * 1_001),
    )
    with pytest.raises(MsGraphCalendarEventContentTooLarge, match=_TOO_LARGE_ERROR) as exc:
        _reader(http).read_calendar_event_content(event=_valid_active_change(), max_chars=1_000)
    assert exc.value.__cause__ is None
    assert _SECRET_BODY not in str(exc.value)


def test_validate_content_enforces_limit() -> None:
    content = _valid_event_content(body_content="a" * 1_001)
    with pytest.raises(MsGraphCalendarEventContentTooLarge, match=_TOO_LARGE_ERROR):
        validate_msgraph_calendar_event_content(
            content,
            event=_valid_active_change(),
            max_chars=1_000,
        )


@pytest.mark.parametrize(
    "max_chars",
    [0, ABSOLUTE_CALENDAR_EVENT_CONTENT_MAX_CHARS + 1, True, "1000", None],
)
def test_validate_content_rejects_invalid_max_chars(max_chars: object) -> None:
    content = _valid_event_content()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        validate_msgraph_calendar_event_content(
            content,
            event=_valid_active_change(),
            max_chars=max_chars,  # type: ignore[arg-type]
        )


# --- request verification ---


def test_request_exact_path_select_and_prefer_header() -> None:
    http, reader = _setup_happy_path()
    reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    assert http.get.call_args.args[0] == _CONTENT_PATH
    params = http.get.call_args.kwargs["params"]
    assert params["$select"] == _SELECT
    assert "$filter" not in params
    assert "$expand" not in params
    assert http.get.call_args.kwargs["headers"] == _TEXT_BODY_HEADERS


def test_request_uses_quoted_mailbox_calendar_and_event_id() -> None:
    http, reader = _setup_happy_path()
    reader.read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)
    path = http.get.call_args.args[0]
    assert _QUOTED_MAILBOX in path
    assert _QUOTED_CALENDAR in path
    assert _QUOTED_EVENT_ID in path


def test_status_404_maps_to_dependency_error() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(status_code=404)
    with pytest.raises(IntegrationDependencyError, match="dependency failure"):
        _reader(http).read_calendar_event_content(event=_valid_active_change(), max_chars=10_000)


# --- delegation ---


def test_graph_rest_client_delegates_calendar_event_content() -> None:
    http, _ = _setup_happy_path()
    result = _graph_client(http).read_calendar_event_content(event=_valid_active_change())
    assert result.body_content == "Hello, world."


def test_collaboration_suite_delegates_calendar_event_content() -> None:
    http, _ = _setup_happy_path()
    suite = _Ms365GraphCollaborationSuite(_graph_client(http))
    result = suite.read_calendar_event_content(event=_valid_active_change())
    assert result.remote_id == _EVENT_ID


def test_transport_and_reader_share_injected_http_client() -> None:
    http, _ = _setup_happy_path()
    client = _graph_client(http)
    client.read_calendar_event_content(event=_valid_active_change())
    assert client._knowledge_transport._http_client is http
    assert client._calendar_content_reader._transport._http_client is http


# --- custom client boundary ---


class _CustomSuiteWithoutCalendarContent(CollaborationSuite):
    def get_message(self, user_id: str, message_id: str):
        raise NotImplementedError

    def list_messages(self, user_id: str, *, folder: str = "inbox", limit: int = 25):
        raise NotImplementedError

    def send_mail(self, user_id: str, *, subject: str, body: str, to):
        raise NotImplementedError

    def list_calendar_events(self, user_id: str, *, start: str, end: str, limit: int = 50):
        raise NotImplementedError

    def get_user(self, user_id: str):
        raise NotImplementedError

    def reply_message(self, user_id: str, message_id: str, *, body: str) -> None:
        raise NotImplementedError

    def create_event(
        self,
        user_id: str,
        *,
        subject: str,
        start: str,
        end: str,
        location: str = "",
        attendees=(),
    ):
        raise NotImplementedError


class _CustomGraphCalendarContentClient(GraphRestClient):
    def __init__(self, content: MsGraphCalendarEventContent, http: MagicMock) -> None:
        super().__init__(_config(), http_client=http)
        self._custom_content = content

    def read_calendar_event_content(
        self,
        *,
        event: MsGraphCalendarEventChange,
        max_chars: int = DEFAULT_CALENDAR_EVENT_CONTENT_MAX_CHARS,
    ) -> MsGraphCalendarEventContent:
        return self._custom_content


def test_custom_client_without_calendar_content_capability_fails() -> None:
    with pytest.raises(IntegrationConfigurationError, match=_CAPABILITY_ERROR):
        _require_calendar_content_client(_CustomSuiteWithoutCalendarContent())


def test_custom_client_malformed_content_rejected() -> None:
    malformed = MsGraphCalendarEventContent.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=_CALENDAR_ID,
        remote_id=_EVENT_ID,
        content_revision=_CHANGE_KEY,
        event_type=MsGraphCalendarEventType.SINGLE_INSTANCE,
        body_kind=MsGraphCalendarBodyKind.TEXT,
        body_content=123,
        start_at=_START_AT,
        end_at=_END_AT,
        created_at=_CREATED_AT,
        last_modified_at=_LAST_MODIFIED_AT,
        importance=MsGraphCalendarImportance.NORMAL,
        sensitivity=MsGraphCalendarSensitivity.NORMAL,
        show_as=MsGraphCalendarShowAs.BUSY,
        response_status=_default_response_status(),
        is_all_day=False,
        is_cancelled=False,
        is_draft=False,
        is_organizer=True,
        is_online_meeting=False,
        has_attachments=False,
        hide_attendees=False,
        allow_new_time_proposals=True,
        response_requested=True,
        is_reminder_on=True,
        reminder_minutes_before_start=15,
        online_meeting_provider=MsGraphCalendarOnlineMeetingProvider.UNKNOWN,
    )
    client = _CustomGraphCalendarContentClient(content=malformed, http=MagicMock())
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _consume_calendar_event_content(
            client,
            event=_valid_active_change(),
            max_chars=10_000,
        )
    assert exc.value.__cause__ is None


def test_custom_client_valid_content_revalidated() -> None:
    supplied = _valid_event_content(
        subject=_SECRET_SUBJECT,
        body_content="x" * 16_001,
        organizer=parse_msgraph_calendar_participant(
            _participant(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
        ),
    )
    client = _CustomGraphCalendarContentClient(content=supplied, http=MagicMock())
    returned = _consume_calendar_event_content(
        client,
        event=_valid_active_change(),
        max_chars=20_000,
    )
    assert returned == supplied
    assert returned is not supplied
    assert returned.organizer is not supplied.organizer
    assert len(returned.body_content) == 16_001


@pytest.mark.parametrize(
    "max_chars",
    [0, ABSOLUTE_CALENDAR_EVENT_CONTENT_MAX_CHARS + 1, True, "1000", None],
)
def test_custom_client_invalid_max_chars_rejected(max_chars: object) -> None:
    supplied = _valid_event_content()
    client = _CustomGraphCalendarContentClient(content=supplied, http=MagicMock())
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _consume_calendar_event_content(
            client,
            event=_valid_active_change(),
            max_chars=max_chars,  # type: ignore[arg-type]
        )


def test_custom_client_identity_mismatch_rejected() -> None:
    supplied = _valid_event_content(remote_id=_OTHER_EVENT_ID)
    client = _CustomGraphCalendarContentClient(content=supplied, http=MagicMock())
    with pytest.raises(MsGraphCalendarEventChanged, match=_CHANGED_ERROR):
        _consume_calendar_event_content(
            client,
            event=_valid_active_change(),
            max_chars=10_000,
        )


# --- security ---


def test_security_repr_and_errors_hide_sensitive_fields() -> None:
    content = _valid_event_content(
        subject=_SECRET_SUBJECT,
        body_content=_SECRET_BODY,
        body_preview="preview secret",
        content_revision=_CHANGE_KEY,
        i_cal_uid=_I_CAL_UID,
        organizer=parse_msgraph_calendar_participant(
            _participant(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
        ),
        location=parse_msgraph_calendar_location(_location_payload(_SECRET_LOCATION)),
        attendees=(
            parse_msgraph_calendar_attendee(
                _attendee_payload(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
            ),
        ),
    )
    rendered = repr(content)
    assert _SECRET_SUBJECT not in rendered
    assert _SECRET_BODY not in rendered
    assert "preview secret" not in rendered
    assert _CHANGE_KEY not in rendered
    assert _I_CAL_UID not in rendered
    assert _SECRET_ADDRESS not in rendered
    assert _SECRET_DISPLAY_NAME not in rendered
    assert _SECRET_LOCATION not in rendered

    participant = parse_msgraph_calendar_participant(
        _participant(_SECRET_ADDRESS, display_name=_SECRET_DISPLAY_NAME)
    )
    participant_rendered = repr(participant)
    assert _SECRET_ADDRESS not in participant_rendered
    assert _SECRET_DISPLAY_NAME not in participant_rendered

    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR) as exc:
        _reader(MagicMock()).read_calendar_event_content(
            event=_valid_active_change(),
            max_chars=0,
        )
    assert _MAILBOX_USER_ID not in str(exc.value)
    assert _CHANGE_KEY not in str(exc.value)
