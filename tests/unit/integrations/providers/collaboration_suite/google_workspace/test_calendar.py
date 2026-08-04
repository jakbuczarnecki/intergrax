from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite import google_workspace
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceSourceKind,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GoogleWorkspaceCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read import (
    GOOGLE_CALENDAR_SOURCE_KIND,
    GoogleCalendarAccessRole,
    GoogleCalendarAttendeeResponseStatus,
    GoogleCalendarConferenceSolutionType,
    GoogleCalendarEventStatus,
    GoogleCalendarEventType,
    GoogleCalendarKnowledgeReader,
    GoogleCalendarReminderMethod,
    GoogleCalendarSyncToken,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.calendar import (
    _GOOGLE_CALENDAR_EVENT_FIELDS,
    _GOOGLE_CALENDAR_EVENTS_FIELDS,
    GoogleCalendarAttendee,
    GoogleCalendarEvent,
    GoogleCalendarEventDateTime,
    GoogleCalendarEventPage,
    GoogleCalendarPerson,
    GoogleCalendarReminder,
    GoogleCalendarReminders,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
    GoogleWorkspaceErrorKind,
    GoogleWorkspaceHttpTransport,
    GoogleWorkspacePageToken,
    GoogleWorkspaceRetryPolicy,
)

_CALENDAR_ID = "team@group.calendar.google.com"
_ENCODED_CALENDAR_ID = "team%40group.calendar.google.com"
_PAGE_TOKEN = GoogleWorkspacePageToken(value="page-1")
_SYNC_TOKEN = GoogleCalendarSyncToken(value="sync-1")
_UNEXPECTED_MESSAGE = "unexpected Google Calendar provider response"


@dataclass
class _RecordingTransport:
    responses: list[object] = field(default_factory=list)
    calls: list[dict[str, object]] = field(default_factory=list)
    exception: Exception | None = None

    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "source_kind": source_kind,
                "relative_path": relative_path,
                "params": dict(params or {}),
                "headers": dict(headers or {}),
            }
        )
        if self.exception is not None:
            raise self.exception
        response = self.responses.pop(0)
        return response  # type: ignore[return-value]


@dataclass(frozen=True)
class _HttpResponse:
    status_code: int
    headers: Mapping[str, str]
    content: bytes

    def json(self) -> object:
        return {}


@dataclass
class _ResponseExecutor:
    response: _HttpResponse
    calls: list[dict[str, object]] = field(default_factory=list)

    def get(
        self,
        *,
        url: str,
        params: Mapping[str, object] | None,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> _HttpResponse:
        self.calls.append(
            {
                "url": url,
                "params": dict(params or {}),
                "headers": dict(headers),
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.response


def _timed(start: str = "2026-01-01T10:00:00Z", end: str = "2026-01-01T11:00:00Z") -> dict[str, object]:
    return {"dateTime": start, "timeZone": "Europe/Warsaw"}


def _event(
    *,
    event_id: str = "event-1",
    status: str = "confirmed",
    cancelled: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": event_id,
        "iCalUID": "ical-1",
        "etag": "\"etag-1\"",
        "status": status,
        "eventType": "default",
        "summary": "Planning",
        "description": "A bounded description",
        "location": "Room 1",
        "htmlLink": "https://calendar.google.com/event",
        "created": "2026-01-01T09:00:00Z",
        "updated": "2026-01-01T09:30:00Z",
        "colorId": "9",
        "visibility": "private",
        "transparency": "opaque",
        "sequence": 2,
        "start": _timed(),
        "end": {"dateTime": "2026-01-01T11:00:00Z", "timeZone": "Europe/Warsaw"},
        "endTimeUnspecified": False,
        "recurrence": ["RRULE:FREQ=WEEKLY", "EXDATE:20260108T100000Z"],
        "recurringEventId": "master-1",
        "originalStartTime": _timed(),
        "creator": {"id": "person-1", "email": "creator@example.com", "displayName": "Creator", "self": True},
        "organizer": {"email": "organizer@example.com", "displayName": "Organizer", "self": False},
        "attendees": [
            {
                "email": "one@example.com",
                "displayName": "One",
                "responseStatus": "accepted",
                "additionalGuests": 1,
            },
            {"email": "two@example.com", "responseStatus": "tentative", "optional": True},
        ],
        "attendeesOmitted": False,
        "guestsCanInviteOthers": True,
        "guestsCanModify": False,
        "guestsCanSeeOtherGuests": True,
        "privateCopy": False,
        "locked": False,
        "hangoutLink": "https://meet.google.com/abc",
        "conferenceData": {
            "conferenceId": "conference-1",
            "signature": "signature-1",
            "notes": "Join",
            "entryPoints": [
                {
                    "entryPointType": "video",
                    "uri": "https://meet.google.com/abc",
                    "label": "Video",
                    "meetingCode": "abc-defg-hij",
                    "passcode": "secret",
                },
                {"entryPointType": "phone", "uri": "tel:+48123456789", "pin": "1234"},
            ],
            "conferenceSolution": {
                "key": {"type": "hangoutsMeet"},
                "name": "Google Meet",
                "iconUri": "https://example/icon.png",
            },
        },
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email", "minutes": 30},
                {"method": "popup", "minutes": 10},
            ],
        },
    }
    if cancelled:
        payload = {"id": event_id, "status": status}
    return payload


def _page(*, terminal: bool = True, events: list[dict[str, object]] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "summary": "Team calendar",
        "description": "Calendar description",
        "updated": "2026-01-01T09:00:00Z",
        "timeZone": "Europe/Warsaw",
        "accessRole": "writer",
        "items": events if events is not None else [_event()],
    }
    payload["nextSyncToken" if terminal else "nextPageToken"] = "sync-next" if terminal else "page-next"
    return payload


def _reader(payload: object) -> tuple[GoogleCalendarKnowledgeReader, _RecordingTransport]:
    transport = _RecordingTransport(responses=[payload])
    return GoogleCalendarKnowledgeReader(transport=transport), transport


def _construct_invalid_page_token(value: str) -> GoogleWorkspacePageToken:
    token = object.__new__(GoogleWorkspacePageToken)
    object.__setattr__(token, "value", value)
    return token


def test_exact_full_request_and_structured_page() -> None:
    reader, transport = _reader(_page(terminal=False))
    page = reader.list_events_page(calendar_id=_CALENDAR_ID, max_results=25)

    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["source_kind"] is GoogleWorkspaceSourceKind.CALENDAR
    assert call["relative_path"] == f"/calendars/{_ENCODED_CALENDAR_ID}/events"
    assert call["params"] == {
        "maxResults": 25,
        "showDeleted": True,
        "singleEvents": False,
        "fields": _GOOGLE_CALENDAR_EVENTS_FIELDS,
    }
    assert call["headers"] == {}
    assert page.calendar_id == _CALENDAR_ID
    assert page.summary == "Team calendar"
    assert page.access_role is GoogleCalendarAccessRole.WRITER
    assert page.next_page_token is not None
    assert page.next_sync_token is None
    event = page.events[0]
    assert event.id == "event-1"
    assert event.status is GoogleCalendarEventStatus.CONFIRMED
    assert event.start is not None and event.start.date_time.endswith("Z")
    assert event.creator is not None and event.creator.email == "creator@example.com"
    assert event.organizer is not None and event.organizer.display_name == "Organizer"
    assert event.attendees[0].response_status is GoogleCalendarAttendeeResponseStatus.ACCEPTED
    assert event.attendees[0].additional_guests == 1
    assert event.recurrence == ("RRULE:FREQ=WEEKLY", "EXDATE:20260108T100000Z")
    assert event.conference_data is not None
    assert event.conference_data.conference_solution is not None
    assert event.conference_data.conference_solution.type is GoogleCalendarConferenceSolutionType.HANGOUTS_MEET
    assert event.reminders is not None
    assert event.reminders.overrides[0].method is GoogleCalendarReminderMethod.EMAIL


def test_from_gmail_event_type_is_parsed() -> None:
    event = {
        "id": "gmail-event",
        "status": "confirmed",
        "eventType": "fromGmail",
        "start": {"dateTime": "2026-01-01T10:00:00Z"},
        "end": {"dateTime": "2026-01-01T11:00:00Z"},
    }
    reader, _ = _reader(_page(events=[event]))

    page = reader.list_events_page(calendar_id=_CALENDAR_ID)

    assert page.events[0].event_type is GoogleCalendarEventType.FROM_GMAIL


def test_none_access_role_is_parsed_on_terminal_page() -> None:
    reader, _ = _reader(
        {
            "accessRole": "none",
            "items": [],
            "nextSyncToken": "sync-next",
        }
    )

    page = reader.list_events_page(calendar_id=_CALENDAR_ID)

    assert page.access_role is GoogleCalendarAccessRole.NONE


def test_sms_reminder_method_is_rejected_safely() -> None:
    event = _event()
    event["reminders"] = {
        "useDefault": False,
        "overrides": [{"method": "sms", "minutes": 10}],
    }
    reader, _ = _reader(_page(events=[event]))

    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE) as exc_info:
        reader.list_events_page(calendar_id=_CALENDAR_ID)

    assert exc_info.value.__cause__ is None
    for secret in (
        _CALENDAR_ID,
        "event-1",
        "Planning",
        "creator@example.com",
        "sync-1",
        "page-1",
        "sms",
    ):
        assert secret not in str(exc_info.value)


def test_full_continuation_request_uses_only_page_token() -> None:
    reader, transport = _reader(_page(terminal=False))
    page = reader.list_events_page(
        calendar_id=_CALENDAR_ID,
        page_token=_PAGE_TOKEN,
    )

    assert page.next_page_token is not None
    assert transport.calls[0]["params"] == {
        "maxResults": 250,
        "showDeleted": True,
        "singleEvents": False,
        "fields": _GOOGLE_CALENDAR_EVENTS_FIELDS,
        "pageToken": "page-1",
    }
    assert "syncToken" not in transport.calls[0]["params"]


def test_incremental_first_page_uses_only_sync_token() -> None:
    reader, transport = _reader(_page())
    page = reader.list_events_page(
        calendar_id=_CALENDAR_ID,
        sync_token=_SYNC_TOKEN,
        max_results=2500,
    )
    assert page.next_sync_token is not None
    assert transport.calls[0]["params"] == {
        "maxResults": 2500,
        "showDeleted": True,
        "singleEvents": False,
        "fields": _GOOGLE_CALENDAR_EVENTS_FIELDS,
        "syncToken": "sync-1",
    }
    assert "timeMin" not in transport.calls[0]["params"]
    assert "updatedMin" not in transport.calls[0]["params"]
    assert "orderBy" not in transport.calls[0]["params"]
    assert "q" not in transport.calls[0]["params"]


def test_incremental_continuation_preserves_both_tokens() -> None:
    reader, transport = _reader(_page(terminal=False))
    page = reader.list_events_page(
        calendar_id=_CALENDAR_ID,
        page_token=_PAGE_TOKEN,
        sync_token=_SYNC_TOKEN,
        max_results=2500,
    )

    assert page.next_page_token is not None
    assert page.next_sync_token is None
    assert transport.calls[0]["params"] == {
        "maxResults": 2500,
        "showDeleted": True,
        "singleEvents": False,
        "fields": _GOOGLE_CALENDAR_EVENTS_FIELDS,
        "pageToken": "page-1",
        "syncToken": "sync-1",
    }
    for incompatible_filter in ("timeMin", "timeMax", "updatedMin", "orderBy", "q", "iCalUID"):
        assert incompatible_filter not in transport.calls[0]["params"]


@pytest.mark.parametrize(
    "calendar_id",
    ["", " ", " team", "team ", "a\x00b", "a\x7fb", "a/b", r"a\b", "x" * 1025, 123, True],
)
def test_invalid_calendar_id_makes_no_call(calendar_id: object) -> None:
    reader, transport = _reader(_page())
    with pytest.raises(IntegrationConfigurationError, match="invalid Google Calendar identifier"):
        reader.list_events_page(calendar_id=calendar_id)  # type: ignore[arg-type]
    assert transport.calls == []


@pytest.mark.parametrize("max_results", [0, 2501, True, 1.0, "25"])
def test_invalid_page_limit_makes_no_call(max_results: object) -> None:
    reader, transport = _reader(_page())
    with pytest.raises(IntegrationConfigurationError, match="invalid Google Calendar page limit"):
        reader.list_events_page(calendar_id=_CALENDAR_ID, max_results=max_results)  # type: ignore[arg-type]
    assert transport.calls == []


@pytest.mark.parametrize(
    ("token_kind", "token"),
    [
        ("page", _construct_invalid_page_token("")),
        ("page", _construct_invalid_page_token("x" * 4097)),
        ("page", _construct_invalid_page_token("\x00page")),
        ("sync", GoogleCalendarSyncToken.model_construct(value="")),
        ("sync", GoogleCalendarSyncToken.model_construct(value="x" * 4097)),
        ("sync", GoogleCalendarSyncToken.model_construct(value="\x00sync")),
    ],
)
def test_invalid_individual_tokens_make_no_call(token_kind: str, token: object) -> None:
    reader, transport = _reader(_page())
    token_argument = {"page_token": token} if token_kind == "page" else {"sync_token": token}
    message = "invalid Google Calendar page token" if token_kind == "page" else "invalid Google Calendar sync token"
    with pytest.raises(IntegrationConfigurationError, match=message):
        reader.list_events_page(calendar_id=_CALENDAR_ID, **token_argument)  # type: ignore[arg-type]
    assert transport.calls == []


def test_token_subclasses_make_no_call() -> None:
    class PageTokenSubclass(GoogleWorkspacePageToken):
        pass

    class SyncTokenSubclass(GoogleCalendarSyncToken):
        pass

    for token_kind, token, message in (
        ("page", PageTokenSubclass(value="page-1"), "invalid Google Calendar page token"),
        ("sync", SyncTokenSubclass(value="sync-1"), "invalid Google Calendar sync token"),
    ):
        reader, transport = _reader(_page())
        token_argument = {"page_token": token} if token_kind == "page" else {"sync_token": token}
        with pytest.raises(IntegrationConfigurationError, match=message):
            reader.list_events_page(calendar_id=_CALENDAR_ID, **token_argument)  # type: ignore[arg-type]
        assert transport.calls == []


def test_public_calendar_enum_values_match_provider_contract() -> None:
    assert GoogleCalendarEventType("fromGmail") is GoogleCalendarEventType.FROM_GMAIL
    assert GoogleCalendarAccessRole("none") is GoogleCalendarAccessRole.NONE
    with pytest.raises(ValueError):
        GoogleCalendarReminderMethod("sms")


@pytest.mark.parametrize(
    "payload",
    [
        {"items": [], "nextPageToken": "page", "nextSyncToken": "sync"},
        {"items": [], "nextPageToken": None},
        {"items": [], "nextSyncToken": None},
        {"items": []},
        {"items": [], "nextSyncToken": " sync"},
        {"items": [], "nextSyncToken": "\x00sync"},
        {"items": [], "nextSyncToken": "x" * 4097},
        {"items": [], "unknown": True, "nextSyncToken": "sync"},
        {"items": "not-a-list", "nextSyncToken": "sync"},
    ],
)
def test_continuation_and_provider_shape_errors_are_safe(payload: dict[str, object]) -> None:
    reader, _ = _reader(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE) as exc_info:
        reader.list_events_page(calendar_id=_CALENDAR_ID)
    assert exc_info.value.__cause__ is None
    assert _CALENDAR_ID not in str(exc_info.value)
    assert "sync" not in str(exc_info.value)


def test_cancelled_tombstone_and_all_day_event() -> None:
    all_day = _event(event_id="all-day")
    all_day["start"] = {"date": "2026-01-10"}
    all_day["end"] = {"date": "2026-01-11"}
    all_day["recurrence"] = []
    cancelled = _event(event_id="cancelled", status="cancelled", cancelled=True)
    reader, _ = _reader(_page(events=[all_day, cancelled]))
    page = reader.list_events_page(calendar_id=_CALENDAR_ID)
    assert page.events[0].start is not None
    assert page.events[0].start.date == "2026-01-10"
    assert page.events[0].start.date_time is None
    assert page.events[1].start is None


def test_integration_delegates_and_disabled_integration_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disabled = GoogleWorkspaceCollaborationSuiteIntegration(
        integration_id="calendar-test",
        provider_id="google_workspace",
    )
    with pytest.raises(IntegrationConfigurationError, match="integration is disabled"):
        disabled.list_calendar_events_page(calendar_id=_CALENDAR_ID)

    reader, transport = _reader(_page())
    monkeypatch.setattr(
        GoogleWorkspaceCollaborationSuiteIntegration,
        "_calendar_reader",
        lambda self: reader,
    )
    integration = GoogleWorkspaceCollaborationSuiteIntegration(
        integration_id="calendar-test",
        provider_id="google_workspace",
    )
    page = integration.list_calendar_events_page(
        calendar_id=_CALENDAR_ID,
        sync_token=_SYNC_TOKEN,
    )
    assert page.events[0].id == "event-1"
    assert transport.calls[0]["source_kind"] is GoogleWorkspaceSourceKind.CALENDAR


@pytest.mark.parametrize(
    "mutation",
    [
        lambda event: event.update(start={"date": "2026-01-01", "dateTime": "2026-01-01T10:00:00Z"}),
        lambda event: event.update(start={}),
        lambda event: event.update(start={"dateTime": "2026-01-01T10:00:00"}),
        lambda event: event.update(start={"date": "2026-02-30"}),
        lambda event: event.update(end={"dateTime": "2026-01-01T09:00:00Z"}),
        lambda event: event.update(end={"date": "2026-01-01"}),
        lambda event: event.update(end={"date": "2026-01-01"}),
    ],
)
def test_date_time_invariants_are_rejected(mutation) -> None:
    event = _event()
    mutation(event)
    reader, _ = _reader(_page(events=[event]))
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.list_events_page(calendar_id=_CALENDAR_ID)


def test_unknown_nested_field_and_explicit_null_are_rejected() -> None:
    event = _event()
    event["attendees"] = [{"email": "x@example.com", "unexpected": True}]
    reader, _ = _reader(_page(events=[event]))
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.list_events_page(calendar_id=_CALENDAR_ID)

    event = _event()
    event["summary"] = None
    reader, _ = _reader(_page(events=[event]))
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.list_events_page(calendar_id=_CALENDAR_ID)


def test_exact_provider_containers_and_bounds() -> None:
    class DictSubclass(dict[str, object]):
        pass

    class ListSubclass(list[object]):
        pass

    event = _event()
    event["creator"] = DictSubclass(event["creator"])  # type: ignore[arg-type]
    reader, _ = _reader(_page(events=[event]))
    with pytest.raises(IntegrationDependencyError):
        reader.list_events_page(calendar_id=_CALENDAR_ID)

    event = _event()
    event["attendees"] = ListSubclass(event["attendees"])  # type: ignore[arg-type]
    reader, _ = _reader(_page(events=[event]))
    with pytest.raises(IntegrationDependencyError):
        reader.list_events_page(calendar_id=_CALENDAR_ID)

    event = _event()
    event["recurrence"] = ["R" * 4097]
    reader, _ = _reader(_page(events=[event]))
    with pytest.raises(IntegrationDependencyError):
        reader.list_events_page(calendar_id=_CALENDAR_ID)


def test_aggregate_text_budget_is_bounded() -> None:
    event = _event()
    event["description"] = "x" * 100_000
    payload = _page(events=[event] * 41)
    reader, _ = _reader(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.list_events_page(calendar_id=_CALENDAR_ID, max_results=50)


def test_transport_error_boundary_and_api_error_passthrough() -> None:
    transport = _RecordingTransport(exception=RuntimeError("private network details"))
    reader = GoogleCalendarKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError, match="Google Calendar provider request failed") as exc_info:
        reader.list_events_page(calendar_id=_CALENDAR_ID)
    assert exc_info.value.__cause__ is None
    assert "private" not in str(exc_info.value)

    api_error = GoogleWorkspaceApiError(
        kind=GoogleWorkspaceErrorKind.NOT_FOUND,
        status_code=404,
        retry_after_seconds=None,
        safe_reason="not_found",
        attempts=1,
    )
    transport = _RecordingTransport(exception=api_error)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        GoogleCalendarKnowledgeReader(transport=transport).list_events_page(
            calendar_id=_CALENDAR_ID
        )
    assert exc_info.value is api_error


@pytest.mark.parametrize(
    ("status_code", "kind", "retryable"),
    [
        (410, GoogleWorkspaceErrorKind.INVALID_REQUEST, False),
        (400, GoogleWorkspaceErrorKind.INVALID_REQUEST, False),
        (429, GoogleWorkspaceErrorKind.RATE_LIMITED, True),
        (500, GoogleWorkspaceErrorKind.TEMPORARY, True),
    ],
)
def test_http_status_classification_preserves_safe_410_boundary(
    status_code: int,
    kind: GoogleWorkspaceErrorKind,
    retryable: bool,
) -> None:
    response = _HttpResponse(
        status_code=status_code,
        headers={"Content-Type": "application/json"},
        content=(
            b'{"error":{"status":"private provider body","errors":[{"reason":"private"}]},'
            b'"syncToken":"expired-sync","access_token":"secret-access-token",'
            b'"calendarId":"private-calendar","eventId":"private-event"}'
        ),
    )
    executor = _ResponseExecutor(response=response)
    transport = GoogleWorkspaceHttpTransport(
        executor=executor,
        retry_policy=GoogleWorkspaceRetryPolicy(max_attempts=1),
    )

    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.CALENDAR,
            relative_path="/calendars/team/events",
        )

    error = exc_info.value
    assert error.status_code == status_code
    assert error.kind is kind
    assert error.retryable is retryable
    assert error.attempts == 1
    assert "private" not in repr(error)
    assert "expired-sync" not in repr(error)
    assert "secret-access-token" not in repr(error)
    assert "private-calendar" not in repr(error)
    assert "private-event" not in repr(error)
    assert len(executor.calls) == 1


@pytest.mark.parametrize(
    ("page_token", "sync_token"),
    [(None, _SYNC_TOKEN), (_PAGE_TOKEN, _SYNC_TOKEN), (None, None)],
)
def test_calendar_reader_propagates_http_410_without_partial_page(
    page_token: GoogleWorkspacePageToken | None,
    sync_token: GoogleCalendarSyncToken | None,
) -> None:
    api_error = GoogleWorkspaceApiError(
        kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
        status_code=410,
        retry_after_seconds=None,
        safe_reason="private provider body",
        attempts=1,
    )
    reader = GoogleCalendarKnowledgeReader(
        transport=_RecordingTransport(exception=api_error)
    )

    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        reader.list_events_page(
            calendar_id=_CALENDAR_ID,
            page_token=page_token,
            sync_token=sync_token,
        )

    assert exc_info.value is api_error
    assert exc_info.value.status_code == 410
    assert exc_info.value.__cause__ is None
    assert "private provider body" not in repr(exc_info.value)
    assert "expired-sync" not in repr(exc_info.value)


def test_strict_public_models_and_construct_bypass() -> None:
    timed = GoogleCalendarEventDateTime(date_time="2026-01-01T10:00:00Z")
    person = GoogleCalendarPerson(email="person@example.com")
    attendee = GoogleCalendarAttendee(
        email="attendee@example.com",
        response_status=GoogleCalendarAttendeeResponseStatus.ACCEPTED,
    )
    reminder = GoogleCalendarReminder(method=GoogleCalendarReminderMethod.EMAIL, minutes=5)
    reminders = GoogleCalendarReminders(use_default=False, overrides=(reminder,))
    event = GoogleCalendarEvent(
        id="event",
        status=GoogleCalendarEventStatus.CONFIRMED,
        start=timed,
        end=GoogleCalendarEventDateTime(date_time="2026-01-01T11:00:00Z"),
        creator=person,
        attendees=(attendee,),
        reminders=reminders,
    )
    page = GoogleCalendarEventPage(
        calendar_id="calendar",
        events=(event,),
        next_sync_token=_SYNC_TOKEN,
    )
    assert page.events[0].creator is not person
    assert page.events[0].id == "event"
    assert "event" not in repr(page)
    with pytest.raises(ValidationError):
        GoogleCalendarEvent(
            id="event",
            status=GoogleCalendarEventStatus.CONFIRMED,
            start=timed,
            end=GoogleCalendarEventDateTime(date_time="2026-01-01T11:00:00Z"),
            attendees=[attendee],  # type: ignore[arg-type]
        )
    constructed_person = GoogleCalendarPerson.model_construct(email=None)
    with pytest.raises(ValidationError):
        GoogleCalendarAttendee(email="x@example.com", organizer=constructed_person)  # type: ignore[arg-type]
    constructed_token = GoogleCalendarSyncToken.model_construct(value="")
    with pytest.raises(ValidationError):
        GoogleCalendarEventPage(
            calendar_id="calendar",
            events=(event,),
            next_sync_token=constructed_token,
        )


def test_public_exports_resolve_exact_objects() -> None:
    assert GOOGLE_CALENDAR_SOURCE_KIND == "calendar"
    assert google_workspace.GoogleCalendarEvent is GoogleCalendarEvent
    assert google_workspace.GoogleCalendarKnowledgeReader is GoogleCalendarKnowledgeReader
    assert google_workspace.GoogleCalendarSyncToken is GoogleCalendarSyncToken


def test_read_event_uses_exact_single_event_request_and_existing_parser() -> None:
    reader, transport = _reader(_event(event_id="event+with"))

    event = reader.read_event(
        calendar_id="team+shared@group.calendar.google.com",
        event_id="event+with",
    )

    assert event.id == "event+with"
    assert event.status is GoogleCalendarEventStatus.CONFIRMED
    assert len(transport.calls) == 1
    assert transport.calls[0] == {
        "source_kind": GoogleWorkspaceSourceKind.CALENDAR,
        "relative_path": (
            "/calendars/team%2Bshared%40group.calendar.google.com/"
                "events/event%2Bwith"
        ),
        "params": {"fields": _GOOGLE_CALENDAR_EVENT_FIELDS},
        "headers": {},
    }


@pytest.mark.parametrize(
    "event_id",
    ["", " ", " event", "event ", "a\x00b", "a\x7fb", "a/b", r"a\b", "x" * 1025, 123, True],
)
def test_read_event_rejects_unsafe_event_id_before_call(event_id: object) -> None:
    reader, transport = _reader(_event())

    with pytest.raises(IntegrationConfigurationError, match="invalid Google Calendar event identifier") as exc_info:
        reader.read_event(calendar_id=_CALENDAR_ID, event_id=event_id)  # type: ignore[arg-type]

    assert exc_info.value.__cause__ is None
    assert transport.calls == []


@pytest.mark.parametrize(
    "calendar_id",
    ["", " ", " team", "team ", "a\x00b", "a\x7fb", "a/b", r"a\b", "x" * 1025, 123, True],
)
def test_read_event_rejects_unsafe_calendar_id_before_call(calendar_id: object) -> None:
    reader, transport = _reader(_event())

    with pytest.raises(IntegrationConfigurationError, match="invalid Google Calendar identifier"):
        reader.read_event(calendar_id=calendar_id, event_id="event-1")  # type: ignore[arg-type]

    assert transport.calls == []


def test_read_event_maps_malformed_and_transport_failures_safely() -> None:
    malformed = _event()
    malformed["unknown"] = "private"
    reader, transport = _reader(malformed)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE) as exc_info:
        reader.read_event(calendar_id=_CALENDAR_ID, event_id="event-1")
    assert exc_info.value.__cause__ is None
    assert _CALENDAR_ID not in str(exc_info.value)
    assert "event-1" not in str(exc_info.value)
    assert "private" not in str(exc_info.value)
    assert len(transport.calls) == 1

    transport = _RecordingTransport(exception=RuntimeError("private transport details"))
    with pytest.raises(IntegrationDependencyError, match="Google Calendar provider request failed") as exc_info:
        GoogleCalendarKnowledgeReader(transport=transport).read_event(
            calendar_id=_CALENDAR_ID,
            event_id="event-1",
        )
    assert exc_info.value.__cause__ is None
    assert "private" not in str(exc_info.value)

    api_error = GoogleWorkspaceApiError(
        kind=GoogleWorkspaceErrorKind.NOT_FOUND,
        status_code=404,
        retry_after_seconds=None,
        safe_reason="not_found",
        attempts=1,
    )
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        GoogleCalendarKnowledgeReader(
            transport=_RecordingTransport(exception=api_error)
        ).read_event(calendar_id=_CALENDAR_ID, event_id="event-1")
    assert exc_info.value is api_error


def test_integration_delegates_single_event_and_disabled_integration_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disabled = GoogleWorkspaceCollaborationSuiteIntegration(
        integration_id="calendar-test",
        provider_id="google_workspace",
    )
    with pytest.raises(IntegrationConfigurationError, match="integration is disabled"):
        disabled.read_calendar_event(calendar_id=_CALENDAR_ID, event_id="event-1")

    reader, transport = _reader(_event(event_id="event-1"))
    monkeypatch.setattr(
        GoogleWorkspaceCollaborationSuiteIntegration,
        "_calendar_reader",
        lambda self: reader,
    )
    integration = GoogleWorkspaceCollaborationSuiteIntegration(
        integration_id="calendar-test",
        provider_id="google_workspace",
    )
    event = integration.read_calendar_event(
        calendar_id=_CALENDAR_ID,
        event_id="event-1",
    )
    assert event.id == "event-1"
    assert transport.calls[0]["source_kind"] is GoogleWorkspaceSourceKind.CALENDAR
