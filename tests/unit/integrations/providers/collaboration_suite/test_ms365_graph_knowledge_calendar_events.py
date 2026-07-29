# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph Calendar knowledge-read calendar-view event delta."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    DEFAULT_GRAPH_BASE_URL,
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeSyncResetRequired,
    MsGraphKnowledgeTransport,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_events import (
    MsGraphCalendarEventChange,
    MsGraphCalendarEventChangeKind,
    MsGraphCalendarEventDeltaPage,
    MsGraphCalendarEventType,
    MsGraphCalendarEventsReader,
    MsGraphCalendarViewWindow,
    format_msgraph_calendar_window_datetime,
    parse_msgraph_calendar_event_change,
    validate_msgraph_calendar_event_change,
    validate_msgraph_calendar_event_delta_page,
    validate_msgraph_calendar_events_continuation,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_inventory import (
    MsGraphCalendar,
    MsGraphCalendarOnlineMeetingProvider,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_MAILBOX_USER_ID = "user@contoso.com"
_OTHER_MAILBOX_USER_ID = "other@contoso.com"
_CALENDAR_ID = "calendar-abc-123"
_OTHER_CALENDAR_ID = "other-calendar"
_EVENT_ID = "AAMkAGI2THSAAA-immutable-event-id"
_OTHER_EVENT_ID = "AAMkAGI2THSBBB"
_SERIES_MASTER_ID = "series-master-id-001"
_CHANGE_KEY = "change-key-secret-value"
_ICAL_UID = (
    "040000008200E00074C5B7101A82E00800000000000000000000000000000000000000000"
)
_QUOTED_MAILBOX = quote(_MAILBOX_USER_ID, safe="")
_QUOTED_CALENDAR_ID = quote(_CALENDAR_ID, safe="")
_QUOTED_OTHER_CALENDAR_ID = quote(_OTHER_CALENDAR_ID, safe="")
_SECRET_TOKEN = "secret-skiptoken-value"
_SECRET_DELTA_TOKEN = "secret-deltatoken-value"
_DELTA_PATH = (
    f"/users/{_QUOTED_MAILBOX}/calendars/{_QUOTED_CALENDAR_ID}/calendarView/delta"
)
_WINDOW_START = datetime(2024, 6, 1, 0, 0, tzinfo=timezone.utc)
_WINDOW_END = datetime(2024, 6, 30, 0, 0, tzinfo=timezone.utc)
_SAFE_ERROR = "unexpected Microsoft Graph Calendar events response"
_REQUEST_ERROR = "invalid Microsoft Graph Calendar events request"
_CONT_ERROR = "invalid Microsoft Graph Calendar events continuation"
_VALIDATION_ERROR = "Microsoft Graph Calendar events delta validation is not configured"
_EVENT_START = "2024-06-01T10:00:00"
_EVENT_END = "2024-06-01T11:00:00"
_LAST_MODIFIED = "2024-06-01T12:00:00Z"
_DELTA_HEADERS_TEMPLATE = (
    'IdType="ImmutableId", '
    'outlook.timezone="UTC", '
    'outlook.body-content-type="text", '
    "odata.maxpagesize={limit}"
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


def _next_link() -> str:
    return (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
        f"{_QUOTED_CALENDAR_ID}/calendarView/delta?$skiptoken={_SECRET_TOKEN}"
    )


def _delta_link() -> str:
    return (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
        f"{_QUOTED_CALENDAR_ID}/calendarView/delta?$deltatoken={_SECRET_DELTA_TOKEN}"
    )


def _odata_next_link(calendar_literal: str = _CALENDAR_ID) -> str:
    escaped = calendar_literal.replace("'", "''")
    return (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/"
        f"calendars('{escaped}')/calendarView/delta?$skiptoken={_SECRET_TOKEN}"
    )


def _page_payload(
    *,
    value: list[dict[str, Any]] | None = None,
    next_link: str | None = None,
    delta_link: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"value": [] if value is None else value}
    if next_link is not None:
        payload["@odata.nextLink"] = next_link
    if delta_link is not None:
        payload["@odata.deltaLink"] = delta_link
    return payload


def _utc_date_time(date_time: str) -> dict[str, str]:
    return {"dateTime": date_time, "timeZone": "UTC"}


def _active_event_payload(
    *,
    event_id: str = _EVENT_ID,
    change_key: str = _CHANGE_KEY,
    event_type: str = "singleInstance",
    start_at: str = _EVENT_START,
    end_at: str = _EVENT_END,
    last_modified_at: str = _LAST_MODIFIED,
    is_all_day: bool = False,
    is_cancelled: bool = False,
    is_draft: bool = False,
    has_attachments: bool = False,
    is_online_meeting: bool = False,
    original_start_at: str | None = None,
    series_master_id: str | None = None,
    i_cal_uid: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": event_id,
        "changeKey": change_key,
        "type": event_type,
        "start": _utc_date_time(start_at),
        "end": _utc_date_time(end_at),
        "lastModifiedDateTime": last_modified_at,
        "isAllDay": is_all_day,
        "isCancelled": is_cancelled,
        "isDraft": is_draft,
        "hasAttachments": has_attachments,
        "isOnlineMeeting": is_online_meeting,
    }
    if original_start_at is not None:
        payload["originalStart"] = _utc_date_time(original_start_at)
    if series_master_id is not None:
        payload["seriesMasterId"] = series_master_id
    if i_cal_uid is not None:
        payload["iCalUId"] = i_cal_uid
    return payload


def _removed_event_payload(
    *,
    event_id: str = _EVENT_ID,
    reason: str = "deleted",
) -> dict[str, Any]:
    return {"id": event_id, "@removed": {"reason": reason}}


def _valid_calendar(**overrides: object) -> MsGraphCalendar:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "remote_id": _CALENDAR_ID,
        "name": "Calendar",
        "change_key": _CHANGE_KEY,
        "is_default_calendar": True,
        "can_edit": True,
        "can_share": False,
        "can_view_private_items": True,
        "is_removable": False,
        "owner": None,
        "allowed_online_meeting_providers": (),
        "default_online_meeting_provider": MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
    }
    defaults.update(overrides)
    return MsGraphCalendar(**defaults)


def _valid_window(**overrides: object) -> MsGraphCalendarViewWindow:
    defaults: dict[str, object] = {
        "start_at": _WINDOW_START,
        "end_at": _WINDOW_END,
    }
    defaults.update(overrides)
    return MsGraphCalendarViewWindow(**defaults)


def _reader(http: MagicMock) -> MsGraphCalendarEventsReader:
    return MsGraphCalendarEventsReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
    )


def _valid_active_change_kwargs(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "calendar_remote_id": _CALENDAR_ID,
        "remote_id": _EVENT_ID,
        "kind": MsGraphCalendarEventChangeKind.ACTIVE,
        "change_key": _CHANGE_KEY,
        "event_type": MsGraphCalendarEventType.SINGLE_INSTANCE,
        "start_at": datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
        "end_at": datetime(2024, 6, 1, 11, 0, tzinfo=timezone.utc),
        "last_modified_at": datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
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


def _valid_delta_page(**overrides: object) -> MsGraphCalendarEventDeltaPage:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "calendar_remote_id": _CALENDAR_ID,
        "window": _valid_window(),
        "items": (_valid_active_change(),),
        "continuation": MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=_delta_link(),
        ),
    }
    defaults.update(overrides)
    return MsGraphCalendarEventDeltaPage(**defaults)


def _parse_active(payload: dict[str, Any]) -> MsGraphCalendarEventChange:
    return parse_msgraph_calendar_event_change(
        payload,
        expected_mailbox_user_id=_MAILBOX_USER_ID,
        expected_calendar_id=_CALENDAR_ID,
    )


def _validate_custom_page(page: MsGraphCalendarEventDeltaPage) -> MsGraphCalendarEventDeltaPage:
    return validate_msgraph_calendar_event_delta_page(
        page,
        calendar=_valid_calendar(),
        window=_valid_window(),
        graph_base_url=_GRAPH_BASE,
    )


# --- UTC window and normalization ---


def test_window_normalizes_non_utc_to_utc() -> None:
    eastern = timezone(timedelta(hours=-4))
    window = MsGraphCalendarViewWindow(
        start_at=datetime(2024, 6, 1, 0, 0, tzinfo=eastern),
        end_at=datetime(2024, 6, 30, 0, 0, tzinfo=eastern),
    )
    assert window.start_at.tzinfo == timezone.utc
    assert window.end_at.tzinfo == timezone.utc
    assert window.start_at == datetime(2024, 6, 1, 4, 0, tzinfo=timezone.utc)


def test_format_window_datetime_utc_z_suffix() -> None:
    value = datetime(2024, 6, 1, 12, 30, 45, tzinfo=timezone.utc)
    assert format_msgraph_calendar_window_datetime(value) == "2024-06-01T12:30:45Z"


def test_format_window_datetime_fractional_seconds() -> None:
    value = datetime(2024, 6, 1, 12, 30, 45, 500000, tzinfo=timezone.utc)
    assert format_msgraph_calendar_window_datetime(value) == "2024-06-01T12:30:45.5Z"


@pytest.mark.parametrize(
    "start_at,end_at",
    [
        (_WINDOW_START, _WINDOW_START),
        (_WINDOW_END, _WINDOW_START),
    ],
)
def test_invalid_window_bounds_rejected(start_at: datetime, end_at: datetime) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphCalendarViewWindow(start_at=start_at, end_at=end_at)


@pytest.mark.parametrize(
    "start_at,end_at",
    [
        (datetime(2024, 6, 1, 0, 0), _WINDOW_END),
        (_WINDOW_START, datetime(2024, 6, 30, 0, 0)),
    ],
)
def test_invalid_window_naive_datetime_rejected(
    start_at: datetime,
    end_at: datetime,
) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphCalendarViewWindow(start_at=start_at, end_at=end_at)


def test_reader_rejects_invalid_window_before_http() -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_delta_page(
            calendar=_valid_calendar(),
            window=MsGraphCalendarViewWindow.model_construct(
                start_at=_WINDOW_END,
                end_at=_WINDOW_START,
            ),
            continuation=None,
            limit=50,
        )
    http.get.assert_not_called()


# --- parser: active events ---


def test_parse_active_event_with_required_metadata() -> None:
    change = _parse_active(
        _active_event_payload(
            original_start_at="2024-06-01T09:00:00",
            series_master_id=_SERIES_MASTER_ID,
            i_cal_uid=_ICAL_UID,
            is_online_meeting=True,
            has_attachments=True,
        )
    )
    assert change.kind is MsGraphCalendarEventChangeKind.ACTIVE
    assert change.remote_id == _EVENT_ID
    assert change.change_key == _CHANGE_KEY
    assert change.event_type is MsGraphCalendarEventType.SINGLE_INSTANCE
    assert change.original_start_at == datetime(2024, 6, 1, 9, 0, tzinfo=timezone.utc)
    assert change.series_master_id == _SERIES_MASTER_ID
    assert change.i_cal_uid == _ICAL_UID
    assert change.is_online_meeting is True
    assert change.has_attachments is True
    assert change.is_removed_from_view is False


@pytest.mark.parametrize(
    ("raw_type", "expected"),
    [
        ("singleInstance", MsGraphCalendarEventType.SINGLE_INSTANCE),
        ("occurrence", MsGraphCalendarEventType.OCCURRENCE),
        ("exception", MsGraphCalendarEventType.EXCEPTION),
        ("seriesMaster", MsGraphCalendarEventType.SERIES_MASTER),
        ("futureEventType", MsGraphCalendarEventType.UNKNOWN),
    ],
)
def test_parse_active_event_types(
    raw_type: str,
    expected: MsGraphCalendarEventType,
) -> None:
    change = _parse_active(_active_event_payload(event_type=raw_type))
    assert change.event_type is expected


def test_parse_active_event_immutable_opaque_id() -> None:
    opaque_id = "AAMkAGI2THSAAA-immutable-opaque-id"
    change = _parse_active(_active_event_payload(event_id=opaque_id))
    assert change.remote_id == opaque_id


# --- parser: removed entries ---


def test_parse_removed_entry_minimal() -> None:
    change = _parse_active(_removed_event_payload())
    assert change.kind is MsGraphCalendarEventChangeKind.REMOVED
    assert change.is_removed_from_view is True
    assert change.removed_reason == "deleted"
    assert change.change_key is None
    assert change.event_type is None


def test_parse_removed_entry_unknown_future_reason() -> None:
    change = _parse_active(_removed_event_payload(reason="futureReason"))
    assert change.removed_reason == "futureReason"


# --- parser: malformed provider ---


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"changeKey": _CHANGE_KEY},
        {"id": None},
        {"id": ""},
        {"id": 123},
        {"id": _EVENT_ID, "@removed": None},
        {"id": _EVENT_ID, "@removed": "deleted"},
        {"id": _EVENT_ID, "@removed": {}},
        {"id": _EVENT_ID, "@removed": {"reason": ""}},
        {"id": _EVENT_ID, "@removed": {"reason": 1}},
        {"id": _EVENT_ID, "changeKey": _CHANGE_KEY},
        {"id": _EVENT_ID, "changeKey": ""},
        {"id": _EVENT_ID, "changeKey": 1},
        _active_event_payload() | {"type": ""},
        _active_event_payload() | {"start": {"dateTime": _EVENT_START}},
        _active_event_payload() | {"start": _utc_date_time(_EVENT_START) | {"timeZone": "PST"}},
        _active_event_payload() | {"end": {"timeZone": "UTC"}},
        _active_event_payload() | {"lastModifiedDateTime": "2024-06-01T12:00:00"},
        _active_event_payload() | {"isAllDay": 1},
        _active_event_payload() | {"isCancelled": 0},
        _active_event_payload() | {"isDraft": 1},
        _active_event_payload() | {"hasAttachments": 0},
        _active_event_payload() | {"isOnlineMeeting": 1},
        _active_event_payload(end_at="2024-06-01T09:00:00"),
        _active_event_payload() | {"seriesMasterId": ""},
        _active_event_payload() | {"iCalUId": ""},
    ],
)
def test_parse_malformed_provider_payload(payload: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        parse_msgraph_calendar_event_change(
            payload,
            expected_mailbox_user_id=_MAILBOX_USER_ID,
            expected_calendar_id=_CALENDAR_ID,
        )
    assert exc.value.__cause__ is None
    assert _EVENT_ID not in str(exc.value)
    assert _CHANGE_KEY not in str(exc.value)
    assert _ICAL_UID not in str(exc.value)


# --- model and deep validation ---


def test_validate_change_returns_new_instance() -> None:
    original = _valid_active_change(i_cal_uid=_ICAL_UID)
    validated = validate_msgraph_calendar_event_change(original)
    assert validated == original
    assert validated is not original


@pytest.mark.parametrize(
    "kwargs",
    [
        {"remote_id": None},
        {"kind": "active"},
        {"change_key": None},
        {"event_type": None},
        {"start_at": None},
        {"end_at": None},
        {"last_modified_at": None},
        {"is_all_day": None},
        {"removed_reason": None, "kind": MsGraphCalendarEventChangeKind.REMOVED},
        {"is_cancelled": 1},
        {"last_modified_at": datetime(2024, 6, 1, 12, 0)},
        {"event_type": "singleInstance"},
        {
            "start_at": datetime(2024, 6, 1, 11, 0, tzinfo=timezone.utc),
            "end_at": datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
        },
    ],
)
def test_model_construct_malformed_change_rejected(kwargs: dict[str, object]) -> None:
    if kwargs.get("kind") == MsGraphCalendarEventChangeKind.REMOVED:
        base = {
            "mailbox_user_id": _MAILBOX_USER_ID,
            "calendar_remote_id": _CALENDAR_ID,
            "remote_id": _EVENT_ID,
            "kind": MsGraphCalendarEventChangeKind.REMOVED,
            "removed_reason": "deleted",
        }
    else:
        base = _valid_active_change_kwargs()
    malformed = MsGraphCalendarEventChange.model_construct(**{**base, **kwargs})
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_calendar_event_change(malformed)
    assert exc.value.__cause__ is None


def test_validate_delta_page_returns_new_instances() -> None:
    original = _valid_delta_page()
    validated = validate_msgraph_calendar_event_delta_page(
        original,
        calendar=_valid_calendar(),
        window=_valid_window(),
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == original
    assert validated is not original
    assert validated.items[0] is not original.items[0]
    assert validated.continuation is not original.continuation


@pytest.mark.parametrize(
    "page_kwargs",
    [
        {"items": None},
        {"nested_malformed": True},
        {"continuation": "bad"},
        {"window_mismatch": True},
        {"calendar_mismatch": True},
    ],
)
def test_model_construct_malformed_page_rejected(page_kwargs: dict[str, object]) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_delta_link(),
    )
    if page_kwargs.get("items") is None:
        malformed = MsGraphCalendarEventDeltaPage.model_construct(
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_remote_id=_CALENDAR_ID,
            window=_valid_window(),
            continuation=continuation,
        )
    elif page_kwargs.get("nested_malformed"):
        malformed = MsGraphCalendarEventDeltaPage.model_construct(
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_remote_id=_CALENDAR_ID,
            window=_valid_window(),
            items=(MsGraphCalendarEventChange.model_construct(),),
            continuation=continuation,
        )
    elif page_kwargs.get("window_mismatch"):
        malformed = MsGraphCalendarEventDeltaPage.model_construct(
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_remote_id=_CALENDAR_ID,
            window=MsGraphCalendarViewWindow(
                start_at=datetime(2024, 7, 1, 0, 0, tzinfo=timezone.utc),
                end_at=datetime(2024, 7, 31, 0, 0, tzinfo=timezone.utc),
            ),
            items=(_valid_active_change(),),
            continuation=continuation,
        )
    elif page_kwargs.get("calendar_mismatch"):
        malformed = MsGraphCalendarEventDeltaPage.model_construct(
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_remote_id=_OTHER_CALENDAR_ID,
            window=_valid_window(),
            items=(_valid_active_change(),),
            continuation=continuation,
        )
    else:
        malformed = MsGraphCalendarEventDeltaPage.model_construct(
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_remote_id=_CALENDAR_ID,
            window=_valid_window(),
            items=(_valid_active_change(),),
            continuation=page_kwargs["continuation"],
        )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_calendar_event_delta_page(
            malformed,
            calendar=_valid_calendar(),
            window=_valid_window(),
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None


def test_delta_page_model_rejects_duplicate_remote_ids() -> None:
    item = _valid_active_change()
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_delta_link(),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphCalendarEventDeltaPage(
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_remote_id=_CALENDAR_ID,
            window=_valid_window(),
            items=(item, item),
            continuation=continuation,
        )


# --- requests and headers ---


def test_initial_request_path_params_and_headers() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(delta_link=_delta_link()))
    window = _valid_window()
    _reader(http).read_delta_page(
        calendar=_valid_calendar(),
        window=window,
        continuation=None,
        limit=25,
    )
    assert http.get.call_args.args[0] == _DELTA_PATH
    params = http.get.call_args.kwargs["params"]
    assert params["startDateTime"] == format_msgraph_calendar_window_datetime(window.start_at)
    assert params["endDateTime"] == format_msgraph_calendar_window_datetime(window.end_at)
    assert "$select" not in params
    assert "$top" not in params
    assert "$filter" not in params
    assert "$orderby" not in params
    assert "$expand" not in params
    assert "$search" not in params
    assert http.get.call_args.kwargs["headers"]["Content-Type"] == "application/json"
    assert http.get.call_args.kwargs["headers"]["Prefer"] == _DELTA_HEADERS_TEMPLATE.format(
        limit=25
    )


def test_continuation_request_uses_full_url_without_params() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(delta_link=_delta_link()))
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(),
    )
    _reader(http).read_delta_page(
        calendar=_valid_calendar(),
        window=_valid_window(),
        continuation=continuation,
        limit=100,
    )
    assert http.get.call_args.args[0] == _next_link()
    assert http.get.call_args.kwargs.get("params") is None
    assert http.get.call_args.kwargs["headers"]["Prefer"] == _DELTA_HEADERS_TEMPLATE.format(
        limit=100
    )


def test_delta_round_uses_immutable_and_utc_headers() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(delta_link=_delta_link()))
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_delta_link(),
    )
    _reader(http).read_delta_page(
        calendar=_valid_calendar(),
        window=_valid_window(),
        continuation=continuation,
        limit=50,
    )
    prefer = http.get.call_args.kwargs["headers"]["Prefer"]
    assert 'IdType="ImmutableId"' in prefer
    assert 'outlook.timezone="UTC"' in prefer
    assert "odata.maxpagesize=50" in prefer


@pytest.mark.parametrize("limit", [0, 1001, True, "25"])
def test_invalid_limit_rejected_before_http(limit: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_delta_page(
            calendar=_valid_calendar(),
            window=_valid_window(),
            continuation=None,
            limit=limit,  # type: ignore[arg-type]
        )
    http.get.assert_not_called()


# --- continuation validation ---


def test_validate_continuation_accepts_next_page_slash_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(),
    )
    validated = validate_msgraph_calendar_events_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_id=_CALENDAR_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation
    assert validated.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE
    assert validated.url == continuation.url


def test_validate_continuation_accepts_delta_slash_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_delta_link(),
    )
    validated = validate_msgraph_calendar_events_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_id=_CALENDAR_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation
    assert validated.kind is MsGraphKnowledgeContinuationKind.DELTA
    assert validated.url == continuation.url


def test_validate_continuation_accepts_odata_key_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_next_link(),
    )
    validated = validate_msgraph_calendar_events_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_id=_CALENDAR_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation
    assert validated.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE
    assert validated.url == continuation.url


def test_validate_continuation_accepts_resource_name_case_variations() -> None:
    url = (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/CALENDARS/"
        f"{_QUOTED_CALENDAR_ID}/CALENDARVIEW/DELTA?$skiptoken={_SECRET_TOKEN}"
    )
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    validated = validate_msgraph_calendar_events_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_id=_CALENDAR_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_accepts_calendar_literal_with_escaped_quotes() -> None:
    calendar_id = "calendar'quote'part"
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_odata_next_link(calendar_id),
    )
    validated = validate_msgraph_calendar_events_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_id=calendar_id,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation
    assert validated.kind is MsGraphKnowledgeContinuationKind.DELTA
    assert validated.url == continuation.url


@pytest.mark.parametrize(
    "url",
    [
        f"https://graph.microsoft.com/v1.0/users/{quote(_OTHER_MAILBOX_USER_ID, safe='')}/"
        f"calendars/{_QUOTED_CALENDAR_ID}/calendarView/delta?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
        f"{_QUOTED_OTHER_CALENDAR_ID}/calendarView/delta?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendar/events?"
        f"$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
        f"{_QUOTED_CALENDAR_ID}/events?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
        f"folder-1/messages/delta?$skiptoken={_SECRET_TOKEN}",
        "https://graph.microsoft.com/v1.0/drives/drive-1/root/delta?$skiptoken=x",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
        f"{_QUOTED_CALENDAR_ID}/calendarView/delta/extra?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars('unterminated"
        f"?$skiptoken={_SECRET_TOKEN}",
    ],
)
def test_validate_continuation_rejects_invalid_urls(url: str) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_calendar_events_continuation(
            continuation,
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_id=_CALENDAR_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert _SECRET_TOKEN not in str(exc.value)
    assert _MAILBOX_USER_ID not in str(exc.value)


def test_validate_continuation_rejects_wrong_object_type() -> None:
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_calendar_events_continuation(
            "bad",
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_id=_CALENDAR_ID,
            graph_base_url=_GRAPH_BASE,
        )


def _assert_malformed_continuation_rejected(continuation: object) -> None:
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_calendar_events_continuation(
            continuation,
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_id=_CALENDAR_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None
    assert _SECRET_TOKEN not in str(exc.value)
    assert _SECRET_DELTA_TOKEN not in str(exc.value)
    assert _MAILBOX_USER_ID not in str(exc.value)
    assert _CALENDAR_ID not in str(exc.value)


@pytest.mark.parametrize(
    "continuation",
    [
        MsGraphKnowledgeContinuation.model_construct(),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
        ),
        MsGraphKnowledgeContinuation.model_construct(url=_delta_link()),
        MsGraphKnowledgeContinuation.model_construct(
            kind="delta",
            url=_delta_link(),
        ),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=123,
        ),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url="",
        ),
    ],
)
def test_validate_continuation_rejects_model_construct_malformed(
    continuation: MsGraphKnowledgeContinuation,
) -> None:
    _assert_malformed_continuation_rejected(continuation)


def test_validate_delta_page_rejects_malformed_continuation_missing_url() -> None:
    malformed = MsGraphCalendarEventDeltaPage.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=_CALENDAR_ID,
        window=_valid_window(),
        items=(_valid_active_change(),),
        continuation=MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
        ),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_calendar_event_delta_page(
            malformed,
            calendar=_valid_calendar(),
            window=_valid_window(),
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None


def test_validate_delta_page_rejects_malformed_continuation_missing_kind() -> None:
    malformed = MsGraphCalendarEventDeltaPage.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=_CALENDAR_ID,
        window=_valid_window(),
        items=(_valid_active_change(),),
        continuation=MsGraphKnowledgeContinuation.model_construct(
            url=_delta_link(),
        ),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_calendar_event_delta_page(
            malformed,
            calendar=_valid_calendar(),
            window=_valid_window(),
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None


# --- delta semantics ---


def test_first_page_with_next_page() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(
            value=[_active_event_payload()],
            next_link=_next_link(),
        )
    )
    page = _reader(http).read_delta_page(
        calendar=_valid_calendar(),
        window=_valid_window(),
        continuation=None,
        limit=50,
    )
    assert page.has_more is True
    assert page.is_complete is False
    assert page.continuation.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE


def test_final_page_with_delta() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(
            value=[_active_event_payload()],
            delta_link=_delta_link(),
        )
    )
    page = _reader(http).read_delta_page(
        calendar=_valid_calendar(),
        window=_valid_window(),
        continuation=None,
        limit=50,
    )
    assert page.has_more is False
    assert page.is_complete is True


def test_incremental_round_from_delta_link() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(
            value=[_active_event_payload(event_id=_OTHER_EVENT_ID)],
            delta_link=_delta_link(),
        )
    )
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_delta_link(),
    )
    page = _reader(http).read_delta_page(
        calendar=_valid_calendar(),
        window=_valid_window(),
        continuation=continuation,
        limit=50,
    )
    assert page.items[0].remote_id == _OTHER_EVENT_ID


def test_duplicate_id_last_occurrence_wins() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(
            value=[
                _active_event_payload(event_id=_EVENT_ID, change_key="version-1"),
                _active_event_payload(event_id=_OTHER_EVENT_ID),
                _removed_event_payload(event_id=_EVENT_ID),
            ],
            delta_link=_delta_link(),
        )
    )
    page = _reader(http).read_delta_page(
        calendar=_valid_calendar(),
        window=_valid_window(),
        continuation=None,
        limit=50,
    )
    assert [item.remote_id for item in page.items] == [_OTHER_EVENT_ID, _EVENT_ID]
    assert page.items[1].kind is MsGraphCalendarEventChangeKind.REMOVED


def test_delta_page_requires_continuation() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(value=[_active_event_payload()])
    )
    with pytest.raises(ValueError, match="unexpected Microsoft Graph knowledge response"):
        _reader(http).read_delta_page(
            calendar=_valid_calendar(),
            window=_valid_window(),
            continuation=None,
            limit=50,
        )


def test_status_410_propagates_sync_reset() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(status_code=410)
    with pytest.raises(MsGraphKnowledgeSyncResetRequired, match="must restart"):
        _reader(http).read_delta_page(
            calendar=_valid_calendar(),
            window=_valid_window(),
            continuation=None,
            limit=50,
        )


# --- custom client validation ---


class _CustomCalendarEventsClient:
    def __init__(self, page: MsGraphCalendarEventDeltaPage) -> None:
        self._page = page

    def read_calendar_events_delta_page(
        self,
        *,
        calendar: MsGraphCalendar,
        window: MsGraphCalendarViewWindow,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphCalendarEventDeltaPage:
        del calendar, window, continuation, limit
        return self._page


def _graph_base_url_for_calendar_events_validation(client: object) -> str:
    if isinstance(client, MsGraphCalendarEventsReader):
        return client._config.graph_base_url
    raise IntegrationConfigurationError(_VALIDATION_ERROR)


def _read_calendar_events_via_custom_client(
    client: _CustomCalendarEventsClient,
) -> MsGraphCalendarEventDeltaPage:
    calendar = _valid_calendar()
    window = _valid_window()
    raw = client.read_calendar_events_delta_page(
        calendar=calendar,
        window=window,
        continuation=None,
        limit=50,
    )
    graph_base_url = _graph_base_url_for_calendar_events_validation(_reader(MagicMock()))
    return validate_msgraph_calendar_event_delta_page(
        raw,
        calendar=calendar,
        window=window,
        graph_base_url=graph_base_url,
    )


def test_custom_client_malformed_page_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _read_calendar_events_via_custom_client(
            _CustomCalendarEventsClient(
                page=MsGraphCalendarEventDeltaPage.model_construct(),
            )
        )
    assert exc.value.__cause__ is None


def test_custom_client_valid_page_revalidated() -> None:
    supplied = _valid_delta_page()
    returned = _read_calendar_events_via_custom_client(
        _CustomCalendarEventsClient(page=supplied)
    )
    assert returned == supplied
    assert returned is not supplied
    assert returned.items[0] is not supplied.items[0]
    assert returned.continuation is not supplied.continuation


def test_custom_client_cross_calendar_continuation_rejected() -> None:
    wrong_continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=(
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
            f"{_QUOTED_OTHER_CALENDAR_ID}/calendarView/delta?$deltatoken={_SECRET_DELTA_TOKEN}"
        ),
    )
    page = MsGraphCalendarEventDeltaPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=_CALENDAR_ID,
        window=_valid_window(),
        items=(_valid_active_change(),),
        continuation=wrong_continuation,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        _read_calendar_events_via_custom_client(_CustomCalendarEventsClient(page=page))


def test_custom_client_validation_not_configured() -> None:
    with pytest.raises(IntegrationConfigurationError, match=_VALIDATION_ERROR):
        _graph_base_url_for_calendar_events_validation(_CustomCalendarEventsClient(_valid_delta_page()))


def test_transport_and_reader_share_injected_http_client() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(delta_link=_delta_link()))
    reader = _reader(http)
    reader.read_delta_page(
        calendar=_valid_calendar(),
        window=_valid_window(),
        continuation=None,
        limit=50,
    )
    assert reader._transport._http_client is http


# --- security ---


def test_security_repr_and_errors_hide_sensitive_fields() -> None:
    change = _valid_active_change(
        change_key=_CHANGE_KEY,
        i_cal_uid=_ICAL_UID,
    )
    rendered = repr(change)
    assert _CHANGE_KEY not in rendered
    assert _ICAL_UID not in rendered

    page = _valid_delta_page(items=(change,))
    page_rendered = repr(page)
    assert _SECRET_DELTA_TOKEN not in page_rendered
    assert "deltaLink" not in page_rendered
    assert "deltatoken" not in page_rendered

    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR) as exc:
        _reader(MagicMock()).read_delta_page(
            calendar=_valid_calendar(),
            window=_valid_window(),
            continuation=None,
            limit=0,
        )
    assert _MAILBOX_USER_ID not in str(exc.value)
