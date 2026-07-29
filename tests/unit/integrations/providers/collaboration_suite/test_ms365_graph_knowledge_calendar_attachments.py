# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph Calendar knowledge-read attachments surface."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.adapter import (
    _Ms365GraphCollaborationSuite,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.client import GraphRestClient
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    DEFAULT_GRAPH_BASE_URL,
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    ABSOLUTE_CALENDAR_ATTACHMENT_MAX_BYTES,
    DEFAULT_CALENDAR_ATTACHMENT_MAX_BYTES,
    MsGraphCalendarAttachment,
    MsGraphCalendarAttachmentKind,
    MsGraphCalendarAttachmentPage,
    MsGraphCalendarAttachmentTooLarge,
    MsGraphCalendarAttachmentsReader,
    MsGraphCalendarEventChange,
    MsGraphCalendarEventChangeKind,
    MsGraphCalendarEventChanged,
    MsGraphCalendarEventType,
    MsGraphCalendarFileAttachmentContent,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeTransport,
    parse_msgraph_calendar_attachment,
    validate_msgraph_calendar_attachment,
    validate_msgraph_calendar_attachment_page,
    validate_msgraph_calendar_attachments_continuation,
    validate_msgraph_calendar_file_attachment_content,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_MAILBOX_USER_ID = "user@contoso.com"
_OTHER_MAILBOX_USER_ID = "other@contoso.com"
_CALENDAR_ID = "calendar-abc-123"
_OTHER_CALENDAR_ID = "other-calendar"
_EVENT_ID = "AAMkAGI2THSAAA-immutable-opaque-id"
_OTHER_EVENT_ID = "AAMkAGI2THSBBB"
_CHANGE_KEY = "change-key-secret-value"
_OTHER_CHANGE_KEY = "other-change-key"
_ATTACHMENT_ID = "att-file-001"
_OTHER_ATTACHMENT_ID = "att-other-002"
_ATTACHMENT_NAME = "report.pdf"
_SECRET_ATTACHMENT_NAME = "secret-attachment-name"
_QUOTED_MAILBOX = quote(_MAILBOX_USER_ID, safe="")
_QUOTED_CALENDAR = quote(_CALENDAR_ID, safe="")
_QUOTED_EVENT_ID = quote(_EVENT_ID, safe="")
_QUOTED_OTHER_EVENT_ID = quote(_OTHER_EVENT_ID, safe="")
_QUOTED_ATTACHMENT_ID = quote(_ATTACHMENT_ID, safe="")
_SECRET_TOKEN = "secret-skiptoken-value"
_ATTACHMENTS_PATH = (
    f"/users/{_QUOTED_MAILBOX}/calendars/{_QUOTED_CALENDAR}"
    f"/events/{_QUOTED_EVENT_ID}/attachments"
)
_VALUE_PATH = (
    f"/users/{_QUOTED_MAILBOX}/calendars/{_QUOTED_CALENDAR}"
    f"/events/{_QUOTED_EVENT_ID}/attachments/{_QUOTED_ATTACHMENT_ID}/$value"
)
_OBSERVATION_PATH = (
    f"/users/{_QUOTED_MAILBOX}/calendars/{_QUOTED_CALENDAR}/events/{_QUOTED_EVENT_ID}"
)
_SELECT = "id,name,contentType,size,isInline,contentId,lastModifiedDateTime"
_OBSERVATION_SELECT = (
    "id,changeKey,type,start,end,originalStart,seriesMasterId,iCalUId,"
    "lastModifiedDateTime,isAllDay,isCancelled,isDraft,hasAttachments,isOnlineMeeting"
)
_SAFE_ERROR = "unexpected Microsoft Graph Calendar attachments response"
_REQUEST_ERROR = "invalid Microsoft Graph Calendar attachments request"
_CONT_ERROR = "invalid Microsoft Graph Calendar attachments continuation"
_UNSUPPORTED_ERROR = (
    "Microsoft Graph Calendar attachment content is not supported for this attachment type"
)
_INVALID_RESPONSE = "Microsoft Graph Calendar attachment response is invalid"
_CHANGED_ERROR = "Microsoft Graph Calendar event changed during read"
_IMMUTABLE_HEADER = {"Prefer": 'IdType="ImmutableId"'}
_OBSERVATION_HEADERS = {"Prefer": 'IdType="ImmutableId", outlook.timezone="UTC"'}


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


def _attachments_next_link() -> str:
    return (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
        f"{_QUOTED_CALENDAR}/events/{_QUOTED_EVENT_ID}/attachments?$skiptoken={_SECRET_TOKEN}"
    )


def _odata_attachments_next_link(
    event_id: str = _EVENT_ID,
    calendar_id: str = _CALENDAR_ID,
) -> str:
    escaped_event = event_id.replace("'", "''")
    escaped_calendar = calendar_id.replace("'", "''")
    return (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/"
        f"calendars('{escaped_calendar}')/events('{escaped_event}')/attachments?$skiptoken={_SECRET_TOKEN}"
    )


def _slash_attachments_next_link(event_id: str, calendar_id: str = _CALENDAR_ID) -> str:
    quoted_event = quote(event_id, safe="")
    quoted_calendar = quote(calendar_id, safe="")
    return (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
        f"{quoted_calendar}/events/{quoted_event}/attachments?$skiptoken={_SECRET_TOKEN}"
    )


def _odata_percent_encoded_attachments_next_link(
    event_id: str,
    calendar_id: str = _CALENDAR_ID,
) -> str:
    escaped_event = event_id.replace("'", "''")
    escaped_calendar = calendar_id.replace("'", "''")
    encoded_event = quote(escaped_event, safe="")
    encoded_calendar = quote(escaped_calendar, safe="")
    return (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/"
        f"calendars('{encoded_calendar}')/events('{encoded_event}')/attachments?$skiptoken={_SECRET_TOKEN}"
    )


def _page_payload(
    *,
    value: list[dict[str, Any]] | None = None,
    next_link: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"value": [] if value is None else value}
    if next_link is not None:
        payload["@odata.nextLink"] = next_link
    return payload


def _observation_payload(
    *,
    event_id: str = _EVENT_ID,
    change_key: str = _CHANGE_KEY,
) -> dict[str, Any]:
    return {
        "id": event_id,
        "changeKey": change_key,
        "type": "singleInstance",
        "start": {"dateTime": "2024-06-01T10:00:00", "timeZone": "UTC"},
        "end": {"dateTime": "2024-06-01T11:00:00", "timeZone": "UTC"},
        "lastModifiedDateTime": "2024-06-01T12:00:00Z",
        "isAllDay": False,
        "isCancelled": False,
        "isDraft": False,
        "hasAttachments": True,
        "isOnlineMeeting": False,
    }


def _attachment_payload(
    *,
    attachment_id: str = _ATTACHMENT_ID,
    odata_type: str = "#microsoft.graph.fileAttachment",
    name: str = _ATTACHMENT_NAME,
    content_type: str | None = "application/pdf",
    size: int = 42,
    is_inline: bool = False,
    content_id: str | None = None,
    include_content_type_key: bool = True,
    content_type_null: bool = False,
    include_content_id_key: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "@odata.type": odata_type,
        "id": attachment_id,
        "name": name,
        "size": size,
        "isInline": is_inline,
        "lastModifiedDateTime": "2024-06-01T12:00:00Z",
    }
    if include_content_type_key:
        payload["contentType"] = None if content_type_null else content_type
    if include_content_id_key:
        payload["contentId"] = content_id
    return payload


class _FakeStreamContext:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: tuple[bytes, ...] = (b"hello",),
        iter_raises: Exception | None = None,
    ) -> None:
        self._status_code = status_code
        self._headers = {} if headers is None else headers
        self._chunks = chunks
        self._iter_raises = iter_raises

    def __enter__(self) -> MagicMock:
        response = MagicMock()
        response.status_code = self._status_code
        response.headers = self._headers

        def _iter_bytes() -> Iterator[bytes]:
            if self._iter_raises is not None:
                raise self._iter_raises
            yield from self._chunks

        response.iter_bytes = _iter_bytes
        return response

    def __exit__(self, *args: object) -> None:
        return None


class _BrokenHeaderMapping(Mapping[str, str]):
    def __getitem__(self, key: str) -> str:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> Iterator[tuple[str, str]]:
        raise RuntimeError("broken headers")


def _valid_active_change(**overrides: object) -> MsGraphCalendarEventChange:
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
        "has_attachments": True,
        "is_online_meeting": False,
    }
    defaults.update(overrides)
    return MsGraphCalendarEventChange(**defaults)


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


def _valid_attachment_kwargs(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "calendar_remote_id": _CALENDAR_ID,
        "event_remote_id": _EVENT_ID,
        "event_revision": _CHANGE_KEY,
        "remote_id": _ATTACHMENT_ID,
        "kind": MsGraphCalendarAttachmentKind.FILE,
        "name": _ATTACHMENT_NAME,
        "content_type": "application/pdf",
        "size_bytes": 42,
        "is_inline": False,
        "content_id": None,
        "last_modified_at": datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return defaults


def _valid_attachment(**overrides: object) -> MsGraphCalendarAttachment:
    return MsGraphCalendarAttachment(**_valid_attachment_kwargs(**overrides))


def _valid_attachment_page(**overrides: object) -> MsGraphCalendarAttachmentPage:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "calendar_remote_id": _CALENDAR_ID,
        "event_remote_id": _EVENT_ID,
        "event_revision": _CHANGE_KEY,
        "items": (_valid_attachment(),),
        "continuation": None,
    }
    defaults.update(overrides)
    return MsGraphCalendarAttachmentPage(**defaults)


def _valid_file_content(
    data: bytes = b"hello-world",
    **overrides: object,
) -> MsGraphCalendarFileAttachmentContent:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "calendar_remote_id": _CALENDAR_ID,
        "event_remote_id": _EVENT_ID,
        "event_revision": _CHANGE_KEY,
        "attachment_remote_id": _ATTACHMENT_ID,
        "name": _ATTACHMENT_NAME,
        "content_type": "application/pdf",
        "is_inline": False,
        "content_id": None,
        "data": data,
        "size_bytes": len(data),
        "content_hash": hashlib.sha256(data).hexdigest(),
    }
    defaults.update(overrides)
    return MsGraphCalendarFileAttachmentContent(**defaults)


def _reader(http: MagicMock) -> MsGraphCalendarAttachmentsReader:
    return MsGraphCalendarAttachmentsReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
        graph_http_client=http,
    )


def _graph_client(http: MagicMock) -> GraphRestClient:
    return GraphRestClient(_config(), http_client=http)


def _parse_attachment(payload: dict[str, Any]) -> MsGraphCalendarAttachment:
    return parse_msgraph_calendar_attachment(payload, event=_valid_active_change())


def _setup_attachments_page(
    http: MagicMock,
    *,
    attachments: list[dict[str, Any]] | None = None,
    next_link: str | None = None,
) -> None:
    observation = _observation_payload()
    attachments_payload = _page_payload(
        value=attachments if attachments is not None else [_attachment_payload()],
        next_link=next_link,
    )
    http.get.side_effect = [
        _json_response(payload=observation),
        _json_response(payload=attachments_payload),
        _json_response(payload=observation),
    ]


def _setup_file_content(
    http: MagicMock,
    *,
    file_bytes: bytes = b"hello-world",
    attachment: MsGraphCalendarAttachment | None = None,
) -> MsGraphCalendarAttachment:
    if attachment is None:
        attachment = _valid_attachment(size_bytes=len(file_bytes))
    observation = _observation_payload()
    http.get.side_effect = [
        _json_response(payload=observation),
        _json_response(payload=observation),
    ]
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": str(len(file_bytes))},
        chunks=(file_bytes,),
    )
    return attachment


# --- parser: attachment kinds ---


@pytest.mark.parametrize(
    ("odata_type", "kind", "binary_supported"),
    [
        ("#microsoft.graph.fileAttachment", MsGraphCalendarAttachmentKind.FILE, True),
        ("#microsoft.graph.itemAttachment", MsGraphCalendarAttachmentKind.ITEM, False),
        ("#microsoft.graph.referenceAttachment", MsGraphCalendarAttachmentKind.REFERENCE, False),
        ("#microsoft.graph.unknownFutureType", MsGraphCalendarAttachmentKind.UNKNOWN, False),
    ],
)
def test_parse_attachment_kinds_and_binary_content_supported(
    odata_type: str,
    kind: MsGraphCalendarAttachmentKind,
    binary_supported: bool,
) -> None:
    attachment = _parse_attachment(_attachment_payload(odata_type=odata_type))
    assert attachment.kind is kind
    assert attachment.binary_content_supported is binary_supported


def test_parse_file_attachment_with_metadata() -> None:
    attachment = _parse_attachment(
        _attachment_payload(
            include_content_id_key=True,
            content_id="cid-inline-1",
            is_inline=True,
            content_type="image/png",
        )
    )
    assert attachment.remote_id == _ATTACHMENT_ID
    assert attachment.name == _ATTACHMENT_NAME
    assert attachment.content_type == "image/png"
    assert attachment.size_bytes == 42
    assert attachment.is_inline is True
    assert attachment.content_id == "cid-inline-1"
    assert attachment.event_revision == _CHANGE_KEY
    assert attachment.calendar_remote_id == _CALENDAR_ID


def test_parse_attachment_null_content_type() -> None:
    attachment = _parse_attachment(
        _attachment_payload(include_content_type_key=True, content_type_null=True)
    )
    assert attachment.content_type is None


def test_parse_attachment_missing_optional_content_type_key() -> None:
    payload = _attachment_payload(include_content_type_key=False)
    attachment = _parse_attachment(payload)
    assert attachment.content_type is None


@pytest.mark.parametrize("content_type", ["", "   "])
def test_parse_attachment_rejects_empty_content_type(content_type: str) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _parse_attachment(_attachment_payload(content_type=content_type))
    assert exc.value.__cause__ is None


@pytest.mark.parametrize("content_id", ["", "   "])
def test_parse_attachment_rejects_empty_content_id(content_id: str) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _parse_attachment(
            _attachment_payload(include_content_id_key=True, content_id=content_id)
        )
    assert exc.value.__cause__ is None


def test_parse_attachment_removed_event_rejected() -> None:
    with pytest.raises(MsGraphCalendarEventChanged):
        parse_msgraph_calendar_attachment(
            _attachment_payload(),
            event=_valid_removed_change(),
        )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"@odata.type": "#microsoft.graph.fileAttachment"},
        {"@odata.type": "#microsoft.graph.fileAttachment", "id": ""},
        {"@odata.type": "#microsoft.graph.fileAttachment", "id": _ATTACHMENT_ID},
        _attachment_payload() | {"name": 123},
        _attachment_payload() | {"size": -1},
        _attachment_payload() | {"size": "42"},
        _attachment_payload() | {"isInline": 1},
        _attachment_payload() | {"lastModifiedDateTime": "2024-06-01T12:00:00"},
        _attachment_payload() | {"@odata.type": ""},
        _attachment_payload() | {"@odata.type": 123},
        _attachment_payload() | {"contentType": 123},
        _attachment_payload(include_content_type_key=True, content_type=""),
        _attachment_payload(include_content_type_key=True, content_type="   "),
        _attachment_payload(include_content_id_key=True, content_id=123),
        _attachment_payload(include_content_id_key=True, content_id=""),
        _attachment_payload(include_content_id_key=True, content_id="   "),
        _attachment_payload() | {"id": "\x00bad"},
    ],
)
def test_parse_malformed_provider_payload(payload: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        parse_msgraph_calendar_attachment(payload, event=_valid_active_change())
    assert exc.value.__cause__ is None
    assert _ATTACHMENT_NAME not in str(exc.value)
    assert _ATTACHMENT_ID not in str(exc.value)


# --- model and deep validation ---


def test_validate_attachment_returns_new_instance() -> None:
    original = _valid_attachment(name=_SECRET_ATTACHMENT_NAME)
    validated = validate_msgraph_calendar_attachment(original)
    assert validated == original
    assert validated is not original


@pytest.mark.parametrize(
    "kwargs",
    [
        {"remote_id": None},
        {"kind": "file"},
        {"event_revision": None},
        {"size_bytes": -1},
        {"size_bytes": "42"},
        {"is_inline": 1},
        {"last_modified_at": datetime(2024, 6, 1, 12, 0)},
    ],
)
def test_model_construct_malformed_attachment_rejected(kwargs: dict[str, object]) -> None:
    malformed = MsGraphCalendarAttachment.model_construct(**{**_valid_attachment_kwargs(), **kwargs})
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_calendar_attachment(malformed)
    assert exc.value.__cause__ is None


def test_validate_attachment_page_returns_new_instances() -> None:
    original = _valid_attachment_page()
    validated = validate_msgraph_calendar_attachment_page(
        original,
        event=_valid_active_change(),
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == original
    assert validated is not original
    assert validated.items[0] is not original.items[0]


@pytest.mark.parametrize(
    "page_kwargs",
    [
        {"items": None},
        {"nested_malformed": True},
        {"continuation": "bad"},
    ],
)
def test_model_construct_malformed_page_rejected(page_kwargs: dict[str, object]) -> None:
    if page_kwargs.get("items") is None:
        malformed = MsGraphCalendarAttachmentPage.model_construct(
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_remote_id=_CALENDAR_ID,
            event_remote_id=_EVENT_ID,
            event_revision=_CHANGE_KEY,
        )
    elif page_kwargs.get("nested_malformed"):
        malformed = MsGraphCalendarAttachmentPage.model_construct(
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_remote_id=_CALENDAR_ID,
            event_remote_id=_EVENT_ID,
            event_revision=_CHANGE_KEY,
            items=(MsGraphCalendarAttachment.model_construct(),),
        )
    else:
        malformed = MsGraphCalendarAttachmentPage.model_construct(
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_remote_id=_CALENDAR_ID,
            event_remote_id=_EVENT_ID,
            event_revision=_CHANGE_KEY,
            items=(_valid_attachment(),),
            continuation=page_kwargs["continuation"],
        )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_calendar_attachment_page(
            malformed,
            event=_valid_active_change(),
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None


def test_attachment_page_model_rejects_duplicate_remote_ids() -> None:
    item = _valid_attachment()
    page = MsGraphCalendarAttachmentPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=_CALENDAR_ID,
        event_remote_id=_EVENT_ID,
        event_revision=_CHANGE_KEY,
        items=(item, item),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        validate_msgraph_calendar_attachment_page(
            page,
            event=_valid_active_change(),
            graph_base_url=_GRAPH_BASE,
        )


def test_validate_page_rejects_stale_event_revision() -> None:
    page = _valid_attachment_page(event_revision=_OTHER_CHANGE_KEY)
    with pytest.raises(MsGraphCalendarEventChanged):
        validate_msgraph_calendar_attachment_page(
            page,
            event=_valid_active_change(),
            graph_base_url=_GRAPH_BASE,
        )


def test_validate_file_content_returns_new_instance() -> None:
    data = b"content-bytes"
    original = _valid_file_content(data=data)
    validated = validate_msgraph_calendar_file_attachment_content(
        original,
        event=_valid_active_change(),
        attachment=_valid_attachment(size_bytes=len(data)),
        max_bytes=1024,
    )
    assert validated == original
    assert validated is not original


@pytest.mark.parametrize(
    "kwargs,error_type",
    [
        ({"data": b"x"}, ValueError),
        ({"size_bytes": 99}, ValueError),
        ({"content_hash": "not-a-valid-sha256-hash"}, ValueError),
        ({"attachment_remote_id": _OTHER_ATTACHMENT_ID}, MsGraphCalendarEventChanged),
        ({"event_revision": _OTHER_CHANGE_KEY}, MsGraphCalendarEventChanged),
        ({"data": bytearray(b"abc")}, ValueError),
        ({"size_bytes": True}, ValueError),
        ({"name": "other.pdf"}, MsGraphCalendarEventChanged),
        ({"content_type": "text/plain"}, MsGraphCalendarEventChanged),
        ({"is_inline": True}, MsGraphCalendarEventChanged),
        ({"content_id": "other-cid"}, MsGraphCalendarEventChanged),
    ],
)
def test_model_construct_malformed_file_content_rejected(
    kwargs: dict[str, object],
    error_type: type[BaseException],
) -> None:
    data = b"abc"
    base = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "calendar_remote_id": _CALENDAR_ID,
        "event_remote_id": _EVENT_ID,
        "event_revision": _CHANGE_KEY,
        "attachment_remote_id": _ATTACHMENT_ID,
        "name": _ATTACHMENT_NAME,
        "data": data,
        "size_bytes": len(data),
        "content_hash": hashlib.sha256(data).hexdigest(),
        "is_inline": False,
    }
    malformed = MsGraphCalendarFileAttachmentContent.model_construct(**{**base, **kwargs})
    with pytest.raises(
        error_type,
        match=_SAFE_ERROR if error_type is ValueError else "changed|supported",
    ) as exc:
        validate_msgraph_calendar_file_attachment_content(
            malformed,
            event=_valid_active_change(),
            attachment=_valid_attachment(size_bytes=len(data)),
            max_bytes=1024,
        )
    assert exc.value.__cause__ is None


@pytest.mark.parametrize(
    "missing_field",
    ["name", "is_inline", "data", "size_bytes", "content_hash"],
)
def test_model_construct_missing_file_content_field_rejected(missing_field: str) -> None:
    data = b"abc"
    base: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "calendar_remote_id": _CALENDAR_ID,
        "event_remote_id": _EVENT_ID,
        "event_revision": _CHANGE_KEY,
        "attachment_remote_id": _ATTACHMENT_ID,
        "name": _ATTACHMENT_NAME,
        "data": data,
        "size_bytes": len(data),
        "content_hash": hashlib.sha256(data).hexdigest(),
        "is_inline": False,
    }
    del base[missing_field]
    malformed = MsGraphCalendarFileAttachmentContent.model_construct(**base)
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_calendar_file_attachment_content(
            malformed,
            event=_valid_active_change(),
            attachment=_valid_attachment(size_bytes=len(data)),
            max_bytes=1024,
        )
    assert exc.value.__cause__ is None


def test_validate_file_content_data_length_mismatch_with_attachment_size() -> None:
    data = b"abc"
    content = _valid_file_content(data=data, size_bytes=len(data))
    with pytest.raises(MsGraphCalendarEventChanged):
        validate_msgraph_calendar_file_attachment_content(
            content,
            event=_valid_active_change(),
            attachment=_valid_attachment(size_bytes=99),
            max_bytes=1024,
        )


def test_validate_file_content_rejects_non_file_attachment() -> None:
    data = b"abc"
    content = _valid_file_content(data=data)
    with pytest.raises(IntegrationConfigurationError, match=_UNSUPPORTED_ERROR):
        validate_msgraph_calendar_file_attachment_content(
            content,
            event=_valid_active_change(),
            attachment=_valid_attachment(kind=MsGraphCalendarAttachmentKind.ITEM, size_bytes=len(data)),
            max_bytes=1024,
        )


def test_validate_file_content_too_large() -> None:
    data = b"toolarge"
    content = _valid_file_content(data=data)
    with pytest.raises(MsGraphCalendarAttachmentTooLarge):
        validate_msgraph_calendar_file_attachment_content(
            content,
            event=_valid_active_change(),
            attachment=_valid_attachment(size_bytes=len(data)),
            max_bytes=3,
        )


# --- requests and headers ---


def test_initial_request_path_and_params() -> None:
    http = MagicMock()
    _setup_attachments_page(http)
    _reader(http).read_attachments_page(
        event=_valid_active_change(),
        continuation=None,
        limit=25,
    )
    list_call = http.get.call_args_list[1]
    assert list_call.args[0] == _ATTACHMENTS_PATH
    params = list_call.kwargs["params"]
    assert params["$top"] == 25
    assert params["$select"] == _SELECT
    assert "$filter" not in params
    assert "$orderby" not in params
    assert "$expand" not in params
    assert "$search" not in params
    assert list_call.kwargs["headers"] == _IMMUTABLE_HEADER


def test_continuation_request_uses_full_url_without_params() -> None:
    http = MagicMock()
    observation = _observation_payload()
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_attachments_next_link(),
    )
    http.get.side_effect = [
        _json_response(payload=observation),
        _json_response(payload=_page_payload()),
        _json_response(payload=observation),
    ]
    _reader(http).read_attachments_page(
        event=_valid_active_change(),
        continuation=continuation,
        limit=100,
    )
    continuation_call = http.get.call_args_list[1]
    assert continuation_call.args[0] == _attachments_next_link()
    assert continuation_call.kwargs.get("params") is None
    assert continuation_call.kwargs["headers"] == _IMMUTABLE_HEADER


def test_observation_requests_use_select_and_immutable_header() -> None:
    http = MagicMock()
    _setup_attachments_page(http)
    _reader(http).read_attachments_page(
        event=_valid_active_change(),
        continuation=None,
        limit=10,
    )
    pre_observation = http.get.call_args_list[0]
    post_observation = http.get.call_args_list[2]
    assert pre_observation.args[0] == _OBSERVATION_PATH
    assert pre_observation.kwargs["params"]["$select"] == _OBSERVATION_SELECT
    assert pre_observation.kwargs["headers"] == _OBSERVATION_HEADERS
    assert post_observation.args[0] == _OBSERVATION_PATH


@pytest.mark.parametrize("limit", [0, 201, True, "25"])
def test_invalid_limit_rejected_before_http(limit: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_attachments_page(
            event=_valid_active_change(),
            continuation=None,
            limit=limit,  # type: ignore[arg-type]
        )
    http.get.assert_not_called()


@pytest.mark.parametrize("max_bytes", [0, ABSOLUTE_CALENDAR_ATTACHMENT_MAX_BYTES + 1, True, "10", None])
def test_invalid_max_bytes_rejected_before_http(max_bytes: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=_valid_attachment(),
            max_bytes=max_bytes,  # type: ignore[arg-type]
        )
    http.get.assert_not_called()
    http.stream.assert_not_called()


# --- continuation validation ---


def test_validate_continuation_accepts_next_page_slash_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_attachments_next_link(),
    )
    validated = validate_msgraph_calendar_attachments_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_id=_CALENDAR_ID,
        event_id=_EVENT_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation
    assert validated.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE
    assert validated.url == continuation.url


def test_validate_continuation_accepts_odata_key_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_attachments_next_link(),
    )
    validated = validate_msgraph_calendar_attachments_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_id=_CALENDAR_ID,
        event_id=_EVENT_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_accepts_event_literal_with_escaped_quotes() -> None:
    event_id = "evt'quote'part"
    calendar_id = "cal'quote'part"
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_attachments_next_link(event_id, calendar_id),
    )
    validated = validate_msgraph_calendar_attachments_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_id=calendar_id,
        event_id=event_id,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_accepts_uppercase_resource_names() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/USERS/{_QUOTED_MAILBOX}/CALENDARS/"
            f"{_QUOTED_CALENDAR}/EVENTS/{_QUOTED_EVENT_ID}/ATTACHMENTS?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    validated = validate_msgraph_calendar_attachments_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_id=_CALENDAR_ID,
        event_id=_EVENT_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_accepts_percent_encoded_event_literal() -> None:
    event_id = "evt/special"
    encoded = quote(event_id, safe="")
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
            f"{_QUOTED_CALENDAR}/events('{encoded}')/attachments?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    validated = validate_msgraph_calendar_attachments_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_id=_CALENDAR_ID,
        event_id=event_id,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


@pytest.mark.parametrize(
    "event_id",
    [
        "AAMk-delta-calendars-attachments",
        "AAMk-DELTA-CALENDARS-attachments",
        "opaque-delta-only",
        "opaque-calendars-only",
        "opaque-events-attachments",
    ],
)
def test_validate_continuation_accepts_opaque_event_id_with_reserved_substrings(
    event_id: str,
) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_slash_attachments_next_link(event_id),
    )
    validated = validate_msgraph_calendar_attachments_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_id=_CALENDAR_ID,
        event_id=event_id,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_accepts_odata_literal_with_quote_percent_and_delta() -> None:
    event_id = "evt'delta/special"
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_percent_encoded_attachments_next_link(event_id),
    )
    validated = validate_msgraph_calendar_attachments_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_id=_CALENDAR_ID,
        event_id=event_id,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_rejects_delta_resource_segment_but_accepts_delta_in_event_id() -> (
    None
):
    resource_delta_url = (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
        f"{_QUOTED_CALENDAR}/events/{_QUOTED_EVENT_ID}/attachments/delta?$skiptoken={_SECRET_TOKEN}"
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_calendar_attachments_continuation(
            MsGraphKnowledgeContinuation(
                kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=resource_delta_url,
            ),
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_id=_CALENDAR_ID,
            event_id=_EVENT_ID,
            graph_base_url=_GRAPH_BASE,
        )

    opaque_event_id = "AAMk-delta-in-opaque-id"
    validated = validate_msgraph_calendar_attachments_continuation(
        MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=_slash_attachments_next_link(opaque_event_id),
        ),
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_id=_CALENDAR_ID,
        event_id=opaque_event_id,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated.url == _slash_attachments_next_link(opaque_event_id)


@pytest.mark.parametrize(
    "url",
    [
        f"https://graph.microsoft.com/v1.0/users/{quote(_OTHER_MAILBOX_USER_ID, safe='')}/"
        f"calendars/{_QUOTED_CALENDAR}/events/{_QUOTED_EVENT_ID}/attachments?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
        f"{quote(_OTHER_CALENDAR_ID, safe='')}/events/{_QUOTED_EVENT_ID}/attachments?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
        f"{_QUOTED_CALENDAR}/events/{_QUOTED_OTHER_EVENT_ID}/attachments?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
        f"{_QUOTED_CALENDAR}/events/{_QUOTED_EVENT_ID}/attachments/delta?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/messages/"
        f"{_QUOTED_EVENT_ID}/attachments?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
        f"{_QUOTED_CALENDAR}/events/{_QUOTED_EVENT_ID}/attachments/{_QUOTED_ATTACHMENT_ID}/$value",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
        f"{_QUOTED_CALENDAR}/events/{_QUOTED_EVENT_ID}/attachments/extra?$skiptoken={_SECRET_TOKEN}",
        "https://graph.microsoft.com/v1.0/drives/drive-1/root/children?$skiptoken=x",
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars('{_QUOTED_CALENDAR}')"
        f"/events('unterminated?$skiptoken={_SECRET_TOKEN}",
    ],
)
def test_validate_continuation_rejects_invalid_urls(url: str) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_calendar_attachments_continuation(
            continuation,
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_id=_CALENDAR_ID,
            event_id=_EVENT_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert _SECRET_TOKEN not in str(exc.value)
    assert _MAILBOX_USER_ID not in str(exc.value)


def test_validate_continuation_rejects_delta_kind() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_attachments_next_link(),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_calendar_attachments_continuation(
            continuation,
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_id=_CALENDAR_ID,
            event_id=_EVENT_ID,
            graph_base_url=_GRAPH_BASE,
        )


def test_validate_continuation_rejects_wrong_object_type() -> None:
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_calendar_attachments_continuation(
            "bad",
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_id=_CALENDAR_ID,
            event_id=_EVENT_ID,
            graph_base_url=_GRAPH_BASE,
        )


def _assert_malformed_continuation_rejected(continuation: object) -> None:
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_calendar_attachments_continuation(
            continuation,
            mailbox_user_id=_MAILBOX_USER_ID,
            calendar_id=_CALENDAR_ID,
            event_id=_EVENT_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None
    assert _SECRET_TOKEN not in str(exc.value)
    assert _MAILBOX_USER_ID not in str(exc.value)
    assert _EVENT_ID not in str(exc.value)


@pytest.mark.parametrize(
    "continuation",
    [
        MsGraphKnowledgeContinuation.model_construct(),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        ),
        MsGraphKnowledgeContinuation.model_construct(url=_attachments_next_link()),
        MsGraphKnowledgeContinuation.model_construct(
            kind="next_page",
            url=_attachments_next_link(),
        ),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=123,
        ),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url="",
        ),
    ],
)
def test_validate_continuation_rejects_model_construct_malformed(
    continuation: MsGraphKnowledgeContinuation,
) -> None:
    _assert_malformed_continuation_rejected(continuation)


def test_validate_page_rejects_delta_continuation_kind() -> None:
    malformed = MsGraphCalendarAttachmentPage.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=_CALENDAR_ID,
        event_remote_id=_EVENT_ID,
        event_revision=_CHANGE_KEY,
        items=(_valid_attachment(),),
        continuation=MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=_attachments_next_link(),
        ),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        validate_msgraph_calendar_attachment_page(
            malformed,
            event=_valid_active_change(),
            graph_base_url=_GRAPH_BASE,
        )


# --- paging semantics ---


def test_first_page_with_next_page() -> None:
    http = MagicMock()
    _setup_attachments_page(http, next_link=_attachments_next_link())
    page = _reader(http).read_attachments_page(
        event=_valid_active_change(),
        continuation=None,
        limit=50,
    )
    assert page.has_more is True
    assert page.continuation is not None
    assert page.continuation.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE


def test_final_page_without_continuation() -> None:
    http = MagicMock()
    _setup_attachments_page(http)
    page = _reader(http).read_attachments_page(
        event=_valid_active_change(),
        continuation=None,
        limit=50,
    )
    assert page.has_more is False
    assert page.continuation is None


def test_event_changed_before_list() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_observation_payload(change_key=_OTHER_CHANGE_KEY),
    )
    with pytest.raises(MsGraphCalendarEventChanged):
        _reader(http).read_attachments_page(
            event=_valid_active_change(),
            continuation=None,
            limit=50,
        )
    http.get.assert_called_once()


def test_event_changed_after_list() -> None:
    http = MagicMock()
    observation_before = _observation_payload()
    observation_after = _observation_payload(change_key=_OTHER_CHANGE_KEY)
    http.get.side_effect = [
        _json_response(payload=observation_before),
        _json_response(payload=_page_payload(value=[_attachment_payload()])),
        _json_response(payload=observation_after),
    ]
    with pytest.raises(MsGraphCalendarEventChanged):
        _reader(http).read_attachments_page(
            event=_valid_active_change(),
            continuation=None,
            limit=50,
        )


# --- unsupported attachment types for content ---


@pytest.mark.parametrize(
    "kind",
    [
        MsGraphCalendarAttachmentKind.ITEM,
        MsGraphCalendarAttachmentKind.REFERENCE,
        MsGraphCalendarAttachmentKind.UNKNOWN,
    ],
)
def test_read_file_content_rejects_non_file_attachment(kind: MsGraphCalendarAttachmentKind) -> None:
    http = MagicMock()
    attachment = _valid_attachment(kind=kind)
    with pytest.raises(IntegrationConfigurationError, match=_UNSUPPORTED_ERROR):
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )
    http.get.assert_not_called()
    http.stream.assert_not_called()


def test_read_file_content_attachment_size_exceeds_limit_before_request() -> None:
    http = MagicMock()
    attachment = _valid_attachment(size_bytes=500)
    with pytest.raises(MsGraphCalendarAttachmentTooLarge):
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=100,
        )
    http.get.assert_not_called()
    http.stream.assert_not_called()


def test_read_file_content_stale_attachment_revision() -> None:
    http = MagicMock()
    attachment = _valid_attachment(event_revision=_OTHER_CHANGE_KEY)
    with pytest.raises(MsGraphCalendarEventChanged):
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )
    http.get.assert_not_called()


# --- file attachment content streaming ---


def test_download_small_file_with_sha256() -> None:
    data = b"small-file-content"
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=data)
    result = _reader(http).read_file_attachment_content(
        event=_valid_active_change(),
        attachment=attachment,
        max_bytes=1024,
    )
    assert isinstance(result, MsGraphCalendarFileAttachmentContent)
    assert result.data == data
    assert result.size_bytes == len(data)
    assert result.content_hash == hashlib.sha256(data).hexdigest()
    assert result.content_type == "application/pdf"
    assert result.event_revision == _CHANGE_KEY


def test_download_empty_file() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"")
    result = _reader(http).read_file_attachment_content(
        event=_valid_active_change(),
        attachment=attachment,
        max_bytes=1024,
    )
    assert result.data == b""
    assert result.size_bytes == 0


def test_download_multiple_chunks() -> None:
    data = b"hello-world"
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=data)
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": str(len(data))},
        chunks=(b"hello", b"-world"),
    )
    result = _reader(http).read_file_attachment_content(
        event=_valid_active_change(),
        attachment=attachment,
        max_bytes=1024,
    )
    assert result.data == data


def test_download_without_content_length() -> None:
    data = b"12345"
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=data)
    http.stream.return_value = _FakeStreamContext(chunks=(data,))
    result = _reader(http).read_file_attachment_content(
        event=_valid_active_change(),
        attachment=attachment,
        max_bytes=1024,
    )
    assert result.data == data


def test_download_content_length_exceeds_limit() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"x")
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "100"},
        chunks=(b"x",),
    )
    with pytest.raises(MsGraphCalendarAttachmentTooLarge):
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=10,
        )


def test_download_malformed_content_length() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"x")
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "not-a-number"},
        chunks=(b"x",),
    )
    with pytest.raises(IntegrationDependencyError, match=_INVALID_RESPONSE):
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


def test_download_bytes_exceed_limit_during_stream() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"abcdefghij")
    http.stream.return_value = _FakeStreamContext(
        chunks=(b"a" * 5, b"b" * 10),
    )
    with pytest.raises(MsGraphCalendarAttachmentTooLarge):
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=10,
        )


def test_download_content_length_mismatch() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"short")
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "10"},
        chunks=(b"short",),
    )
    with pytest.raises(MsGraphCalendarEventChanged):
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


def test_download_attachment_size_mismatch() -> None:
    data = b"hello"
    http = MagicMock()
    attachment = _valid_attachment(size_bytes=99)
    http.get.side_effect = [
        _json_response(payload=_observation_payload()),
        _json_response(payload=_observation_payload()),
    ]
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": str(len(data))},
        chunks=(data,),
    )
    with pytest.raises(MsGraphCalendarEventChanged):
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


def test_download_chunk_not_bytes() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"hello")

    class _BadChunkStream(_FakeStreamContext):
        def __enter__(self) -> MagicMock:
            response = super().__enter__()
            response.iter_bytes = lambda: iter(["not-bytes"])  # type: ignore[assignment]
            return response

    http.stream.return_value = _BadChunkStream(chunks=())
    with pytest.raises(IntegrationDependencyError, match=_INVALID_RESPONSE):
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


@pytest.mark.parametrize("status_code", [206, 302, 500])
def test_download_bad_status(status_code: int) -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"x")
    http.stream.return_value = _FakeStreamContext(status_code=status_code, chunks=(b"x",))
    with pytest.raises(IntegrationDependencyError):
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


@pytest.mark.parametrize("status_code", [401, 403])
def test_download_configuration_errors(status_code: int) -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"x")
    http.stream.return_value = _FakeStreamContext(status_code=status_code, chunks=(b"x",))
    with pytest.raises(IntegrationConfigurationError):
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


def test_download_stream_transport_exception() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"x")
    http.stream.side_effect = RuntimeError("stream failed")
    with pytest.raises(IntegrationDependencyError) as exc:
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )
    assert exc.value.__cause__ is None


def test_download_request_path_and_headers() -> None:
    data = b"bytes"
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=data)
    _reader(http).read_file_attachment_content(
        event=_valid_active_change(),
        attachment=attachment,
        max_bytes=1024,
    )
    call = http.stream.call_args
    assert call.args[1] == _VALUE_PATH
    assert call.kwargs["follow_redirects"] is False
    assert call.kwargs["headers"] == {
        "Accept": "application/octet-stream",
        "Prefer": 'IdType="ImmutableId"',
    }
    assert "params" not in call.kwargs


def test_download_integer_header_key_rejected() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"hello")
    http.stream.return_value = _FakeStreamContext(
        headers={1: "5"},
        chunks=(b"hello",),
    )
    with pytest.raises(IntegrationDependencyError, match=_INVALID_RESPONSE) as exc:
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )
    assert exc.value.__cause__ is None


def test_download_duplicate_content_length_different_case_rejected() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"hello")
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "5", "content-length": "5"},
        chunks=(b"hello",),
    )
    with pytest.raises(IntegrationDependencyError, match=_INVALID_RESPONSE):
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


def test_download_headers_items_raises_rejected() -> None:
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=b"hello")
    http.stream.return_value = _FakeStreamContext(
        headers=_BrokenHeaderMapping(),
        chunks=(b"hello",),
    )
    with pytest.raises(IntegrationDependencyError, match=_INVALID_RESPONSE):
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


def test_event_changed_after_download_bytes_not_returned() -> None:
    http = MagicMock()
    data = b"hello"
    attachment = _valid_attachment(size_bytes=len(data))
    http.get.side_effect = [
        _json_response(payload=_observation_payload()),
        _json_response(payload=_observation_payload(change_key=_OTHER_CHANGE_KEY)),
    ]
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": str(len(data))},
        chunks=(data,),
    )
    with pytest.raises(MsGraphCalendarEventChanged):
        _reader(http).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


def test_observation_before_and_after_download() -> None:
    data = b"payload"
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=data)
    _reader(http).read_file_attachment_content(
        event=_valid_active_change(),
        attachment=attachment,
        max_bytes=1024,
    )
    assert http.get.call_count == 2
    assert http.get.call_args_list[0].args[0] == _OBSERVATION_PATH
    assert http.get.call_args_list[1].args[0] == _OBSERVATION_PATH
    assert http.stream.call_count == 1


# --- delegation ---


def test_graph_rest_client_delegates_calendar_attachments_page() -> None:
    http = MagicMock()
    _setup_attachments_page(http)
    page = _graph_client(http).read_calendar_attachments_page(event=_valid_active_change())
    assert page.items[0].remote_id == _ATTACHMENT_ID


def test_graph_rest_client_delegates_calendar_file_attachment_content() -> None:
    data = b"delegated"
    http = MagicMock()
    attachment = _setup_file_content(http, file_bytes=data)
    result = _graph_client(http).read_calendar_file_attachment_content(
        event=_valid_active_change(),
        attachment=attachment,
    )
    assert result.data == data


def test_collaboration_suite_delegates_calendar_attachments() -> None:
    http = MagicMock()
    _setup_attachments_page(http)
    suite = _Ms365GraphCollaborationSuite(_graph_client(http))
    page = suite.read_calendar_attachments_page(event=_valid_active_change())
    assert page.items[0].kind is MsGraphCalendarAttachmentKind.FILE


def test_transport_and_reader_share_injected_http_client() -> None:
    http = MagicMock()
    _setup_attachments_page(http)
    client = _graph_client(http)
    client.read_calendar_attachments_page(event=_valid_active_change())
    assert client._knowledge_transport._http_client is http
    assert client._calendar_attachments_reader._transport._http_client is http
    assert client._calendar_attachments_reader._graph_http_client is http


def test_no_new_http_client_created() -> None:
    http = MagicMock()
    _setup_attachments_page(http)
    client = _graph_client(http)
    client.read_calendar_attachments_page(event=_valid_active_change())
    assert client._http_client is http


class _CustomGraphCalendarAttachmentsClient(GraphRestClient):
    def __init__(self, page: MsGraphCalendarAttachmentPage, http: MagicMock) -> None:
        super().__init__(_config(), http_client=http)
        self._custom_page = page

    def read_calendar_attachments_page(
        self,
        *,
        event: MsGraphCalendarEventChange,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphCalendarAttachmentPage:
        return self._custom_page


class _CustomCalendarAttachmentsReader:
    def __init__(self, page: MsGraphCalendarAttachmentPage) -> None:
        self._page = page

    def read_calendar_attachments_page(
        self,
        *,
        event: MsGraphCalendarEventChange,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphCalendarAttachmentPage:
        return self._page

    def read_calendar_file_attachment_content(
        self,
        *,
        event: MsGraphCalendarEventChange,
        attachment: MsGraphCalendarAttachment,
        max_bytes: int = DEFAULT_CALENDAR_ATTACHMENT_MAX_BYTES,
    ) -> MsGraphCalendarFileAttachmentContent:
        data = b"custom-binary"
        return MsGraphCalendarFileAttachmentContent(
            mailbox_user_id=event.mailbox_user_id,
            calendar_remote_id=event.calendar_remote_id,
            event_remote_id=event.remote_id,
            event_revision=event.change_key or "",
            attachment_remote_id=attachment.remote_id,
            name=attachment.name,
            content_type=attachment.content_type,
            is_inline=attachment.is_inline,
            content_id=attachment.content_id,
            data=data,
            size_bytes=len(data),
            content_hash=hashlib.sha256(data).hexdigest(),
        )


def test_custom_client_malformed_page_rejected() -> None:
    malformed_page = MsGraphCalendarAttachmentPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=_CALENDAR_ID,
        event_remote_id=_EVENT_ID,
        event_revision=_CHANGE_KEY,
        items=(MsGraphCalendarAttachment.model_construct(),),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_calendar_attachment_page(
            _CustomGraphCalendarAttachmentsClient(
                page=malformed_page,
                http=MagicMock(),
            ).read_calendar_attachments_page(event=_valid_active_change()),
            event=_valid_active_change(),
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None


def test_custom_client_valid_page_revalidated() -> None:
    supplied = _valid_attachment_page()
    returned = validate_msgraph_calendar_attachment_page(
        _CustomGraphCalendarAttachmentsClient(page=supplied, http=MagicMock()).read_calendar_attachments_page(
            event=_valid_active_change(),
        ),
        event=_valid_active_change(),
        graph_base_url=_GRAPH_BASE,
    )
    assert returned == supplied
    assert returned is not supplied
    assert returned.items[0] is not supplied.items[0]


def test_custom_client_rejects_malformed_continuation_without_url() -> None:
    malformed_page = MsGraphCalendarAttachmentPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=_CALENDAR_ID,
        event_remote_id=_EVENT_ID,
        event_revision=_CHANGE_KEY,
        items=(_valid_attachment(),),
        continuation=MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        ),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_calendar_attachment_page(
            malformed_page,
            event=_valid_active_change(),
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None


def test_custom_client_cross_event_continuation_rejected() -> None:
    wrong_continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
            f"{_QUOTED_CALENDAR}/events/{_QUOTED_OTHER_EVENT_ID}/attachments?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    page = MsGraphCalendarAttachmentPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=_CALENDAR_ID,
        event_remote_id=_EVENT_ID,
        event_revision=_CHANGE_KEY,
        items=(_valid_attachment(),),
        continuation=wrong_continuation,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        validate_msgraph_calendar_attachment_page(
            page,
            event=_valid_active_change(),
            graph_base_url=_GRAPH_BASE,
        )


@pytest.mark.parametrize(
    "kind",
    [
        MsGraphCalendarAttachmentKind.ITEM,
        MsGraphCalendarAttachmentKind.REFERENCE,
        MsGraphCalendarAttachmentKind.UNKNOWN,
    ],
)
def test_custom_client_non_file_attachment_content_rejected(kind: MsGraphCalendarAttachmentKind) -> None:
    attachment = _valid_attachment(kind=kind, size_bytes=13)
    custom = _CustomCalendarAttachmentsReader(page=_valid_attachment_page())
    with pytest.raises(IntegrationConfigurationError, match=_UNSUPPORTED_ERROR):
        validate_msgraph_calendar_file_attachment_content(
            custom.read_calendar_file_attachment_content(
                event=_valid_active_change(),
                attachment=attachment,
            ),
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=1024,
        )


@pytest.mark.parametrize("max_bytes", [0, ABSOLUTE_CALENDAR_ATTACHMENT_MAX_BYTES + 1, True, "10", None])
def test_custom_client_invalid_max_bytes_rejected(max_bytes: object) -> None:
    attachment = _valid_attachment(size_bytes=13)
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(MagicMock()).read_file_attachment_content(
            event=_valid_active_change(),
            attachment=attachment,
            max_bytes=max_bytes,  # type: ignore[arg-type]
        )


# --- security ---


def test_security_repr_and_errors_hide_sensitive_fields() -> None:
    attachment = _valid_attachment(name=_SECRET_ATTACHMENT_NAME)
    rendered = repr(attachment)
    assert _SECRET_ATTACHMENT_NAME not in rendered
    assert _CHANGE_KEY not in rendered

    page = _valid_attachment_page(
        continuation=MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=_attachments_next_link(),
        )
    )
    page_rendered = repr(page)
    assert _SECRET_TOKEN not in page_rendered
    assert "nextLink" not in page_rendered
    assert "skiptoken" not in page_rendered

    data = b"secret-bytes-payload"
    content = _valid_file_content(data=data)
    content_rendered = repr(content)
    assert data.decode() not in content_rendered


def test_default_max_bytes_constants() -> None:
    assert DEFAULT_CALENDAR_ATTACHMENT_MAX_BYTES == 10 * 1024 * 1024
    assert ABSOLUTE_CALENDAR_ATTACHMENT_MAX_BYTES == 25 * 1024 * 1024


def test_occurrence_with_datetimeoffset_original_start_observation_flow() -> None:
    http = MagicMock()
    occurrence_observation = {
        "id": _EVENT_ID,
        "changeKey": _CHANGE_KEY,
        "type": "occurrence",
        "start": {"dateTime": "2024-06-01T10:00:00", "timeZone": "UTC"},
        "end": {"dateTime": "2024-06-01T11:00:00", "timeZone": "UTC"},
        "originalStart": "2024-06-01T09:00:00Z",
        "seriesMasterId": "series-master-id",
        "lastModifiedDateTime": "2024-06-01T12:00:00Z",
        "isAllDay": False,
        "isCancelled": False,
        "isDraft": False,
        "hasAttachments": True,
        "isOnlineMeeting": False,
    }
    http.get.side_effect = [
        _json_response(payload=occurrence_observation),
        _json_response(payload=_page_payload(value=[_attachment_payload()])),
        _json_response(payload=occurrence_observation),
    ]
    event = _valid_active_change(
        event_type=MsGraphCalendarEventType.OCCURRENCE,
        original_start_at=datetime(2024, 6, 1, 9, 0, tzinfo=timezone.utc),
        series_master_id="series-master-id",
    )
    page = _reader(http).read_attachments_page(event=event, continuation=None, limit=50)
    assert len(page.items) == 1
    assert http.get.call_count == 3
