# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for MsGraphCalendarKnowledgeAdapter."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationDependencyError,
)
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    DEFAULT_GRAPH_BASE_URL,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    DEFAULT_CALENDAR_EVENT_CONTENT_MAX_CHARS,
    MSGRAPH_CALENDAR_SOURCE_KIND,
    MSGRAPH_MAIL_SOURCE_KIND,
    MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
    MsGraphCalendar,
    MsGraphCalendarAttachment,
    MsGraphCalendarAttachmentKind,
    MsGraphCalendarAttachmentPage,
    MsGraphCalendarBodyKind,
    MsGraphCalendarEventChange,
    MsGraphCalendarEventChangeKind,
    MsGraphCalendarEventChanged,
    MsGraphCalendarEventContent,
    MsGraphCalendarEventContentTooLarge,
    MsGraphCalendarEventDeltaPage,
    MsGraphCalendarEventSnapshotPage,
    MsGraphCalendarEventType,
    MsGraphCalendarImportance,
    MsGraphCalendarOnlineMeetingProvider,
    MsGraphCalendarResponseStatus,
    MsGraphCalendarResponseType,
    MsGraphCalendarSensitivity,
    MsGraphCalendarShowAs,
    MsGraphCalendarViewWindow,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
)
from intergrax.runtime.vendor_knowledge.adapters.confluence_pages import (
    register_confluence_pages_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.jira_issues import (
    register_jira_issues_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_calendar import (
    MSGRAPH_CALENDAR_CURSOR_VERSION,
    MSGRAPH_CALENDAR_SCOPE_TYPE,
    MsGraphCalendarKnowledgeAdapter,
    _MsGraphCalendarCursor,
    _MsGraphCalendarRevision,
    _MsGraphCalendarScope,
    encode_msgraph_calendar_scope_id,
    register_msgraph_calendar_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_drive import (
    MSGRAPH_DRIVE_SCOPE_TYPE,
    register_msgraph_drive_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_mail import (
    MSGRAPH_MAIL_SCOPE_TYPE,
    register_msgraph_mail_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_channel import (
    MSGRAPH_TEAMS_CHANNEL_SCOPE_TYPE,
    register_msgraph_teams_channel_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_chat import (
    register_msgraph_teams_chat_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.contracts import VendorKnowledgeAdapter
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeChangeKind,
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgeItemIdentity,
    KnowledgeItemProvenance,
    KnowledgeItemRevision,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_MAILBOX_USER_ID = "user@contoso.com"
_OTHER_MAILBOX_USER_ID = "other@contoso.com"
_CALENDAR_ID = "calendar-abc-123"
_OTHER_CALENDAR_ID = "other-calendar"
_EVENT_ID = "AAMkAGI2THSAAA-immutable-event-id"
_OTHER_EVENT_ID = "AAMkAGI2THSBBB"
_CHANGE_KEY = "change-key-secret-value"
_SECRET_SKIP = "secret-skiptoken-value"
_SECRET_DELTA = "secret-deltatoken-value"
_WINDOW_START = datetime(2024, 6, 1, 0, 0, tzinfo=timezone.utc)
_WINDOW_END = datetime(2024, 6, 30, 0, 0, tzinfo=timezone.utc)
_EVENT_START = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
_EVENT_END = datetime(2024, 6, 1, 11, 0, tzinfo=timezone.utc)
_LAST_MODIFIED = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
_STRUCTURED_RECORD_SCHEMA = "msgraph.calendar.event.knowledge.v1"
_STRUCTURED_RECORD_MIME = "application/vnd.intergrax.msgraph-calendar-event+json"
_REMOVAL_SEMANTICS = "removed_from_synchronized_calendar_window_view"
_QUOTED_MAILBOX = quote(_MAILBOX_USER_ID, safe="")
_QUOTED_CALENDAR = quote(_CALENDAR_ID, safe="")
_QUOTED_OTHER_CALENDAR = quote(_OTHER_CALENDAR_ID, safe="")
_NEXT_URL = (
    f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/"
    f"calendarView/delta?$skiptoken={_SECRET_SKIP}"
)
_DELTA_URL = (
    f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/"
    f"calendarView/delta?$deltatoken={_SECRET_DELTA}"
)
_OTHER_DELTA_URL = (
    f"https://graph.microsoft.com/v1.0/users/{quote(_OTHER_MAILBOX_USER_ID, safe='')}/"
    f"calendarView/delta?$deltatoken=other"
)
_SNAPSHOT_NEXT_URL = (
    f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
    f"{_QUOTED_OTHER_CALENDAR}/calendarView?$skiptoken={_SECRET_SKIP}"
)
_ATTACHMENTS_NEXT_URL = (
    f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
    f"{_QUOTED_CALENDAR}/events/{quote(_EVENT_ID, safe='')}/attachments?$skiptoken={_SECRET_SKIP}"
)
_ATTACHMENT_ID = "attachment-001"


def _encode_canonical_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _window() -> MsGraphCalendarViewWindow:
    return MsGraphCalendarViewWindow(start_at=_WINDOW_START, end_at=_WINDOW_END)


def _default_calendar(**overrides: object) -> MsGraphCalendar:
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


def _snapshot_calendar(**overrides: object) -> MsGraphCalendar:
    return _default_calendar(
        remote_id=_OTHER_CALENDAR_ID,
        name="Other Calendar",
        is_default_calendar=False,
        **overrides,
    )


def _scope_id(
    *,
    calendar: MsGraphCalendar | None = None,
    window: MsGraphCalendarViewWindow | None = None,
) -> str:
    return encode_msgraph_calendar_scope_id(
        calendar=calendar or _default_calendar(),
        window=window or _window(),
    )


def _snapshot_scope_id(
    *,
    calendar: MsGraphCalendar | None = None,
    window: MsGraphCalendarViewWindow | None = None,
) -> str:
    return encode_msgraph_calendar_scope_id(
        calendar=calendar or _snapshot_calendar(),
        window=window or _window(),
    )


def _encode_event_identity(
    *,
    event_remote_id: str = _EVENT_ID,
    mailbox_user_id: str = _MAILBOX_USER_ID,
    calendar_remote_id: str = _CALENDAR_ID,
) -> str:
    return _encode_canonical_payload(
        {
            "schema_version": "msgraph.calendar.event-id.v1",
            "mailbox_user_id": mailbox_user_id,
            "calendar_remote_id": calendar_remote_id,
            "event_remote_id": event_remote_id,
        }
    )


def _encode_revision(change_key: str = _CHANGE_KEY) -> str:
    return _encode_canonical_payload(
        {
            "schema_version": "msgraph.calendar.revision.v1",
            "change_key": change_key,
        }
    )


def _source(
    *,
    remote_scope_id: str | None = None,
    remote_scope_type: str = MSGRAPH_CALENDAR_SCOPE_TYPE,
    parameters: dict[str, Any] | None = None,
    provider_id: str = MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    integration_kind: IntegrationCategory = IntegrationCategory.COLLABORATION_SUITE,
    source_kind: str = MSGRAPH_CALENDAR_SOURCE_KIND,
    safe_display_name: str = "Calendar",
) -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        scope=KnowledgeSourceScope(
            remote_scope_id=remote_scope_id or _scope_id(),
            remote_scope_type=remote_scope_type,
            safe_display_name=safe_display_name,
            parameters=parameters or {},
        ),
    )


def _snapshot_source(
    *,
    remote_scope_id: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> KnowledgeSourceRef:
    return _source(
        remote_scope_id=remote_scope_id or _snapshot_scope_id(),
        safe_display_name="Other Calendar",
        parameters=parameters,
    )


def _active_event(
    *,
    remote_id: str = _EVENT_ID,
    mailbox_user_id: str = _MAILBOX_USER_ID,
    calendar_remote_id: str = _CALENDAR_ID,
    change_key: str = _CHANGE_KEY,
    has_attachments: bool = False,
) -> MsGraphCalendarEventChange:
    return MsGraphCalendarEventChange(
        mailbox_user_id=mailbox_user_id,
        calendar_remote_id=calendar_remote_id,
        remote_id=remote_id,
        kind=MsGraphCalendarEventChangeKind.ACTIVE,
        change_key=change_key,
        event_type=MsGraphCalendarEventType.SINGLE_INSTANCE,
        start_at=_EVENT_START,
        end_at=_EVENT_END,
        last_modified_at=_LAST_MODIFIED,
        is_all_day=False,
        is_cancelled=False,
        is_draft=False,
        has_attachments=has_attachments,
        is_online_meeting=False,
    )


def _removed_event(
    *,
    remote_id: str = _OTHER_EVENT_ID,
    mailbox_user_id: str = _MAILBOX_USER_ID,
    calendar_remote_id: str = _CALENDAR_ID,
) -> MsGraphCalendarEventChange:
    return MsGraphCalendarEventChange(
        mailbox_user_id=mailbox_user_id,
        calendar_remote_id=calendar_remote_id,
        remote_id=remote_id,
        kind=MsGraphCalendarEventChangeKind.REMOVED,
        removed_reason="deleted",
    )


def _delta_page(
    *,
    items: tuple[MsGraphCalendarEventChange, ...],
    continuation_kind: MsGraphKnowledgeContinuationKind,
    url: str,
    mailbox_user_id: str = _MAILBOX_USER_ID,
    calendar_remote_id: str = _CALENDAR_ID,
    window: MsGraphCalendarViewWindow | None = None,
) -> MsGraphCalendarEventDeltaPage:
    return MsGraphCalendarEventDeltaPage(
        mailbox_user_id=mailbox_user_id,
        calendar_remote_id=calendar_remote_id,
        window=window or _window(),
        items=items,
        continuation=MsGraphKnowledgeContinuation(kind=continuation_kind, url=url),
    )


def _snapshot_page(
    *,
    items: tuple[MsGraphCalendarEventChange, ...],
    continuation_url: str | None = None,
    mailbox_user_id: str = _MAILBOX_USER_ID,
    calendar_remote_id: str = _OTHER_CALENDAR_ID,
    window: MsGraphCalendarViewWindow | None = None,
) -> MsGraphCalendarEventSnapshotPage:
    continuation = None
    if continuation_url is not None:
        continuation = MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=continuation_url,
        )
    return MsGraphCalendarEventSnapshotPage(
        mailbox_user_id=mailbox_user_id,
        calendar_remote_id=calendar_remote_id,
        window=window or _window(),
        items=items,
        continuation=continuation,
    )


def _default_response_status() -> MsGraphCalendarResponseStatus:
    return MsGraphCalendarResponseStatus(response=MsGraphCalendarResponseType.ORGANIZER)


def _event_content(
    *,
    remote_id: str = _EVENT_ID,
    calendar_remote_id: str = _CALENDAR_ID,
    change_key: str = _CHANGE_KEY,
    body_content: str = "Hello, world.",
    has_attachments: bool = False,
) -> MsGraphCalendarEventContent:
    return MsGraphCalendarEventContent(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=calendar_remote_id,
        remote_id=remote_id,
        content_revision=change_key,
        event_type=MsGraphCalendarEventType.SINGLE_INSTANCE,
        body_kind=MsGraphCalendarBodyKind.TEXT,
        body_content=body_content,
        start_at=_EVENT_START,
        end_at=_EVENT_END,
        created_at=_EVENT_START - timedelta(hours=1),
        last_modified_at=_LAST_MODIFIED,
        importance=MsGraphCalendarImportance.NORMAL,
        sensitivity=MsGraphCalendarSensitivity.NORMAL,
        show_as=MsGraphCalendarShowAs.BUSY,
        response_status=_default_response_status(),
        is_all_day=False,
        is_cancelled=False,
        is_draft=False,
        is_organizer=True,
        is_online_meeting=False,
        has_attachments=has_attachments,
        hide_attendees=False,
        allow_new_time_proposals=True,
        response_requested=True,
        is_reminder_on=True,
        reminder_minutes_before_start=15,
        online_meeting_provider=MsGraphCalendarOnlineMeetingProvider.UNKNOWN,
    )


def _valid_attachment(**overrides: object) -> MsGraphCalendarAttachment:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "calendar_remote_id": _CALENDAR_ID,
        "event_remote_id": _EVENT_ID,
        "event_revision": _CHANGE_KEY,
        "remote_id": _ATTACHMENT_ID,
        "kind": MsGraphCalendarAttachmentKind.FILE,
        "name": "agenda.pdf",
        "content_type": "application/pdf",
        "size_bytes": 1024,
        "is_inline": False,
        "last_modified_at": _LAST_MODIFIED,
    }
    defaults.update(overrides)
    return MsGraphCalendarAttachment(**defaults)


class _FakeCalendarCollaborationSuite(CollaborationSuite):
    def __init__(
        self,
        *,
        delta_pages: list[MsGraphCalendarEventDeltaPage] | None = None,
        snapshot_pages: list[MsGraphCalendarEventSnapshotPage] | None = None,
        content_by_key: dict[tuple[str, str], MsGraphCalendarEventContent] | None = None,
        attachment_pages: dict[tuple[str, str], MsGraphCalendarAttachmentPage] | None = None,
    ) -> None:
        self._delta_pages = list(delta_pages or [])
        self._snapshot_pages = list(snapshot_pages or [])
        self._content_by_key = dict(content_by_key or {})
        self._attachment_pages = dict(attachment_pages or {})
        self.delta_calls: list[dict[str, Any]] = []
        self.snapshot_calls: list[dict[str, Any]] = []
        self.content_calls: list[dict[str, Any]] = []
        self.attachment_calls: list[dict[str, Any]] = []
        self.forbidden_calls: list[str] = []

    def read_calendar_events_delta_page_by_reference(
        self,
        *,
        calendar,
        window: MsGraphCalendarViewWindow,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int,
    ) -> MsGraphCalendarEventDeltaPage:
        self.delta_calls.append(
            {
                "calendar": calendar,
                "window": window,
                "continuation": continuation,
                "limit": limit,
            }
        )
        if not self._delta_pages:
            raise IntegrationDependencyError("no delta pages configured")
        return self._delta_pages.pop(0)

    def read_calendar_events_snapshot_page_by_reference(
        self,
        *,
        calendar,
        window: MsGraphCalendarViewWindow,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int,
    ) -> MsGraphCalendarEventSnapshotPage:
        self.snapshot_calls.append(
            {
                "calendar": calendar,
                "window": window,
                "continuation": continuation,
                "limit": limit,
            }
        )
        if not self._snapshot_pages:
            raise IntegrationDependencyError("no snapshot pages configured")
        return self._snapshot_pages.pop(0)

    def read_calendar_event_content(
        self,
        *,
        event: MsGraphCalendarEventChange,
        max_chars: int,
    ) -> MsGraphCalendarEventContent:
        self.content_calls.append({"event": event, "max_chars": max_chars})
        return self._content_by_key[(event.remote_id, event.change_key)]

    def read_calendar_attachments_page(
        self,
        *,
        event: MsGraphCalendarEventChange,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 200,
    ) -> MsGraphCalendarAttachmentPage:
        self.attachment_calls.append(
            {
                "event": event,
                "continuation": continuation,
                "limit": limit,
            }
        )
        return self._attachment_pages[(event.remote_id, event.change_key)]

    def read_calendar_file_attachment_content(
        self,
        *,
        event: MsGraphCalendarEventChange,
        attachment: MsGraphCalendarAttachment,
        max_bytes: int,
    ):
        self.forbidden_calls.append("file_attachment_content")
        raise AssertionError("file attachment content must not be called")

    def read_mail_messages_delta_page(self, **kwargs: Any):
        self.forbidden_calls.append("mail")
        raise AssertionError("mail must not be called")

    def read_teams_chat_messages_snapshot_page_by_reference(self, **kwargs: Any):
        self.forbidden_calls.append("teams_chat")
        raise AssertionError("teams chat must not be called")

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


class _CalendarTestIntegration(Ms365GraphCollaborationSuiteIntegration):
    def _graph_base_url_for_calendar_validation(self) -> str:
        return DEFAULT_GRAPH_BASE_URL


def _integration(fake: _FakeCalendarCollaborationSuite) -> Ms365GraphCollaborationSuiteIntegration:
    return _CalendarTestIntegration.from_client(fake, enabled=True)


def _encode_cursor(payload: dict[str, Any]) -> KnowledgeCursor:
    return KnowledgeCursor(
        value=_encode_canonical_payload(payload),
        version=MSGRAPH_CALENDAR_CURSOR_VERSION,
    )


def _cursor_window_fields(window: MsGraphCalendarViewWindow | None = None) -> dict[str, str]:
    dumped = (window or _window()).model_dump(mode="json")
    return {
        "window_start_at": dumped["start_at"],
        "window_end_at": dumped["end_at"],
    }


def _base_metadata(*, has_attachments: bool = False) -> dict[str, object]:
    return {
        "event_state": "active",
        "event_type": "single_instance",
        "start_at": _EVENT_START.isoformat(),
        "end_at": _EVENT_END.isoformat(),
        "original_start_at": None,
        "last_modified_at": _LAST_MODIFIED.isoformat(),
        "series_master_id": None,
        "i_cal_uid": None,
        "is_all_day": False,
        "is_cancelled": False,
        "is_draft": False,
        "has_attachments": has_attachments,
        "is_online_meeting": False,
        "removal_semantics": _REMOVAL_SEMANTICS,
    }


def _event_descriptor(
    *,
    event_remote_id: str = _EVENT_ID,
    change_key: str = _CHANGE_KEY,
    mailbox_user_id: str = _MAILBOX_USER_ID,
    calendar_remote_id: str = _CALENDAR_ID,
    metadata: dict[str, Any] | None = None,
    metadata_only: bool = False,
    content_available: bool = True,
    item_type: str = "msgraph_calendar_event",
    content_mode: KnowledgeContentMode = KnowledgeContentMode.STRUCTURED_RECORD,
    provenance_source_kind: str = MSGRAPH_CALENDAR_SOURCE_KIND,
) -> KnowledgeItemDescriptor:
    opaque_id = _encode_event_identity(
        event_remote_id=event_remote_id,
        mailbox_user_id=mailbox_user_id,
        calendar_remote_id=calendar_remote_id,
    )
    base_metadata = _base_metadata()
    resolved_metadata = metadata if metadata_only else {**base_metadata, **(metadata or {})}
    return KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(
            remote_id=opaque_id,
            parent_remote_id=None,
            logical_key=None,
        ),
        revision=KnowledgeItemRevision(
            version=_encode_revision(change_key),
            etag=None,
            updated_at=_LAST_MODIFIED,
        ),
        title="Calendar event",
        item_type=item_type,
        content_mode=content_mode,
        content_available=content_available,
        provenance=KnowledgeItemProvenance(
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=provenance_source_kind,
            remote_id=opaque_id,
        ),
        metadata=resolved_metadata,
    )


def _assert_invalid_descriptor_boundary(exc_info: pytest.ExceptionInfo[VendorKnowledgeError]) -> None:
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert err.retryable is False
    rendered = f"{err!r} {err.safe_message}"
    for secret in (
        _MAILBOX_USER_ID,
        _CALENDAR_ID,
        _EVENT_ID,
        _CHANGE_KEY,
        "Hello, world.",
        "2024-06-01",
        "graph.microsoft.com",
        "Authorization",
    ):
        assert secret not in rendered


def _assert_error_hides_secrets(err: VendorKnowledgeError) -> None:
    rendered = f"{err!r} {err.safe_message}"
    for secret in (
        _MAILBOX_USER_ID,
        _CALENDAR_ID,
        _EVENT_ID,
        _CHANGE_KEY,
        _SECRET_SKIP,
        _SECRET_DELTA,
        _NEXT_URL,
        _DELTA_URL,
        "skiptoken",
        "deltatoken",
        "Hello, world.",
    ):
        assert secret not in rendered


async def _fetch_content_invalid_descriptor(
    item: KnowledgeItemDescriptor,
) -> pytest.ExceptionInfo[VendorKnowledgeError]:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        content_by_key={(_EVENT_ID, _CHANGE_KEY): _event_content(body_content="x")}
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=item,
        )
    return exc_info


# --- 1. Identity and registration ---


async def test_adapter_identity() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    assert adapter.provider_id == MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    assert adapter.integration_kind is IntegrationCategory.COLLABORATION_SUITE
    assert adapter.source_kind == MSGRAPH_CALENDAR_SOURCE_KIND
    assert isinstance(adapter, VendorKnowledgeAdapter)


async def test_registry_registration_and_coexistence() -> None:
    registry = KnowledgeAdapterRegistry()
    calendar_adapter = register_msgraph_calendar_knowledge_adapter(registry)
    register_msgraph_drive_knowledge_adapter(registry)
    register_msgraph_mail_knowledge_adapter(registry)
    register_msgraph_teams_chat_knowledge_adapter(registry)
    register_msgraph_teams_channel_knowledge_adapter(registry)
    register_confluence_pages_knowledge_adapter(registry)
    register_jira_issues_knowledge_adapter(registry)
    assert isinstance(calendar_adapter, MsGraphCalendarKnowledgeAdapter)
    assert isinstance(registry.resolve(source=_source()), MsGraphCalendarKnowledgeAdapter)


async def test_duplicate_registry_registration_rejected() -> None:
    registry = KnowledgeAdapterRegistry()
    register_msgraph_calendar_knowledge_adapter(registry)
    with pytest.raises(ValueError):
        register_msgraph_calendar_knowledge_adapter(registry)


# --- 2. Capabilities ---


async def test_capabilities_exact_set() -> None:
    caps = MsGraphCalendarKnowledgeAdapter().capabilities
    assert caps.full_inventory is True
    assert caps.incremental_changes is True
    assert caps.reconciliation is True
    assert caps.content_fetch is True
    assert caps.binary_content is False
    assert caps.rich_text_content is False
    assert caps.structured_content is True
    assert caps.permissions is False
    assert caps.tombstones is True
    assert caps.remote_versions is True


async def test_primary_delta_scope_capabilities() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    info = await adapter.inspect_scope(
        integration=_integration(_FakeCalendarCollaborationSuite()),
        source=_source(),
    )
    caps = info.capabilities
    assert caps.incremental_changes is True
    assert caps.tombstones is True


async def test_snapshot_scope_capabilities() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    info = await adapter.inspect_scope(
        integration=_integration(_FakeCalendarCollaborationSuite()),
        source=_snapshot_source(),
    )
    caps = info.capabilities
    assert caps.incremental_changes is False
    assert caps.tombstones is False
    assert caps.reconciliation is True


# --- 3. Scope ---


async def test_valid_calendar_scope_inspect() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    info = await adapter.inspect_scope(
        integration=_integration(_FakeCalendarCollaborationSuite()),
        source=_source(),
    )
    assert info.capabilities.structured_content is True
    assert info.source.scope.remote_scope_type == MSGRAPH_CALENDAR_SCOPE_TYPE


async def test_encoded_scope_id_hides_raw_ids() -> None:
    scope_id = _scope_id()
    assert _MAILBOX_USER_ID not in scope_id
    assert _CALENDAR_ID not in scope_id
    padding = "=" * (-len(scope_id) % 4)
    scope = _MsGraphCalendarScope.model_validate(
        json.loads(base64.urlsafe_b64decode(scope_id + padding).decode())
    )
    assert repr(scope) == (
        "_MsGraphCalendarScope(schema_version='msgraph.calendar.scope.v1', "
        "is_default_calendar=True, sync_strategy='primary_delta')"
    )


async def test_scope_encoding_round_trip_and_timezone_normalization() -> None:
    offset = timezone(timedelta(hours=1))
    local_start = datetime(2024, 6, 1, 1, 0, tzinfo=offset)
    local_end = datetime(2024, 6, 30, 1, 0, tzinfo=offset)
    window = MsGraphCalendarViewWindow(start_at=local_start, end_at=local_end)
    scope_id = encode_msgraph_calendar_scope_id(calendar=_default_calendar(), window=window)
    decoded = _MsGraphCalendarScope.model_validate(
        json.loads(base64.urlsafe_b64decode(scope_id + "==").decode())
    )
    assert decoded.mailbox_user_id == _MAILBOX_USER_ID
    assert decoded.calendar_remote_id == _CALENDAR_ID
    assert decoded.sync_strategy == "primary_delta"
    assert decoded.window_start_at == datetime(2024, 6, 1, 0, 0, tzinfo=timezone.utc)
    assert decoded.window_end_at == datetime(2024, 6, 30, 0, 0, tzinfo=timezone.utc)


async def test_snapshot_scope_uses_snapshot_strategy() -> None:
    scope_id = _snapshot_scope_id()
    decoded = _MsGraphCalendarScope.model_validate(
        json.loads(base64.urlsafe_b64decode(scope_id + "==").decode())
    )
    assert decoded.sync_strategy == "snapshot"
    assert decoded.is_default_calendar is False


@pytest.mark.parametrize(
    "source",
    [
        _source(provider_id="other"),
        _source(integration_kind=IntegrationCategory.COLLABORATION_SUITE, source_kind="wrong"),
        _source(source_kind=MSGRAPH_MAIL_SOURCE_KIND),
        _source(source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND),
        _source(remote_scope_type=MSGRAPH_MAIL_SCOPE_TYPE),
        _source(remote_scope_type=MSGRAPH_DRIVE_SCOPE_TYPE),
        _source(remote_scope_type=MSGRAPH_TEAMS_CHANNEL_SCOPE_TYPE),
        _source(parameters={"window": "forbidden"}),
    ],
)
async def test_invalid_scope_rejected(source: KnowledgeSourceRef) -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(integration=_integration(fake), source=source, cursor=None, limit=50)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert fake.delta_calls == []
    assert fake.snapshot_calls == []


@pytest.mark.parametrize(
    "remote_scope_id",
    [
        "not-base64!!!",
        _encode_canonical_payload({"schema_version": "wrong.v1"}),
        _encode_canonical_payload(
            {
                "schema_version": "msgraph.calendar.scope.v1",
                "mailbox_user_id": _MAILBOX_USER_ID,
                "calendar_remote_id": _CALENDAR_ID,
                "is_default_calendar": True,
                "sync_strategy": "snapshot",
                "window_start_at": _WINDOW_START.isoformat(),
                "window_end_at": _WINDOW_END.isoformat(),
            }
        ),
    ],
)
async def test_malformed_scope_payload_rejected(remote_scope_id: str) -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(remote_scope_id=remote_scope_id),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert fake.delta_calls == []


async def test_wrong_integration_object_rejected() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(integration=object(), source=_source(), cursor=None, limit=50)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


@pytest.mark.parametrize("limit", [1, 50, 200, 1000])
async def test_limit_accepted_and_provider_clamped(limit: int) -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=limit)
    assert fake.delta_calls[0]["limit"] == min(limit, 200)


@pytest.mark.parametrize("limit", [0, -1, 1001, "50", 50.0])
async def test_invalid_limit_rejected(limit: object) -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake), source=_source(), cursor=None, limit=limit  # type: ignore[arg-type]
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert fake.delta_calls == []


# --- 4. Primary delta ---


async def test_primary_delta_first_page_uses_no_continuation() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_NEXT_URL,
            )
        ]
    )
    page = await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=50)
    assert fake.delta_calls[0]["continuation"] is None
    assert page.has_more is True
    assert page.next_cursor is not None
    assert page.proposed_checkpoint == page.next_cursor


async def test_primary_delta_checkpoint_semantics() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    page = await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=50)
    assert page.has_more is False
    assert page.next_cursor is None
    assert page.proposed_checkpoint is not None
    assert _SECRET_DELTA not in page.proposed_checkpoint.value


async def test_primary_delta_continuation_round_trip() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_NEXT_URL,
            ),
            _delta_page(
                items=(_active_event(remote_id=_OTHER_EVENT_ID),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            ),
        ]
    )
    first = await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=50)
    second = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=first.next_cursor,
        limit=50,
    )
    assert fake.delta_calls[1]["continuation"].url == _NEXT_URL
    assert second.has_more is False
    assert second.next_cursor is None


async def test_primary_delta_incremental_from_delta_cursor() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    cursor = _encode_cursor(
        {
            "schema_version": MSGRAPH_CALENDAR_CURSOR_VERSION,
            "mailbox_user_id": _MAILBOX_USER_ID,
            "calendar_remote_id": _CALENDAR_ID,
            "sync_strategy": "primary_delta",
            **_cursor_window_fields(),
            "phase": "delta",
            "continuation_url": _DELTA_URL,
        }
    )
    await adapter.read_page(integration=_integration(fake), source=_source(), cursor=cursor, limit=50)
    continuation = fake.delta_calls[0]["continuation"]
    assert continuation is not None
    assert continuation.kind == MsGraphKnowledgeContinuationKind.DELTA
    assert continuation.url == _DELTA_URL


async def test_active_event_maps_to_upsert_and_removed_to_deleted_tombstone() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(), _removed_event()),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    page = await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=50)
    assert page.changes[0].kind is KnowledgeChangeKind.UPSERT
    assert page.changes[0].descriptor is not None
    assert page.changes[0].descriptor.metadata["removal_semantics"] == _REMOVAL_SEMANTICS
    assert page.changes[1].kind is KnowledgeChangeKind.DELETED
    assert page.changes[1].descriptor is None


async def test_removal_semantics_is_window_scoped_not_global_deletion() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    page = await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=50)
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.metadata["removal_semantics"] == _REMOVAL_SEMANTICS
    assert "global" not in json.dumps(descriptor.metadata)
    assert "permanent" not in json.dumps(descriptor.metadata)
    assert "mailbox" not in json.dumps(descriptor.metadata)


# --- 5. Snapshot ---


async def test_snapshot_first_page_uses_no_continuation() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        snapshot_pages=[
            _snapshot_page(items=(_active_event(calendar_remote_id=_OTHER_CALENDAR_ID),), continuation_url=_SNAPSHOT_NEXT_URL)
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake), source=_snapshot_source(), cursor=None, limit=50
    )
    assert fake.snapshot_calls[0]["continuation"] is None
    assert page.has_more is True
    assert page.next_cursor is not None


async def test_snapshot_complete_checkpoint() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        snapshot_pages=[
            _snapshot_page(items=(_active_event(calendar_remote_id=_OTHER_CALENDAR_ID),))
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake), source=_snapshot_source(), cursor=None, limit=50
    )
    assert page.has_more is False
    assert page.next_cursor is None
    assert page.proposed_checkpoint is not None
    padding = "=" * (-len(page.proposed_checkpoint.value) % 4)
    decoded = _MsGraphCalendarCursor.model_validate(
        json.loads(base64.urlsafe_b64decode(page.proposed_checkpoint.value + padding).decode())
    )
    assert decoded.phase == "complete"
    assert decoded.sync_strategy == "snapshot"


async def test_snapshot_continuation_round_trip() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        snapshot_pages=[
            _snapshot_page(
                items=(_active_event(calendar_remote_id=_OTHER_CALENDAR_ID),),
                continuation_url=_SNAPSHOT_NEXT_URL,
            ),
            _snapshot_page(items=(_active_event(calendar_remote_id=_OTHER_CALENDAR_ID, remote_id=_OTHER_EVENT_ID),)),
        ]
    )
    first = await adapter.read_page(
        integration=_integration(fake), source=_snapshot_source(), cursor=None, limit=50
    )
    second = await adapter.read_page(
        integration=_integration(fake),
        source=_snapshot_source(),
        cursor=first.next_cursor,
        limit=50,
    )
    assert fake.snapshot_calls[1]["continuation"].url == _SNAPSHOT_NEXT_URL
    assert second.has_more is False


async def test_completed_snapshot_cursor_rejected() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        snapshot_pages=[
            _snapshot_page(items=(_active_event(calendar_remote_id=_OTHER_CALENDAR_ID),))
        ]
    )
    complete = _encode_cursor(
        {
            "schema_version": MSGRAPH_CALENDAR_CURSOR_VERSION,
            "mailbox_user_id": _MAILBOX_USER_ID,
            "calendar_remote_id": _OTHER_CALENDAR_ID,
            "sync_strategy": "snapshot",
            **_cursor_window_fields(),
            "phase": "complete",
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_snapshot_source(),
            cursor=complete,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert fake.snapshot_calls == []


async def test_snapshot_rejects_removed_items() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        snapshot_pages=[
            MsGraphCalendarEventSnapshotPage.model_construct(
                mailbox_user_id=_MAILBOX_USER_ID,
                calendar_remote_id=_OTHER_CALENDAR_ID,
                window=_window(),
                items=(_removed_event(calendar_remote_id=_OTHER_CALENDAR_ID),),
                continuation=None,
            )
        ]
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake), source=_snapshot_source(), cursor=None, limit=50
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    _assert_error_hides_secrets(exc_info.value)


# --- 6. Cursor isolation ---


async def test_cursor_binding_rejects_mismatched_mailbox() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_NEXT_URL,
            )
        ]
    )
    cursor = _encode_cursor(
        {
            "schema_version": MSGRAPH_CALENDAR_CURSOR_VERSION,
            "mailbox_user_id": _OTHER_MAILBOX_USER_ID,
            "calendar_remote_id": _CALENDAR_ID,
            "sync_strategy": "primary_delta",
            **_cursor_window_fields(),
            "phase": "next_page",
            "continuation_url": _NEXT_URL,
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(integration=_integration(fake), source=_source(), cursor=cursor, limit=50)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert fake.delta_calls == []


async def test_cursor_binding_rejects_mismatched_calendar() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_NEXT_URL,
            )
        ]
    )
    cursor = _encode_cursor(
        {
            "schema_version": MSGRAPH_CALENDAR_CURSOR_VERSION,
            "mailbox_user_id": _MAILBOX_USER_ID,
            "calendar_remote_id": _OTHER_CALENDAR_ID,
            "sync_strategy": "primary_delta",
            **_cursor_window_fields(),
            "phase": "next_page",
            "continuation_url": _NEXT_URL,
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(integration=_integration(fake), source=_source(), cursor=cursor, limit=50)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR


async def test_cursor_binding_rejects_mismatched_window() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_NEXT_URL,
            )
        ]
    )
    mismatched_window = MsGraphCalendarViewWindow(
        start_at=datetime(2024, 6, 2, 0, 0, tzinfo=timezone.utc),
        end_at=_WINDOW_END,
    )
    cursor = _encode_cursor(
        {
            "schema_version": MSGRAPH_CALENDAR_CURSOR_VERSION,
            "mailbox_user_id": _MAILBOX_USER_ID,
            "calendar_remote_id": _CALENDAR_ID,
            "sync_strategy": "primary_delta",
            **_cursor_window_fields(mismatched_window),
            "phase": "next_page",
            "continuation_url": _NEXT_URL,
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(integration=_integration(fake), source=_source(), cursor=cursor, limit=50)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR


async def test_cursor_binding_rejects_mismatched_sync_strategy() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_NEXT_URL,
            )
        ]
    )
    cursor = _encode_cursor(
        {
            "schema_version": MSGRAPH_CALENDAR_CURSOR_VERSION,
            "mailbox_user_id": _MAILBOX_USER_ID,
            "calendar_remote_id": _CALENDAR_ID,
            "sync_strategy": "snapshot",
            **_cursor_window_fields(),
            "phase": "next_page",
            "continuation_url": _NEXT_URL,
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(integration=_integration(fake), source=_source(), cursor=cursor, limit=50)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR


@pytest.mark.parametrize(
    "cursor",
    [
        KnowledgeCursor(value="not-base64", version=MSGRAPH_CALENDAR_CURSOR_VERSION),
        KnowledgeCursor(value="!!!", version=MSGRAPH_CALENDAR_CURSOR_VERSION),
        KnowledgeCursor(value=_encode_cursor({"bad": 1}).value, version="other.v1"),
        _encode_cursor(
            {
                "schema_version": MSGRAPH_CALENDAR_CURSOR_VERSION,
                "mailbox_user_id": _MAILBOX_USER_ID,
                "calendar_remote_id": _CALENDAR_ID,
                "sync_strategy": "primary_delta",
                **_cursor_window_fields(),
                "phase": "next_page",
                "continuation_url": "",
            }
        ),
    ],
)
async def test_invalid_cursor_rejected(cursor: KnowledgeCursor) -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FakeCalendarCollaborationSuite()),
            source=_source(),
            cursor=cursor,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR


async def test_cursor_url_hidden_from_repr_and_errors() -> None:
    cursor = _MsGraphCalendarCursor(
        schema_version=MSGRAPH_CALENDAR_CURSOR_VERSION,
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=_CALENDAR_ID,
        sync_strategy="primary_delta",
        window_start_at=_WINDOW_START,
        window_end_at=_WINDOW_END,
        phase="next_page",
        continuation_url=_NEXT_URL,
    )
    rendered = repr(cursor)
    assert _SECRET_SKIP not in rendered
    assert _NEXT_URL not in rendered


# --- 7. Descriptor and content ---


async def test_event_descriptor_mapping() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    page = await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=50)
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.item_type == "msgraph_calendar_event"
    assert descriptor.content_mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert descriptor.content_available is True
    assert descriptor.title == "Calendar event"
    assert descriptor.identity.remote_id != _EVENT_ID
    assert set(descriptor.metadata.keys()) == set(_base_metadata().keys())


async def test_fetch_content_structured_record() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        content_by_key={(_EVENT_ID, _CHANGE_KEY): _event_content(body_content="Team sync")}
    )
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_event_descriptor(),
    )
    assert content.mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert content.mime_type == _STRUCTURED_RECORD_MIME
    record = content.structured_record
    assert record["schema"] == _STRUCTURED_RECORD_SCHEMA
    assert record["body"]["content"] == "Team sync"
    assert record["attachments"]["attachment_binary_content_included"] is False
    expected_hash = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert content.content_hash == expected_hash
    assert fake.content_calls[0]["max_chars"] == DEFAULT_CALENDAR_EVENT_CONTENT_MAX_CHARS


async def test_fetch_content_descriptor_mismatch_rejected_before_integration() -> None:
    exc_info = await _fetch_content_invalid_descriptor(_event_descriptor(item_type="wrong_type"))
    _assert_invalid_descriptor_boundary(exc_info)


@pytest.mark.parametrize(
    "item_factory",
    [
        lambda: _event_descriptor(item_type="wrong_type"),
        lambda: _event_descriptor(content_mode=KnowledgeContentMode.BINARY),
        lambda: _event_descriptor(content_available=False),
        lambda: _event_descriptor(mailbox_user_id=_OTHER_MAILBOX_USER_ID),
        lambda: _event_descriptor(calendar_remote_id=_OTHER_CALENDAR_ID),
        lambda: _event_descriptor(provenance_source_kind=MSGRAPH_MAIL_SOURCE_KIND),
        lambda: _event_descriptor().model_copy(
            update={
                "revision": KnowledgeItemRevision.model_construct(
                    version="not-valid-base64!!!",
                    etag=None,
                    updated_at=_LAST_MODIFIED,
                )
            }
        ),
        lambda: _event_descriptor(metadata={"removal_semantics": "deleted"}, metadata_only=True),
    ],
)
async def test_fetch_content_rejects_model_construct_descriptor(item_factory) -> None:
    exc_info = await _fetch_content_invalid_descriptor(item_factory())
    _assert_invalid_descriptor_boundary(exc_info)


async def test_fetch_permissions_unsupported() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_permissions(
            integration=_integration(fake),
            source=_source(),
            item=_event_descriptor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert fake.content_calls == []
    assert fake.delta_calls == []
    assert fake.attachment_calls == []


async def test_fetch_content_event_changed_maps_to_dependency_unavailable() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()

    class _ChangedSuite(_FakeCalendarCollaborationSuite):
        def read_calendar_event_content(self, *, event, max_chars: int):
            raise MsGraphCalendarEventChanged()

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(_ChangedSuite()),
            source=_source(),
            item=_event_descriptor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    _assert_error_hides_secrets(exc_info.value)


async def test_fetch_content_content_too_large_maps_to_configuration_error() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()

    class _TooLargeSuite(_FakeCalendarCollaborationSuite):
        def read_calendar_event_content(self, *, event, max_chars: int):
            raise MsGraphCalendarEventContentTooLarge()

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(_TooLargeSuite()),
            source=_source(),
            item=_event_descriptor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert exc_info.value.retryable is False
    _assert_error_hides_secrets(exc_info.value)


# --- 8. Attachments ---


async def test_has_attachments_false_skips_attachment_call() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        content_by_key={(_EVENT_ID, _CHANGE_KEY): _event_content(has_attachments=False)}
    )
    await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_event_descriptor(),
    )
    assert fake.attachment_calls == []


async def test_has_attachments_true_fetches_attachment_inventory() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        content_by_key={(_EVENT_ID, _CHANGE_KEY): _event_content(has_attachments=True)},
        attachment_pages={
            (_EVENT_ID, _CHANGE_KEY): MsGraphCalendarAttachmentPage(
                mailbox_user_id=_MAILBOX_USER_ID,
                calendar_remote_id=_CALENDAR_ID,
                event_remote_id=_EVENT_ID,
                event_revision=_CHANGE_KEY,
                items=(_valid_attachment(),),
                continuation=None,
            )
        },
    )
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_event_descriptor(metadata={"has_attachments": True}),
    )
    assert fake.attachment_calls
    record = content.structured_record
    assert record["attachments"]["attachment_inventory_included"] is True
    assert record["attachments"]["items"]


async def test_attachment_inventory_over_limit_fails_closed() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        content_by_key={(_EVENT_ID, _CHANGE_KEY): _event_content(has_attachments=True)},
        attachment_pages={
            (_EVENT_ID, _CHANGE_KEY): MsGraphCalendarAttachmentPage(
                mailbox_user_id=_MAILBOX_USER_ID,
                calendar_remote_id=_CALENDAR_ID,
                event_remote_id=_EVENT_ID,
                event_revision=_CHANGE_KEY,
                items=(_valid_attachment(),),
                continuation=MsGraphKnowledgeContinuation(
                    kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                    url=_ATTACHMENTS_NEXT_URL,
                ),
            )
        },
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=_event_descriptor(metadata={"has_attachments": True}),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert exc_info.value.retryable is False
    _assert_error_hides_secrets(exc_info.value)


# --- 9. Security and provider boundaries ---


async def test_forbidden_provider_calls_not_made_during_paging() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=50)
    assert fake.forbidden_calls == []


async def test_read_page_malformed_provider_page_maps_to_invalid_provider_response() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            MsGraphCalendarEventDeltaPage.model_construct(
                mailbox_user_id=_MAILBOX_USER_ID,
                calendar_remote_id=_CALENDAR_ID,
                window=_window(),
                items=(_active_event(mailbox_user_id=_OTHER_MAILBOX_USER_ID),),
                continuation=MsGraphKnowledgeContinuation(
                    kind=MsGraphKnowledgeContinuationKind.DELTA,
                    url=_DELTA_URL,
                ),
            )
        ]
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=50)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    _assert_error_hides_secrets(exc_info.value)


async def test_integration_dependency_error_translated() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()

    class _BrokenSuite(_FakeCalendarCollaborationSuite):
        def read_calendar_events_delta_page_by_reference(self, **kwargs: Any):
            raise IntegrationDependencyError("boom")

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_BrokenSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert "boom" not in str(exc_info.value)


async def test_public_objects_do_not_leak_secrets() -> None:
    adapter = MsGraphCalendarKnowledgeAdapter()
    fake = _FakeCalendarCollaborationSuite(
        delta_pages=[
            _delta_page(
                items=(_active_event(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_NEXT_URL,
            )
        ],
        content_by_key={(_EVENT_ID, _CHANGE_KEY): _event_content()},
    )
    page = await adapter.read_page(
        integration=_integration(fake), source=_source(), cursor=None, limit=50
    )
    blob = json.dumps(page.model_dump(mode="json"))
    for secret in (_SECRET_SKIP, _SECRET_DELTA, _NEXT_URL, "skiptoken", "deltatoken"):
        assert secret not in blob
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_event_descriptor(),
    )
    content_blob = json.dumps(content.model_dump(mode="json"))
    assert _NEXT_URL not in content_blob
    assert "Team sync" not in repr(page)


def test_private_revision_model_rejects_control_characters() -> None:
    with pytest.raises(ValidationError):
        _MsGraphCalendarRevision(
            schema_version="msgraph.calendar.revision.v1",
            change_key="bad\x00key",
        )


def test_production_file_has_no_assertions() -> None:
    adapter_path = (
        Path(__file__).resolve().parents[4]
        / "intergrax/runtime/vendor_knowledge/adapters/ms365_graph_calendar.py"
    )
    text = adapter_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if " assert " in f" {line} " or stripped.startswith("assert "):
            raise AssertionError(f"assert found in production adapter: {line}")
