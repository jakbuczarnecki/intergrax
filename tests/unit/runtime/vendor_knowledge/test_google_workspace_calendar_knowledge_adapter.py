"""Unit tests for GoogleWorkspaceCalendarKnowledgeAdapter."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceBinaryPayload,
    GoogleWorkspaceBinaryTransport,
    GoogleWorkspaceSourceKind,
    GoogleWorkspaceTransport,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.calendar import (
    GOOGLE_CALENDAR_SOURCE_KIND,
    GoogleCalendarAttendee,
    GoogleCalendarAttendeeResponseStatus,
    GoogleCalendarConferenceData,
    GoogleCalendarConferenceEntryPoint,
    GoogleCalendarConferenceSolution,
    GoogleCalendarConferenceSolutionType,
    GoogleCalendarEvent,
    GoogleCalendarEventDateTime,
    GoogleCalendarEventPage,
    GoogleCalendarEventStatus,
    GoogleCalendarEventType,
    GoogleCalendarPerson,
    GoogleCalendarReminder,
    GoogleCalendarReminderMethod,
    GoogleCalendarReminders,
    GoogleCalendarSyncToken,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
    GoogleWorkspaceErrorKind,
    GoogleWorkspacePageToken,
)
from intergrax.runtime.vendor_knowledge.adapters import (
    GOOGLE_CALENDAR_CURSOR_VERSION,
    GOOGLE_CALENDAR_ITEM_METADATA_VERSION,
    GOOGLE_CALENDAR_SCOPE_TYPE,
    GOOGLE_CALENDAR_STRUCTURED_RECORD_MIME_TYPE,
    GOOGLE_CALENDAR_STRUCTURED_RECORD_SCHEMA,
    GoogleWorkspaceCalendarKnowledgeAdapter,
    register_google_workspace_calendar_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_calendar import (
    _build_structured_record,
    _compute_content_hash,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
    to_source_ref,
)
from intergrax.runtime.vendor_knowledge.contracts import VendorKnowledgeAdapter
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeChangeKind,
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgeItemIdentity,
    KnowledgeItemProvenance,
    KnowledgeItemRevision,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_CALENDAR_ID = "team@group.calendar.google.com"
_CONNECTION_REF = "connection-google-calendar"
_EVENT_ID = "event-1"
_SYNC_TOKEN = "sync-1"
_PAGE_TOKEN = "page-1"


def _event(
    *,
    event_id: str = _EVENT_ID,
    status: GoogleCalendarEventStatus = GoogleCalendarEventStatus.CONFIRMED,
    summary: str | None = "Planning",
    sequence: int | None = 2,
    all_day: bool = False,
    recurring: bool = True,
    attendee_response: GoogleCalendarAttendeeResponseStatus = (
        GoogleCalendarAttendeeResponseStatus.ACCEPTED
    ),
    conference: bool = True,
) -> GoogleCalendarEvent:
    if status is GoogleCalendarEventStatus.CANCELLED:
        return GoogleCalendarEvent(id=event_id, status=status)
    start = (
        GoogleCalendarEventDateTime(date="2026-01-01")
        if all_day
        else GoogleCalendarEventDateTime(
            date_time="2026-01-01T10:00:00Z",
            time_zone="Europe/Warsaw",
        )
    )
    end = (
        GoogleCalendarEventDateTime(date="2026-01-02")
        if all_day
        else GoogleCalendarEventDateTime(
            date_time="2026-01-01T11:00:00Z",
            time_zone="Europe/Warsaw",
        )
    )
    return GoogleCalendarEvent(
        id=event_id,
        i_cal_uid=f"ical-{event_id}",
        etag='"etag-1"',
        status=status,
        event_type=GoogleCalendarEventType.FROM_GMAIL,
        summary=summary,
        description="Description",
        location="Room 1",
        created="2026-01-01T09:00:00Z",
        updated="2026-01-01T09:30:00Z",
        visibility="private",
        transparency="opaque",
        sequence=sequence,
        start=start,
        end=end,
        recurrence=("RRULE:FREQ=WEEKLY",) if recurring else (),
        recurring_event_id="master-1" if recurring else None,
        original_start_time=start if recurring else None,
        creator=GoogleCalendarPerson(email="creator@example.com"),
        organizer=GoogleCalendarPerson(email="organizer@example.com"),
        attendees=(
            GoogleCalendarAttendee(
                email="attendee@example.com",
                response_status=attendee_response,
            ),
        ),
        attendees_omitted=False,
        guests_can_invite_others=True,
        guests_can_modify=False,
        guests_can_see_other_guests=True,
        private_copy=False,
        locked=False,
        conference_data=(
            GoogleCalendarConferenceData(
                conference_id="conference-1",
                conference_solution=GoogleCalendarConferenceSolution(
                    type=GoogleCalendarConferenceSolutionType.HANGOUTS_MEET,
                    name="Google Meet",
                ),
                entry_points=(
                    GoogleCalendarConferenceEntryPoint(
                        entry_point_type="video",
                        uri="https://meet.google.com/abc",
                        passcode="secret",
                    ),
                ),
            )
            if conference
            else None
        ),
        reminders=GoogleCalendarReminders(
            use_default=False,
            overrides=(
                GoogleCalendarReminder(
                    method=GoogleCalendarReminderMethod.EMAIL,
                    minutes=10,
                ),
            ),
        ),
    )


def _page(
    *,
    events: tuple[GoogleCalendarEvent, ...] = (_event(),),
    next_page_token: str | None = None,
    next_sync_token: str | None = _SYNC_TOKEN,
) -> GoogleCalendarEventPage:
    return GoogleCalendarEventPage(
        calendar_id=_CALENDAR_ID,
        events=events,
        next_page_token=(
            GoogleWorkspacePageToken(value=next_page_token)
            if next_page_token is not None
            else None
        ),
        next_sync_token=(
            GoogleCalendarSyncToken(value=next_sync_token)
            if next_sync_token is not None
            else None
        ),
    )


@dataclass
class _FakeIntegration:
    pages: list[GoogleCalendarEventPage] = field(default_factory=list)
    events: list[GoogleCalendarEvent] = field(default_factory=list)
    page_calls: list[dict[str, Any]] = field(default_factory=list)
    event_calls: list[dict[str, Any]] = field(default_factory=list)
    exception: Exception | None = None

    def list_calendar_events_page(self, **kwargs: Any) -> GoogleCalendarEventPage:
        self.page_calls.append(kwargs)
        if self.exception is not None:
            raise self.exception
        return self.pages.pop(0)

    def read_calendar_event(self, **kwargs: Any) -> GoogleCalendarEvent:
        self.event_calls.append(kwargs)
        if self.exception is not None:
            raise self.exception
        return self.events.pop(0)


class _StubTransport(GoogleWorkspaceBinaryTransport):
    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        raise NotImplementedError

    def get_binary(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        max_bytes: int,
        range_limited: bool,
    ) -> GoogleWorkspaceBinaryPayload:
        raise NotImplementedError


class _StubFamily:
    transport: GoogleWorkspaceTransport = _StubTransport()


class _BoundIntegration(GoogleWorkspaceCollaborationSuiteIntegration):
    _fake: _FakeIntegration = PrivateAttr()

    @classmethod
    def from_fake(cls, fake: _FakeIntegration) -> _BoundIntegration:
        bound = cls.from_client(_StubFamily(), enabled=True)
        bound._fake = fake
        return bound

    def list_calendar_events_page(self, **kwargs: Any) -> GoogleCalendarEventPage:
        return self._fake.list_calendar_events_page(**kwargs)

    def read_calendar_event(self, **kwargs: Any) -> GoogleCalendarEvent:
        return self._fake.read_calendar_event(**kwargs)


def _integration(fake: _FakeIntegration) -> GoogleWorkspaceCollaborationSuiteIntegration:
    return _BoundIntegration.from_fake(fake)


def _source(
    *,
    remote_scope_id: str = _CALENDAR_ID,
    remote_scope_type: str = GOOGLE_CALENDAR_SCOPE_TYPE,
    parameters: dict[str, Any] | None = None,
    provider_id: str = GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    integration_kind: IntegrationCategory = IntegrationCategory.COLLABORATION_SUITE,
    source_kind: str = GOOGLE_CALENDAR_SOURCE_KIND,
    connection_ref: str | None = _CONNECTION_REF,
    safe_display_name: str = "Team calendar",
) -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        connection_ref=connection_ref,
        scope=KnowledgeSourceScope(
            remote_scope_id=remote_scope_id,
            remote_scope_type=remote_scope_type,
            safe_display_name=safe_display_name,
            parameters=parameters or {},
        ),
    )


def _binding() -> KnowledgeSourceBinding:
    return KnowledgeSourceBinding(
        binding_id="binding-calendar-1",
        tenant_id="tenant-1",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_CALENDAR_SOURCE_KIND,
        connection_ref=_CONNECTION_REF,
        safe_display_name="Team calendar",
        scope=KnowledgeSourceScope(
            remote_scope_id=_CALENDAR_ID,
            remote_scope_type=GOOGLE_CALENDAR_SCOPE_TYPE,
            safe_display_name="Team calendar",
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
    )


def _decode_cursor(cursor: KnowledgeCursor) -> dict[str, Any]:
    raw = base64.urlsafe_b64decode(cursor.value + "=" * (-len(cursor.value) % 4))
    return json.loads(raw.decode("utf-8"))


async def _read_page(
    adapter: GoogleWorkspaceCalendarKnowledgeAdapter,
    fake: _FakeIntegration,
    *,
    source: KnowledgeSourceRef | None = None,
    cursor: KnowledgeCursor | None = None,
    limit: int = 50,
):
    return await adapter.read_page(
        integration=_integration(fake),
        source=source or _source(),
        cursor=cursor,
        limit=limit,
    )


async def test_identity_capabilities_and_explicit_registration() -> None:
    adapter = GoogleWorkspaceCalendarKnowledgeAdapter()
    assert isinstance(adapter, VendorKnowledgeAdapter)
    assert adapter.provider_id == GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID
    assert adapter.integration_kind is IntegrationCategory.COLLABORATION_SUITE
    assert adapter.source_kind == GOOGLE_CALENDAR_SOURCE_KIND
    assert adapter.capabilities.model_dump() == {
        "full_inventory": True,
        "incremental_changes": True,
        "content_fetch": True,
        "binary_content": False,
        "rich_text_content": False,
        "structured_content": True,
        "permissions": False,
        "tombstones": True,
        "remote_versions": True,
        "reconciliation": True,
    }
    registry = KnowledgeAdapterRegistry()
    assert register_google_workspace_calendar_knowledge_adapter(registry) is not None
    assert registry.resolve(source=_source()) is not None


async def test_inspection_and_binding_facade_preserve_exact_source_without_calls() -> None:
    fake = _FakeIntegration()
    adapter = GoogleWorkspaceCalendarKnowledgeAdapter()
    source = to_source_ref(_binding())
    info = await adapter.inspect_scope(
        integration=_integration(fake),
        source=source,
    )
    assert info.source == source
    assert info.source.connection_ref == _CONNECTION_REF
    assert info.safe_display_name == "Team calendar"
    assert fake.page_calls == []
    resolver = type(
        "_Resolver",
        (),
        {"resolve": lambda self, *, source: _integration(fake)},
    )()
    registry = KnowledgeAdapterRegistry()
    register_google_workspace_calendar_knowledge_adapter(registry)
    result = await VendorKnowledgeFacadeService(
        tenant_id="tenant-1",
        resolver=resolver,
        adapter_registry=registry,
    ).inspect_source(source=source)
    assert result.source == source
    assert fake.page_calls == []


@pytest.mark.parametrize(
    "source",
    [
        _source(provider_id="other"),
        _source(integration_kind=IntegrationCategory.ISSUE_TRACKER),
        _source(source_kind="drive"),
        _source(remote_scope_type="other"),
        _source(parameters={"unexpected": "value"}),
        _source(remote_scope_id="team/main"),
        _source(remote_scope_id="team\\main"),
        _source(remote_scope_id="team\x00main"),
        _source(remote_scope_id="x" * 1025),
    ],
)
async def test_invalid_scope_is_safe_and_makes_no_call(source: KnowledgeSourceRef) -> None:
    fake = _FakeIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await _read_page(GoogleWorkspaceCalendarKnowledgeAdapter(), fake, source=source)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert exc_info.value.retryable is False
    assert exc_info.value.__cause__ is None
    assert fake.page_calls == []
    assert _CALENDAR_ID not in repr(exc_info.value)
    assert _CONNECTION_REF not in repr(exc_info.value)


async def test_full_inventory_cursor_transitions() -> None:
    adapter = GoogleWorkspaceCalendarKnowledgeAdapter()
    fake = _FakeIntegration(pages=[_page(next_page_token=_PAGE_TOKEN, next_sync_token=None)])
    first = await _read_page(adapter, fake, limit=123)
    assert first.has_more is True
    assert first.proposed_checkpoint is None
    assert _decode_cursor(first.next_cursor)["phase"] == "inventory"
    assert fake.page_calls == [
        {
            "calendar_id": _CALENDAR_ID,
            "page_token": None,
            "sync_token": None,
            "max_results": 123,
        }
    ]

    fake = _FakeIntegration(pages=[_page(next_sync_token="sync-terminal")])
    terminal = await _read_page(adapter, fake)
    assert terminal.next_cursor is None
    assert terminal.has_more is False
    checkpoint = _decode_cursor(terminal.proposed_checkpoint)
    assert checkpoint["phase"] == "changes"
    assert checkpoint["sync_token"] == "sync-terminal"
    assert checkpoint["page_token"] is None


async def test_incremental_pagination_preserves_original_sync_token() -> None:
    adapter = GoogleWorkspaceCalendarKnowledgeAdapter()
    fake = _FakeIntegration(pages=[_page(next_page_token="next-page", next_sync_token=None)])
    cursor = KnowledgeCursor(
        value=base64.urlsafe_b64encode(
            json.dumps(
                {
                    "schema_version": GOOGLE_CALENDAR_CURSOR_VERSION,
                    "scope_fingerprint": hashlib.sha256(
                        f"google_workspace\x00calendar\x00{_CALENDAR_ID}".encode()
                    ).hexdigest(),
                    "phase": "changes",
                    "page_token": None,
                    "sync_token": "original-sync",
                },
                sort_keys=True,
                separators=(",", ":"),
                ).encode()
        ).decode().rstrip("="),
        version=GOOGLE_CALENDAR_CURSOR_VERSION,
    )
    page = await _read_page(adapter, fake, cursor=cursor)
    assert fake.page_calls[0]["sync_token"].value == "original-sync"
    assert fake.page_calls[0]["page_token"] is None
    continuation = _decode_cursor(page.next_cursor)
    assert continuation["sync_token"] == "original-sync"
    assert continuation["page_token"] == "next-page"
    assert page.proposed_checkpoint is None

    fake = _FakeIntegration(pages=[_page(next_sync_token="new-sync")])
    page = await _read_page(adapter, fake, cursor=page.next_cursor)
    assert fake.page_calls[0]["sync_token"].value == "original-sync"
    assert fake.page_calls[0]["page_token"].value == "next-page"
    assert _decode_cursor(page.proposed_checkpoint)["sync_token"] == "new-sync"


async def test_event_mapping_order_and_nested_structured_descriptor() -> None:
    events = (
        _event(event_id="confirmed"),
        _event(
            event_id="tentative-all-day",
            status=GoogleCalendarEventStatus.TENTATIVE,
            all_day=True,
            recurring=False,
        ),
        _event(event_id="recurring"),
        _event(
            event_id="cancelled",
            status=GoogleCalendarEventStatus.CANCELLED,
        ),
    )
    page = await _read_page(
        GoogleWorkspaceCalendarKnowledgeAdapter(),
        _FakeIntegration(pages=[_page(events=events)]),
    )
    assert [change.remote_id for change in page.changes] == [
        "confirmed",
        "tentative-all-day",
        "recurring",
        "cancelled",
    ]
    assert [change.kind for change in page.changes] == [
        KnowledgeChangeKind.UPSERT,
        KnowledgeChangeKind.UPSERT,
        KnowledgeChangeKind.UPSERT,
        KnowledgeChangeKind.DELETED,
    ]
    assert page.changes[-1].descriptor is None
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.identity.remote_id == "confirmed"
    assert descriptor.identity.parent_remote_id is None
    assert descriptor.identity.logical_key is None
    assert descriptor.title == "Planning"
    assert descriptor.item_type == "google_workspace_calendar_event"
    assert descriptor.content_mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert descriptor.content_available is True
    assert descriptor.revision.version == "2"
    assert descriptor.revision.etag == '"etag-1"'
    assert descriptor.revision.updated_at == datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
    assert descriptor.provenance.web_url is None
    assert descriptor.metadata["schema_version"] == GOOGLE_CALENDAR_ITEM_METADATA_VERSION
    assert descriptor.metadata["structured_record_schema"] == GOOGLE_CALENDAR_STRUCTURED_RECORD_SCHEMA
    assert descriptor.metadata["status"] == "confirmed"
    assert descriptor.metadata["event_type"] == "fromGmail"
    assert descriptor.metadata["recurring_event"] is True
    assert descriptor.metadata["all_day"] is False
    assert "ical" not in descriptor.identity.remote_id


async def test_structured_record_and_content_hash_fence() -> None:
    event = _event()
    record = _build_structured_record(_CALENDAR_ID, event)
    assert record["schema_version"] == GOOGLE_CALENDAR_STRUCTURED_RECORD_SCHEMA
    assert record["calendar_id"] == _CALENDAR_ID
    nested = record["event"]
    assert nested["start"]["date_time"] == "2026-01-01T10:00:00Z"
    assert nested["end"]["date_time"] == "2026-01-01T11:00:00Z"
    assert nested["recurrence"] == ["RRULE:FREQ=WEEKLY"]
    assert nested["attendees"][0]["response_status"] == "accepted"
    assert nested["conference_data"]["entry_points"][0]["passcode"] == "secret"
    assert nested["reminders"]["overrides"][0]["minutes"] == 10
    reordered = {key: record[key] for key in reversed(tuple(record))}
    assert _compute_content_hash(record) == _compute_content_hash(reordered)

    adapter = GoogleWorkspaceCalendarKnowledgeAdapter()
    fake = _FakeIntegration(pages=[_page()], events=[event])
    page = await _read_page(adapter, fake)
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=descriptor,
    )
    assert content.mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert content.mime_type == GOOGLE_CALENDAR_STRUCTURED_RECORD_MIME_TYPE
    assert content.content_hash == descriptor.revision.content_hash
    assert content.structured_record == record
    assert fake.event_calls == [{"calendar_id": _CALENDAR_ID, "event_id": _EVENT_ID}]

    changed = _event(attendee_response=GoogleCalendarAttendeeResponseStatus.TENTATIVE)
    fake = _FakeIntegration(events=[changed])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=descriptor,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert exc_info.value.__cause__ is None
    assert exc_info.value.safe_message == "Google Workspace Calendar event content changed since descriptor creation"

    fake = _FakeIntegration(events=[_event(status=GoogleCalendarEventStatus.CANCELLED)])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=descriptor,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True


async def test_permissions_and_error_mapping_make_no_provider_call() -> None:
    adapter = GoogleWorkspaceCalendarKnowledgeAdapter()
    fake = _FakeIntegration(pages=[_page()])
    page = await _read_page(adapter, fake)
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_permissions(
            integration=_integration(fake),
            source=_source(),
            item=descriptor,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert exc_info.value.retryable is False
    assert fake.event_calls == []

    for kind, code, retryable in (
        (
            GoogleWorkspaceErrorKind.AUTHENTICATION,
            VendorKnowledgeErrorCode.AUTHENTICATION_FAILED,
            False,
        ),
        (
            GoogleWorkspaceErrorKind.AUTHORIZATION,
            VendorKnowledgeErrorCode.AUTHORIZATION_DENIED,
            False,
        ),
        (
            GoogleWorkspaceErrorKind.NOT_FOUND,
            VendorKnowledgeErrorCode.REMOTE_ITEM_NOT_FOUND,
            False,
        ),
        (GoogleWorkspaceErrorKind.RATE_LIMITED, VendorKnowledgeErrorCode.RATE_LIMITED, True),
        (
            GoogleWorkspaceErrorKind.TEMPORARY,
            VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
            True,
        ),
        (
            GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            False,
        ),
        (
            GoogleWorkspaceErrorKind.UNEXPECTED_REDIRECT,
            VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            False,
        ),
        (
            GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE,
            VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
            False,
        ),
        (
            GoogleWorkspaceErrorKind.INVALID_REQUEST,
            VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
            False,
        ),
    ):
        api_error = GoogleWorkspaceApiError(
            kind=kind,
            status_code=400,
            retry_after_seconds=None,
            safe_reason="private-provider-detail",
            attempts=1,
        )
        with pytest.raises(VendorKnowledgeError) as mapped:
            await _read_page(
                adapter,
                _FakeIntegration(exception=api_error),
            )
        assert mapped.value.code is code
        assert mapped.value.retryable is retryable
        assert mapped.value.__cause__ is None
        assert "private-provider-detail" not in repr(mapped.value)

    with pytest.raises(VendorKnowledgeError) as mapped:
        await _read_page(adapter, _FakeIntegration(exception=RuntimeError("private")))
    assert mapped.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert mapped.value.retryable is True
    assert "private" not in repr(mapped.value)


async def test_cursor_rejects_wrong_scope_version_padding_and_invalid_phase() -> None:
    adapter = GoogleWorkspaceCalendarKnowledgeAdapter()
    fake = _FakeIntegration()
    valid_payload = {
        "schema_version": GOOGLE_CALENDAR_CURSOR_VERSION,
        "scope_fingerprint": hashlib.sha256(
            f"google_workspace\x00calendar\x00{_CALENDAR_ID}".encode()
        ).hexdigest(),
        "phase": "inventory",
        "page_token": "secret-page",
        "sync_token": None,
    }

    def cursor(payload: dict[str, Any], *, version: str = GOOGLE_CALENDAR_CURSOR_VERSION) -> KnowledgeCursor:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return KnowledgeCursor(
            value=base64.urlsafe_b64encode(raw).decode().rstrip("="),
            version=version,
        )

    invalid = [
        cursor({**valid_payload, "scope_fingerprint": "0" * 64}),
        cursor(valid_payload, version="wrong"),
        KnowledgeCursor(
            value=cursor(valid_payload).value + "=",
            version=GOOGLE_CALENDAR_CURSOR_VERSION,
        ),
        cursor({**valid_payload, "phase": "inventory", "page_token": None}),
        cursor({**valid_payload, "phase": "changes", "sync_token": None}),
    ]
    for item in invalid:
        with pytest.raises(VendorKnowledgeError) as exc_info:
            await _read_page(adapter, fake, cursor=item)
        assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
        assert fake.page_calls == []
    assert _CALENDAR_ID not in repr(cursor(valid_payload))
    assert "secret-page" not in repr(cursor(valid_payload))


async def test_descriptor_mutations_are_rejected_before_single_event_call() -> None:
    adapter = GoogleWorkspaceCalendarKnowledgeAdapter()
    fake = _FakeIntegration(pages=[_page()])
    page = await _read_page(adapter, fake)
    descriptor = page.changes[0].descriptor
    assert descriptor is not None

    mutations = [
        lambda d: KnowledgeItemIdentity.model_construct(
            remote_id=d.identity.remote_id,
            parent_remote_id="parent",
            logical_key=None,
        ),
        lambda d: KnowledgeItemRevision.model_construct(
            version=d.revision.version,
            etag=d.revision.etag,
            content_hash="bad",
            acl_hash=None,
            updated_at=d.revision.updated_at,
        ),
        lambda d: KnowledgeItemProvenance.model_construct(
            provider_id=d.provenance.provider_id,
            source_kind="drive",
            remote_id=d.provenance.remote_id,
            web_url=None,
            safe_locator=None,
        ),
    ]
    for mutation in mutations:
        mutated = descriptor.model_copy(
            update={
                "identity": mutation(descriptor),
            }
        )
        if mutation is mutations[1]:
            mutated = descriptor.model_copy(
                update={"revision": mutation(descriptor)}
            )
        elif mutation is mutations[2]:
            mutated = descriptor.model_copy(
                update={"provenance": mutation(descriptor)}
            )
        fake.event_calls.clear()
        with pytest.raises(VendorKnowledgeError) as exc_info:
            await adapter.fetch_content(
                integration=_integration(fake),
                source=_source(),
                item=mutated,
            )
        assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
        assert exc_info.value.retryable is False
        assert fake.event_calls == []


async def test_wrong_integration_and_malformed_page_fail_closed() -> None:
    adapter = GoogleWorkspaceCalendarKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=object(),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE

    fake = _FakeIntegration(
        pages=[
            GoogleCalendarEventPage.model_construct(
                calendar_id="other-calendar",
                events=(_event(),),
                next_sync_token=None,
                next_page_token=None,
            )
        ]
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await _read_page(adapter, fake)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert _CALENDAR_ID not in repr(exc_info.value)
