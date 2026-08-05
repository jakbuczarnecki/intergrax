"""End-to-end durable Google Workspace Calendar synchronization proof."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import pytest

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.base import IntegrationCategory
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
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
    GoogleWorkspaceErrorKind,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.calendar import (
    _GOOGLE_CALENDAR_EVENT_FIELDS,
    _GOOGLE_CALENDAR_EVENTS_FIELDS,
    GOOGLE_CALENDAR_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.adapters import (
    GOOGLE_CALENDAR_CURSOR_VERSION,
    GOOGLE_CALENDAR_ITEM_METADATA_VERSION,
    GOOGLE_CALENDAR_SCOPE_TYPE,
    GOOGLE_CALENDAR_STRUCTURED_RECORD_MIME_TYPE,
    GOOGLE_CALENDAR_STRUCTURED_RECORD_SCHEMA,
    register_google_workspace_calendar_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
    to_source_ref,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeChangeKind,
    KnowledgeContentMode,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from intergrax.runtime.vendor_knowledge.sync_coordinator import (
    VendorKnowledgeSyncCoordinator,
)
from intergrax.runtime.vendor_knowledge.sync_document_store import (
    DocumentStoreKnowledgeRemoteItemStateRepository,
    DocumentStoreKnowledgeSourceLeaseRepository,
    DocumentStoreKnowledgeSyncCheckpointRepository,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeRemoteItemStatus,
    KnowledgeSyncMode,
    KnowledgeSyncRunStatus,
)
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    IdempotentRecordingSink,
    RecordingBindingService,
    durable_reconciliation_coordinator_kwargs,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_TENANT_ID = "tenant-calendar-proof"
_BINDING_ID = "google-calendar-binding"
_FENCE_BINDING_ID = "google-calendar-fence-binding"
_CALENDAR_ID = "team@group.calendar.google.com"
_CONNECTION_REF = "conn-google-calendar-proof"
_PAGE_2 = "inventory-page-2"
_INCREMENTAL_PAGE_2 = "incremental-page-2"
_SYNC_V1 = "sync-v1"
_SYNC_V2 = "sync-v2"
_SYNC_V3 = "sync-v3"
_PAGE_SIZE = 100
_SCOPE_FINGERPRINT = hashlib.sha256(
    f"google_workspace\x00calendar\x00{_CALENDAR_ID}".encode()
).hexdigest()


def _timed(
    start: str = "2026-01-01T10:00:00Z",
    end: str = "2026-01-01T11:00:00Z",
) -> tuple[dict[str, object], dict[str, object]]:
    return (
        {"dateTime": start, "timeZone": "Europe/Warsaw"},
        {"dateTime": end, "timeZone": "Europe/Warsaw"},
    )


def _event(
    event_id: str,
    *,
    summary: str,
    sequence: int,
    etag: str,
    status: str = "confirmed",
    all_day: bool = False,
    recurring: bool = False,
    attendee_response: str = "accepted",
) -> dict[str, object]:
    if all_day:
        start: dict[str, object] = {"date": "2026-02-10"}
        end: dict[str, object] = {"date": "2026-02-11"}
    else:
        start, end = _timed()
    payload: dict[str, object] = {
        "id": event_id,
        "iCalUID": f"ical-{event_id}",
        "etag": etag,
        "status": status,
        "eventType": "fromGmail",
        "summary": summary,
        "description": f"{summary} description",
        "location": "Room 1",
        "created": "2026-01-01T09:00:00Z",
        "updated": "2026-01-01T09:30:00Z",
        "visibility": "private",
        "transparency": "opaque",
        "sequence": sequence,
        "start": start,
        "end": end,
        "endTimeUnspecified": False,
        "recurrence": ["RRULE:FREQ=WEEKLY"] if recurring else [],
        "creator": {
            "id": "creator-id",
            "email": "creator@example.com",
            "displayName": "Creator",
            "self": True,
        },
        "organizer": {
            "email": "organizer@example.com",
            "displayName": "Organizer",
            "self": False,
        },
        "attendees": [
            {
                "email": "attendee@example.com",
                "displayName": "Attendee",
                "responseStatus": attendee_response,
            }
        ],
        "attendeesOmitted": False,
        "guestsCanInviteOthers": True,
        "guestsCanModify": False,
        "guestsCanSeeOtherGuests": True,
        "privateCopy": False,
        "locked": False,
        "conferenceData": {
            "conferenceId": "conference-1",
            "entryPoints": [
                {
                    "entryPointType": "video",
                    "uri": "https://meet.google.com/abc",
                    "label": "Video",
                    "meetingCode": "abc-defg-hij",
                    "passcode": "secret",
                }
            ],
            "conferenceSolution": {
                "key": {"type": "hangoutsMeet"},
                "name": "Google Meet",
            },
        },
        "reminders": {
            "useDefault": False,
            "overrides": [{"method": "email", "minutes": 10}],
        },
    }
    if recurring:
        payload["recurringEventId"] = "event-gamma"
        payload["originalStartTime"] = start
    return payload


def _cancelled_event(event_id: str) -> dict[str, object]:
    return {"id": event_id, "status": "cancelled"}


def _page(
    events: list[dict[str, object]],
    *,
    next_page_token: str | None = None,
    next_sync_token: str | None = None,
) -> dict[str, object]:
    assert (next_page_token is None) != (next_sync_token is None)
    payload: dict[str, object] = {
        "summary": "Team calendar",
        "description": "Calendar description",
        "updated": "2026-01-01T09:00:00Z",
        "timeZone": "Europe/Warsaw",
        "accessRole": "writer",
        "items": events,
    }
    if next_page_token is not None:
        payload["nextPageToken"] = next_page_token
    if next_sync_token is not None:
        payload["nextSyncToken"] = next_sync_token
    return payload


@dataclass(frozen=True)
class _QueuedResponse:
    kind: str
    relative_path: str
    params: dict[str, object]
    payload: dict[str, object] = field(default_factory=dict)
    error: GoogleWorkspaceApiError | None = None


@dataclass
class _DeterministicCalendarTransport(GoogleWorkspaceBinaryTransport):
    responses: list[_QueuedResponse] = field(default_factory=list)
    calls: list[dict[str, object]] = field(default_factory=list)

    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        actual_params = dict(params or {})
        actual_headers = dict(headers or {})
        self.calls.append(
            {
                "source_kind": source_kind,
                "relative_path": relative_path,
                "params": actual_params,
                "headers": actual_headers,
            }
        )
        if not self.responses:
            raise AssertionError("Calendar transport response queue is empty")
        expected = self.responses.pop(0)
        assert source_kind is GoogleWorkspaceSourceKind.CALENDAR
        assert relative_path == expected.relative_path
        assert actual_params == expected.params
        assert actual_headers == {}
        kind = "collection" if relative_path.endswith("/events") else "event"
        assert kind == expected.kind
        if expected.error is not None:
            raise expected.error
        return copy.deepcopy(expected.payload)

    def get_binary(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        max_bytes: int,
        range_limited: bool,
    ) -> GoogleWorkspaceBinaryPayload:
        raise AssertionError("Calendar proof must not make binary calls")


def _collection_response(
    transport: _DeterministicCalendarTransport,
    payload: dict[str, object],
    *,
    page_token: str | None = None,
    sync_token: str | None = None,
) -> None:
    params: dict[str, object] = {
        "maxResults": _PAGE_SIZE,
        "showDeleted": True,
        "singleEvents": False,
        "fields": _GOOGLE_CALENDAR_EVENTS_FIELDS,
    }
    if page_token is not None:
        params["pageToken"] = page_token
    if sync_token is not None:
        params["syncToken"] = sync_token
    transport.responses.append(
        _QueuedResponse(
            kind="collection",
            relative_path=f"/calendars/{quote(_CALENDAR_ID, safe='')}/events",
            params=params,
            payload=payload,
        )
    )


def _event_response(
    transport: _DeterministicCalendarTransport,
    event_id: str,
    payload: dict[str, object],
) -> None:
    transport.responses.append(
        _QueuedResponse(
            kind="event",
            relative_path=(
                f"/calendars/{quote(_CALENDAR_ID, safe='')}/events/"
                f"{quote(event_id, safe='')}"
            ),
            params={"fields": _GOOGLE_CALENDAR_EVENT_FIELDS},
            payload=payload,
        )
    )


def _collection_error_response(
    transport: _DeterministicCalendarTransport,
    *,
    sync_token: str,
    page_token: str | None = None,
    status_code: int = 410,
) -> None:
    params: dict[str, object] = {
        "maxResults": _PAGE_SIZE,
        "showDeleted": True,
        "singleEvents": False,
        "fields": _GOOGLE_CALENDAR_EVENTS_FIELDS,
        "syncToken": sync_token,
    }
    if page_token is not None:
        params["pageToken"] = page_token
    transport.responses.append(
        _QueuedResponse(
            kind="collection",
            relative_path=f"/calendars/{quote(_CALENDAR_ID, safe='')}/events",
            params=params,
            error=GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
                status_code=status_code,
                retry_after_seconds=None,
                safe_reason="private provider response",
                attempts=1,
            ),
        )
    )


@dataclass
class _Resolver:
    integration: GoogleWorkspaceCollaborationSuiteIntegration
    sources: list[object] = field(default_factory=list)

    def resolve(
        self,
        *,
        source: object,
    ) -> GoogleWorkspaceCollaborationSuiteIntegration:
        self.sources.append(source)
        return self.integration


@dataclass
class _Runtime:
    coordinator: VendorKnowledgeSyncCoordinator
    transport: _DeterministicCalendarTransport
    binding_service: RecordingBindingService
    resolver: _Resolver


class _Client:
    def __init__(self, transport: GoogleWorkspaceTransport) -> None:
        self.transport = transport


def _binding(binding_id: str = _BINDING_ID) -> KnowledgeSourceBinding:
    return KnowledgeSourceBinding(
        binding_id=binding_id,
        tenant_id=_TENANT_ID,
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_CALENDAR_SOURCE_KIND,
        connection_ref=_CONNECTION_REF,
        safe_display_name="Calendar Proof",
        scope=KnowledgeSourceScope(
            remote_scope_id=_CALENDAR_ID,
            remote_scope_type=GOOGLE_CALENDAR_SCOPE_TYPE,
            safe_display_name="Calendar Proof",
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
        broad_scope=False,
    )


def _build_runtime(
    *,
    document_store: InMemoryDocumentStore,
    sink: IdempotentRecordingSink,
    binding: KnowledgeSourceBinding,
    owner_id: str,
) -> _Runtime:
    transport = _DeterministicCalendarTransport()
    integration = GoogleWorkspaceCollaborationSuiteIntegration.from_client(
        _Client(transport), enabled=True
    )
    resolver = _Resolver(integration=integration)
    registry = KnowledgeAdapterRegistry()
    assert register_google_workspace_calendar_knowledge_adapter(registry) is not None
    binding_service = RecordingBindingService(binding=binding)
    facade = VendorKnowledgeFacadeService(
        tenant_id=_TENANT_ID,
        resolver=resolver,
        adapter_registry=registry,
    )
    lease_repo = DocumentStoreKnowledgeSourceLeaseRepository(document_store)
    checkpoint_repo = DocumentStoreKnowledgeSyncCheckpointRepository(document_store)
    state_repo = DocumentStoreKnowledgeRemoteItemStateRepository(document_store)
    coordinator = VendorKnowledgeSyncCoordinator(
        tenant_id=_TENANT_ID,
        owner_id=owner_id,
        binding_service=binding_service,  # type: ignore[arg-type]
        facade=facade,
        lease_repository=lease_repo,
        checkpoint_repository=checkpoint_repo,
        item_state_repository=state_repo,
        sink=sink,
        lease_ttl_seconds=30,
        **durable_reconciliation_coordinator_kwargs(
            state_repository=state_repo,
            document_store=document_store,
        ),
    )
    return _Runtime(coordinator, transport, binding_service, resolver)


def _decode_cursor(value: str) -> dict[str, Any]:
    raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    return json.loads(raw.decode("utf-8"))


def _checkpoint(
    document_store: InMemoryDocumentStore,
    binding_id: str = _BINDING_ID,
):
    return DocumentStoreKnowledgeSyncCheckpointRepository(document_store).get(
        tenant_id=_TENANT_ID,
        binding_id=binding_id,
    )


def _state(
    document_store: InMemoryDocumentStore,
    remote_id: str,
    binding_id: str = _BINDING_ID,
):
    return DocumentStoreKnowledgeRemoteItemStateRepository(document_store).get(
        tenant_id=_TENANT_ID,
        binding_id=binding_id,
        remote_id=remote_id,
    )


def _envelope(batch: Any, remote_id: str) -> Any:
    return next(item for item in batch.envelopes if item.remote_id == remote_id)


def _assert_cursor(
    checkpoint: Any,
    *,
    phase: str,
    page_token: str | None,
    sync_token: str | None,
) -> None:
    assert checkpoint.cursor.version == GOOGLE_CALENDAR_CURSOR_VERSION
    assert _decode_cursor(checkpoint.cursor.value) == {
        "schema_version": GOOGLE_CALENDAR_CURSOR_VERSION,
        "scope_fingerprint": _SCOPE_FINGERPRINT,
        "phase": phase,
        "page_token": page_token,
        "sync_token": sync_token,
    }


def _assert_no_private_values(value: object) -> None:
    blob = repr(value)
    for private in (
        _CONNECTION_REF,
        "attendee@example.com",
        "secret",
        "raw provider response",
        "Authorization",
        "access_token",
    ):
        assert private not in blob


def _assert_calendar_request(
    call: dict[str, object],
    *,
    path: str,
    params: dict[str, object],
) -> None:
    assert call == {
        "source_kind": GoogleWorkspaceSourceKind.CALENDAR,
        "relative_path": path,
        "params": params,
        "headers": {},
    }


async def test_calendar_durable_multipage_restart_update_delete_and_content_fence() -> None:
    binding = _binding()
    source = to_source_ref(binding)
    assert source == to_source_ref(binding)
    assert source.connection_ref == _CONNECTION_REF

    alpha_v1 = _event(
        "event-alpha",
        summary="Alpha",
        sequence=1,
        etag='"alpha-v1"',
    )
    beta = _event(
        "event-beta",
        summary="Beta",
        sequence=1,
        etag='"beta-v1"',
        status="tentative",
        all_day=True,
    )
    gamma = _event(
        "event-gamma",
        summary="Gamma",
        sequence=1,
        etag='"gamma-v1"',
        recurring=True,
    )
    alpha_v2 = _event(
        "event-alpha",
        summary="Alpha updated",
        sequence=2,
        etag='"alpha-v2"',
    )
    delta = _event(
        "event-delta",
        summary="Delta",
        sequence=1,
        etag='"delta-v1"',
    )

    document_store = InMemoryDocumentStore()
    sink = IdempotentRecordingSink()

    # A1: initial inventory page with a durable continuation.
    runtime_a1 = _build_runtime(
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-calendar-a1",
    )
    _collection_response(
        runtime_a1.transport,
        _page([alpha_v1, beta], next_page_token=_PAGE_2),
    )
    _event_response(runtime_a1.transport, "event-alpha", alpha_v1)
    _event_response(runtime_a1.transport, "event-beta", beta)
    result_a1 = await runtime_a1.coordinator.sync_once(binding_id=_BINDING_ID)
    assert result_a1.status is KnowledgeSyncRunStatus.COMPLETED
    assert result_a1.mode is KnowledgeSyncMode.INCREMENTAL
    assert result_a1.has_more is True
    assert result_a1.checkpoint_advanced is True
    assert result_a1.changes_count == 2
    assert result_a1.active_count == 2
    assert result_a1.tombstone_count == 0
    assert len(sink.calls) == 1
    assert len(sink.calls[0].envelopes) == 2
    assert len(sink.durable_delivery_ids) == 1
    checkpoint_a1 = _checkpoint(document_store)
    assert checkpoint_a1 is not None
    _assert_cursor(
        checkpoint_a1,
        phase="inventory",
        page_token=_PAGE_2,
        sync_token=None,
    )
    assert [_state(document_store, item) for item in ("event-alpha", "event-beta")] == [
        _state(document_store, item) for item in ("event-alpha", "event-beta")
    ]
    assert all(
        _state(document_store, item).status is KnowledgeRemoteItemStatus.ACTIVE
        for item in ("event-alpha", "event-beta")
    )
    _assert_calendar_request(
        runtime_a1.transport.calls[0],
        path=f"/calendars/{quote(_CALENDAR_ID, safe='')}/events",
        params={
            "maxResults": _PAGE_SIZE,
            "showDeleted": True,
            "singleEvents": False,
            "fields": _GOOGLE_CALENDAR_EVENTS_FIELDS,
        },
    )
    assert runtime_a1.binding_service.resolve_calls == [_BINDING_ID] * 1
    assert runtime_a1.resolver.sources
    assert all(item == source for item in runtime_a1.resolver.sources)
    assert all(item.connection_ref == _CONNECTION_REF for item in runtime_a1.resolver.sources)

    # Descriptor and structured content proof for the rich timed event.
    alpha_initial = _envelope(sink.calls[0], "event-alpha")
    alpha_descriptor = alpha_initial.descriptor
    assert alpha_descriptor is not None
    assert alpha_descriptor.identity.remote_id == "event-alpha"
    assert alpha_descriptor.identity.logical_key is None
    assert alpha_descriptor.provenance.remote_id == "event-alpha"
    assert alpha_descriptor.provenance.web_url is None
    assert alpha_descriptor.provenance.safe_locator is None
    assert alpha_descriptor.item_type == "google_workspace_calendar_event"
    assert alpha_descriptor.content_mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert alpha_descriptor.content_available is True
    assert alpha_descriptor.revision.version == "1"
    assert alpha_descriptor.revision.etag == '"alpha-v1"'
    assert alpha_descriptor.revision.content_hash is not None
    assert re.fullmatch(r"[a-f0-9]{64}", alpha_descriptor.revision.content_hash)
    assert alpha_descriptor.revision.updated_at.tzinfo is not None
    assert alpha_descriptor.revision.updated_at.utcoffset() is not None
    assert alpha_descriptor.metadata == {
        "schema_version": GOOGLE_CALENDAR_ITEM_METADATA_VERSION,
        "structured_record_schema": GOOGLE_CALENDAR_STRUCTURED_RECORD_SCHEMA,
        "status": "confirmed",
        "event_type": "fromGmail",
        "recurring_event": False,
        "all_day": False,
        "calendar_id_hash": hashlib.sha256(_CALENDAR_ID.encode()).hexdigest(),
    }
    metadata_blob = json.dumps(alpha_descriptor.metadata)
    for forbidden in (
        _CALENDAR_ID,
        "Alpha",
        "description",
        "Room 1",
        "attendee@example.com",
        "https://meet.google.com/abc",
        "secret",
        _CONNECTION_REF,
        _SYNC_V1,
        _PAGE_2,
    ):
        assert forbidden not in metadata_blob
    assert alpha_initial.content is not None
    assert alpha_initial.content.mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert alpha_initial.content.mime_type == GOOGLE_CALENDAR_STRUCTURED_RECORD_MIME_TYPE
    assert alpha_initial.content.content_hash == alpha_descriptor.revision.content_hash
    assert alpha_initial.content.structured_record is not None
    structured = alpha_initial.content.structured_record
    assert structured["schema_version"] == GOOGLE_CALENDAR_STRUCTURED_RECORD_SCHEMA
    assert structured["calendar_id"] == _CALENDAR_ID
    assert structured["event"]["summary"] == "Alpha"
    assert structured["event"]["description"] == "Alpha description"
    assert structured["event"]["location"] == "Room 1"
    assert structured["event"]["start"]["date_time"] == "2026-01-01T10:00:00Z"
    assert structured["event"]["end"]["date_time"] == "2026-01-01T11:00:00Z"
    assert structured["event"]["attendees"][0]["response_status"] == "accepted"
    assert structured["event"]["conference_data"]["entry_points"][0]["uri"] == (
        "https://meet.google.com/abc"
    )
    assert structured["event"]["reminders"]["overrides"][0]["minutes"] == 10
    assert structured["event"]["visibility"] == "private"
    assert structured["event"]["transparency"] == "opaque"

    beta_initial = _envelope(sink.calls[0], "event-beta")
    assert beta_initial.content is not None
    assert beta_initial.content.structured_record is not None
    assert beta_initial.content.structured_record["event"]["start"]["date"] == "2026-02-10"
    assert beta_initial.content.structured_record["event"]["end"]["date"] == "2026-02-11"

    # A2: fresh runtime resumes the inventory page, without repeating A1.
    runtime_a2 = _build_runtime(
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-calendar-a2",
    )
    _collection_response(
        runtime_a2.transport,
        _page([gamma], next_sync_token=_SYNC_V1),
        page_token=_PAGE_2,
    )
    _event_response(runtime_a2.transport, "event-gamma", gamma)
    result_a2 = await runtime_a2.coordinator.sync_once(binding_id=_BINDING_ID)
    assert result_a2.status is KnowledgeSyncRunStatus.COMPLETED
    assert result_a2.has_more is False
    assert result_a2.checkpoint_advanced is True
    assert result_a2.changes_count == 1
    assert len(sink.calls) == 2
    assert [call["relative_path"] for call in runtime_a2.transport.calls] == [
        f"/calendars/{quote(_CALENDAR_ID, safe='')}/events",
        f"/calendars/{quote(_CALENDAR_ID, safe='')}/events/event-gamma",
    ]
    assert runtime_a2.transport.calls[0]["params"] == {
        "maxResults": _PAGE_SIZE,
        "showDeleted": True,
        "singleEvents": False,
        "fields": _GOOGLE_CALENDAR_EVENTS_FIELDS,
        "pageToken": _PAGE_2,
    }
    assert all(_PAGE_2 not in repr(call) for call in runtime_a1.transport.calls)
    checkpoint_a2 = _checkpoint(document_store)
    assert checkpoint_a2 is not None
    _assert_cursor(
        checkpoint_a2,
        phase="changes",
        page_token=None,
        sync_token=_SYNC_V1,
    )
    assert all(
        _state(document_store, item).status is KnowledgeRemoteItemStatus.ACTIVE
        for item in ("event-alpha", "event-beta", "event-gamma")
    )
    gamma_content = _envelope(sink.calls[1], "event-gamma").content
    assert gamma_content is not None and gamma_content.structured_record is not None
    assert gamma_content.structured_record["event"]["recurrence"] == [
        "RRULE:FREQ=WEEKLY"
    ]
    assert gamma_content.structured_record["event"]["recurring_event_id"] == "event-gamma"

    # Re-applying the already committed page delivery is idempotent.
    await sink.apply_batch(batch=sink.calls[0])
    assert len(sink.durable_delivery_ids) == 2
    assert len(sink.durable_delivery_ids) == len(set(sink.durable_delivery_ids))

    # C1: incremental update and deletion retain sync-v1 across pagination.
    runtime_c1 = _build_runtime(
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-calendar-c1",
    )
    _collection_response(
        runtime_c1.transport,
        _page(
            [alpha_v2, _cancelled_event("event-beta")],
            next_page_token=_INCREMENTAL_PAGE_2,
        ),
        sync_token=_SYNC_V1,
    )
    _event_response(runtime_c1.transport, "event-alpha", alpha_v2)
    result_c1 = await runtime_c1.coordinator.sync_once(binding_id=_BINDING_ID)
    assert result_c1.status is KnowledgeSyncRunStatus.COMPLETED
    assert result_c1.has_more is True
    assert result_c1.changes_count == 2
    assert result_c1.active_count == 1
    assert result_c1.tombstone_count == 1
    assert len(runtime_c1.transport.calls) == 2
    assert runtime_c1.transport.calls[0]["params"]["syncToken"] == _SYNC_V1
    assert "pageToken" not in runtime_c1.transport.calls[0]["params"]
    assert runtime_c1.transport.calls[1]["relative_path"].endswith("/event-alpha")
    assert all(
        not call["relative_path"].endswith("/event-beta")
        for call in runtime_c1.transport.calls
    )
    checkpoint_c1 = _checkpoint(document_store)
    assert checkpoint_c1 is not None
    _assert_cursor(
        checkpoint_c1,
        phase="changes",
        page_token=_INCREMENTAL_PAGE_2,
        sync_token=_SYNC_V1,
    )
    assert _state(document_store, "event-alpha").status is KnowledgeRemoteItemStatus.ACTIVE
    assert _state(document_store, "event-beta").status is KnowledgeRemoteItemStatus.DELETED
    assert _state(document_store, "event-gamma").status is KnowledgeRemoteItemStatus.ACTIVE
    assert _envelope(sink.calls[3], "event-beta").change_kind is KnowledgeChangeKind.DELETED
    assert _envelope(sink.calls[3], "event-beta").content is None

    # C2: fresh runtime resumes C1's page and introduces only event-delta.
    runtime_c2 = _build_runtime(
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-calendar-c2",
    )
    _collection_response(
        runtime_c2.transport,
        _page([delta], next_sync_token=_SYNC_V2),
        page_token=_INCREMENTAL_PAGE_2,
        sync_token=_SYNC_V1,
    )
    _event_response(runtime_c2.transport, "event-delta", delta)
    result_c2 = await runtime_c2.coordinator.sync_once(binding_id=_BINDING_ID)
    assert result_c2.status is KnowledgeSyncRunStatus.COMPLETED
    assert result_c2.has_more is False
    assert result_c2.changes_count == 1
    assert len(runtime_c2.transport.calls) == 2
    assert runtime_c2.transport.calls[0]["params"] == {
        "maxResults": _PAGE_SIZE,
        "showDeleted": True,
        "singleEvents": False,
        "fields": _GOOGLE_CALENDAR_EVENTS_FIELDS,
        "pageToken": _INCREMENTAL_PAGE_2,
        "syncToken": _SYNC_V1,
    }
    checkpoint_c2 = _checkpoint(document_store)
    assert checkpoint_c2 is not None
    _assert_cursor(
        checkpoint_c2,
        phase="changes",
        page_token=None,
        sync_token=_SYNC_V2,
    )
    assert all(
        _state(document_store, item) is not None
        for item in ("event-alpha", "event-beta", "event-gamma", "event-delta")
    )
    assert _state(document_store, "event-beta").status is KnowledgeRemoteItemStatus.DELETED
    assert _state(document_store, "event-gamma").status is KnowledgeRemoteItemStatus.ACTIVE
    assert _state(document_store, "event-delta").status is KnowledgeRemoteItemStatus.ACTIVE
    alpha_updated = _envelope(sink.calls[3], "event-alpha")
    assert alpha_updated.descriptor is not None
    assert alpha_updated.descriptor.revision.version == "2"
    assert alpha_updated.descriptor.revision.content_hash != alpha_descriptor.revision.content_hash
    assert _envelope(sink.calls[4], "event-delta").change_kind is KnowledgeChangeKind.UPSERT
    assert len(
        {
            item.remote_id
            for index in (0, 1, 3, 4)
            for item in sink.calls[index].envelopes
        }
    ) == 4
    assert len(sink.durable_delivery_ids) == 4
    assert len(sink.durable_delivery_ids) == len(set(sink.durable_delivery_ids))
    assert sink.calls[0].delivery_id != sink.calls[3].delivery_id
    assert sink.calls[1].delivery_id != sink.calls[4].delivery_id
    assert _PAGE_2 not in repr(checkpoint_a2)
    assert _INCREMENTAL_PAGE_2 not in repr(checkpoint_c2)

    # D: post-incremental empty sync advances the terminal token without replay.
    runtime_d = _build_runtime(
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-calendar-d",
    )
    _collection_response(runtime_d.transport, _page([], next_sync_token=_SYNC_V3), sync_token=_SYNC_V2)
    result_d = await runtime_d.coordinator.sync_once(binding_id=_BINDING_ID)
    assert result_d.status is KnowledgeSyncRunStatus.COMPLETED
    assert result_d.changes_count == 0
    assert result_d.active_count == 0
    assert result_d.tombstone_count == 0
    assert result_d.has_more is False
    assert len(runtime_d.transport.calls) == 1
    assert len(sink.calls) == 6
    assert sink.calls[5].envelopes == ()
    assert sink.calls[5].delivery_id not in sink.durable_delivery_ids[:4]
    checkpoint_d = _checkpoint(document_store)
    assert checkpoint_d is not None
    _assert_cursor(checkpoint_d, phase="changes", page_token=None, sync_token=_SYNC_V3)
    assert {
        item: _state(document_store, item).status.value
        for item in ("event-alpha", "event-beta", "event-gamma", "event-delta")
    } == {
        "event-alpha": "active",
        "event-beta": "deleted",
        "event-gamma": "active",
        "event-delta": "active",
    }

    durable_blob = json.dumps(
        {
            "results": [result_a1.model_dump(mode="json"), result_a2.model_dump(mode="json")],
            "checkpoints": [
                checkpoint_a1.model_dump(mode="json"),
                checkpoint_a2.model_dump(mode="json"),
                checkpoint_c1.model_dump(mode="json"),
                checkpoint_c2.model_dump(mode="json"),
                checkpoint_d.model_dump(mode="json"),
            ],
            "states": [
                _state(document_store, item).model_dump(mode="json")
                for item in ("event-alpha", "event-beta", "event-gamma", "event-delta")
            ],
            "delivery_ids": sink.durable_delivery_ids,
        }
    )
    _assert_no_private_values(durable_blob)
    _assert_no_private_values(repr(checkpoint_a1.cursor))
    _assert_no_private_values(repr(checkpoint_c1.cursor))

    # E: content-fence failure happens before sink, state, or checkpoint effects.
    fence_store = InMemoryDocumentStore()
    fence_sink = IdempotentRecordingSink()
    fence_binding = _binding(_FENCE_BINDING_ID)
    runtime_fence = _build_runtime(
        document_store=fence_store,
        sink=fence_sink,
        binding=fence_binding,
        owner_id="owner-calendar-fence",
    )
    alpha_fence_v1 = _event(
        "event-alpha",
        summary="Fence Alpha",
        sequence=1,
        etag='"fence-v1"',
    )
    alpha_fence_v2 = _event(
        "event-alpha",
        summary="Fence Alpha changed",
        sequence=2,
        etag='"fence-v2"',
    )
    _collection_response(
        runtime_fence.transport,
        _page([alpha_fence_v1], next_sync_token="fence-sync"),
    )
    _event_response(runtime_fence.transport, "event-alpha", alpha_fence_v2)
    with pytest.raises(VendorKnowledgeError) as error_info:
        await runtime_fence.coordinator.sync_once(binding_id=_FENCE_BINDING_ID)
    error = error_info.value
    assert error.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert error.retryable is True
    assert error.__cause__ is None
    assert len(runtime_fence.transport.calls) == 2
    assert fence_sink.calls == []
    assert fence_sink.durable_delivery_ids == []
    assert _checkpoint(fence_store, _FENCE_BINDING_ID) is None
    assert _state(fence_store, "event-alpha", _FENCE_BINDING_ID) is None
    for private in (
        _CALENDAR_ID,
        "event-alpha",
        "Fence Alpha",
        "description",
        "attendee@example.com",
        "https://meet.google.com/abc",
        "secret",
        _CONNECTION_REF,
        "fence-sync",
        _PAGE_2,
    ):
        assert private not in repr(error)

    assert not runtime_a1.transport.responses
    assert not runtime_a2.transport.responses
    assert not runtime_c1.transport.responses
    assert not runtime_c2.transport.responses
    assert not runtime_d.transport.responses
    assert not runtime_fence.transport.responses


async def test_expired_calendar_sync_token_preserves_state_and_reconciles() -> None:
    document_store = InMemoryDocumentStore()
    sink = IdempotentRecordingSink()
    binding = _binding()
    present = _event(
        "event-present",
        summary="Present",
        sequence=1,
        etag='"present-v1"',
    )
    missing = _event(
        "event-missing",
        summary="Missing",
        sequence=1,
        etag='"missing-v1"',
    )

    initial = _build_runtime(
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-expired-initial",
    )
    _collection_response(
        initial.transport,
        _page([present, missing], next_sync_token="expired-sync"),
    )
    _event_response(initial.transport, "event-present", present)
    _event_response(initial.transport, "event-missing", missing)
    await initial.coordinator.sync_once(binding_id=_BINDING_ID)

    checkpoint_before = _checkpoint(document_store)
    assert checkpoint_before is not None
    _assert_cursor(
        checkpoint_before,
        phase="changes",
        page_token=None,
        sync_token="expired-sync",
    )
    states_before = {
        remote_id: _state(document_store, remote_id).model_dump(mode="json")
        for remote_id in ("event-present", "event-missing")
    }
    sink_calls_before = len(sink.calls)
    deliveries_before = list(sink.durable_delivery_ids)

    failed = _build_runtime(
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-expired-failed",
    )
    _collection_error_response(failed.transport, sync_token="expired-sync")
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await failed.coordinator.sync_once(binding_id=_BINDING_ID)

    error = exc_info.value
    assert error.code is VendorKnowledgeErrorCode.RECONCILIATION_REQUIRED
    assert error.retryable is False
    assert error.__cause__ is None
    assert "expired-sync" not in repr(error)
    assert "private provider response" not in repr(error)
    assert _CALENDAR_ID not in repr(error)
    assert _CONNECTION_REF not in repr(error)
    assert len(sink.calls) == sink_calls_before
    assert sink.durable_delivery_ids == deliveries_before
    assert {
        remote_id: _state(document_store, remote_id).model_dump(mode="json")
        for remote_id in ("event-present", "event-missing")
    } == states_before
    checkpoint_after_failure = _checkpoint(document_store)
    assert checkpoint_after_failure == checkpoint_before

    repeated = _build_runtime(
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-expired-repeated",
    )
    _collection_error_response(repeated.transport, sync_token="expired-sync")
    with pytest.raises(VendorKnowledgeError) as repeated_error:
        await repeated.coordinator.sync_once(binding_id=_BINDING_ID)
    assert repeated_error.value.code is VendorKnowledgeErrorCode.RECONCILIATION_REQUIRED
    assert repeated.transport.calls[0]["params"]["syncToken"] == "expired-sync"
    assert _checkpoint(document_store) == checkpoint_before

    reconciliation_a = _build_runtime(
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-reconcile-a",
    )
    _collection_response(
        reconciliation_a.transport,
        _page([present], next_page_token="reconcile-page-2"),
    )
    _event_response(reconciliation_a.transport, "event-present", present)
    first_reconciliation = await reconciliation_a.coordinator.reconcile_once(
        binding_id=_BINDING_ID,
        restart=True,
        operation_id="calendar-expired-reconciliation-1",
    )
    assert first_reconciliation.mode is KnowledgeSyncMode.RECONCILIATION
    assert first_reconciliation.has_more is True
    assert reconciliation_a.transport.calls[0]["params"].get("syncToken") is None
    assert reconciliation_a.transport.calls[0]["params"].get("pageToken") is None
    assert _checkpoint(document_store) == checkpoint_before
    assert len(sink.calls) == sink_calls_before + 1

    with pytest.raises(VendorKnowledgeError) as blocked:
        await reconciliation_a.coordinator.sync_once(binding_id=_BINDING_ID)
    assert blocked.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert blocked.value.retryable is False

    reconciliation_b = _build_runtime(
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-reconcile-b",
    )
    _collection_response(
        reconciliation_b.transport,
        _page([], next_sync_token="reconciled-sync"),
        page_token="reconcile-page-2",
    )
    final_reconciliation = await reconciliation_b.coordinator.reconcile_once(
        binding_id=_BINDING_ID,
        restart=False,
        operation_id="calendar-expired-reconciliation-1",
        trigger_delivery_id=first_reconciliation.delivery_id,
    )
    assert final_reconciliation.mode is KnowledgeSyncMode.RECONCILIATION
    assert final_reconciliation.checkpoint_advanced is True
    assert reconciliation_b.transport.calls[0]["params"]["pageToken"] == "reconcile-page-2"
    assert reconciliation_b.transport.calls[0]["params"].get("syncToken") is None
    checkpoint_after_reconciliation = _checkpoint(document_store)
    assert checkpoint_after_reconciliation is not None
    _assert_cursor(
        checkpoint_after_reconciliation,
        phase="changes",
        page_token=None,
        sync_token="reconciled-sync",
    )
    assert _state(document_store, "event-present").status is KnowledgeRemoteItemStatus.ACTIVE
    assert _state(document_store, "event-missing").status is KnowledgeRemoteItemStatus.DELETED
    assert len(sink.durable_delivery_ids) == len(set(sink.durable_delivery_ids))
    assert len(sink.calls) == sink_calls_before + 2

    subsequent = _build_runtime(
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-expired-subsequent",
    )
    _collection_response(
        subsequent.transport,
        _page([], next_sync_token="reconciled-sync-2"),
        sync_token="reconciled-sync",
    )
    result = await subsequent.coordinator.sync_once(binding_id=_BINDING_ID)
    assert result.status is KnowledgeSyncRunStatus.COMPLETED
    assert subsequent.transport.calls[0]["params"]["syncToken"] == "reconciled-sync"
    assert _checkpoint(document_store) != checkpoint_before
