# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""End-to-end Microsoft Graph Calendar knowledge adapter proof through facade and coordinator."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import pytest

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    DEFAULT_GRAPH_BASE_URL,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    MSGRAPH_CALENDAR_SOURCE_KIND,
    MsGraphCalendar,
    MsGraphCalendarAttachment,
    MsGraphCalendarAttachmentKind,
    MsGraphCalendarAttachmentPage,
    MsGraphCalendarBodyKind,
    MsGraphCalendarEventChange,
    MsGraphCalendarEventChangeKind,
    MsGraphCalendarEventChanged,
    MsGraphCalendarEventContent,
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
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_calendar import (
    MSGRAPH_CALENDAR_CURSOR_VERSION,
    MSGRAPH_CALENDAR_SCOPE_TYPE,
    encode_msgraph_calendar_scope_id,
    register_msgraph_calendar_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.utils import attribute_access
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
from intergrax.runtime.vendor_knowledge.models import (
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
    KnowledgeSyncSinkReceipt,
    KnowledgeSyncSinkReceiptStatus,
    KnowledgeSyncRunStatus,
)
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    InMemoryCheckpointRepository,
    IdempotentRecordingSink,
    InMemoryRemoteItemStateRepository,
    RecordingBindingService,
    durable_reconciliation_coordinator_kwargs,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_MAILBOX_USER_ID = "user-abc-123"
_DEFAULT_CALENDAR_ID = "default-calendar-id"
_OTHER_CALENDAR_ID = "other-calendar"
_EVT_1 = "evt-001"
_EVT_2 = "evt-002"
_EVT_3 = "evt-003"
_CK_1 = "ck-evt-1"
_CK_2 = "ck-evt-2"
_CK_3 = "ck-evt-3"
_CK_2_UPDATED = "ck-evt-2-v2"
_ATTACHMENT_ID = "att-file-001"
_SECRET_SKIP = "super-secret-skiptoken"
_SECRET_DELTA = "checkpoint-deltatoken-1"
_SECRET_DELTA_2 = "checkpoint-deltatoken-2"
_WINDOW_START = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
_WINDOW_END = datetime(2024, 2, 1, 0, 0, tzinfo=timezone.utc)
_EVENT_START = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
_EVENT_END = datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)
_CREATED_TS = datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc)
_LAST_MODIFIED_TS = datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)
_QUOTED_MAILBOX = quote(_MAILBOX_USER_ID, safe="")
_QUOTED_OTHER_CALENDAR = quote(_OTHER_CALENDAR_ID, safe="")
_DELTA_NEXT_URL = (
    f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendarView/delta"
    f"?$skiptoken={_SECRET_SKIP}"
)
_DELTA_CHECKPOINT_URL = (
    f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendarView/delta"
    f"?$deltatoken={_SECRET_DELTA}"
)
_INCREMENTAL_DELTA_URL = (
    f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendarView/delta"
    f"?$deltatoken={_SECRET_DELTA_2}"
)
_SNAPSHOT_NEXT_URL = (
    f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
    f"{_QUOTED_OTHER_CALENDAR}/calendarView?$skiptoken={_SECRET_SKIP}"
)
_PRIMARY_BINDING_ID = "msgraph-calendar-primary-binding"
_NON_PRIMARY_BINDING_ID = "msgraph-calendar-non-primary-binding"


def _window() -> MsGraphCalendarViewWindow:
    return MsGraphCalendarViewWindow(start_at=_WINDOW_START, end_at=_WINDOW_END)


def _default_calendar() -> MsGraphCalendar:
    return MsGraphCalendar(
        mailbox_user_id=_MAILBOX_USER_ID,
        remote_id=_DEFAULT_CALENDAR_ID,
        name="Calendar",
        change_key="calendar-change-key",
        is_default_calendar=True,
        can_edit=True,
        can_share=True,
        can_view_private_items=False,
        is_removable=False,
        owner=None,
        allowed_online_meeting_providers=(
            MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
        ),
        default_online_meeting_provider=MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
    )


def _other_calendar() -> MsGraphCalendar:
    return MsGraphCalendar(
        mailbox_user_id=_MAILBOX_USER_ID,
        remote_id=_OTHER_CALENDAR_ID,
        name="Other Calendar",
        change_key="other-calendar-change-key",
        is_default_calendar=False,
        can_edit=True,
        can_share=True,
        can_view_private_items=False,
        is_removable=False,
        owner=None,
        allowed_online_meeting_providers=(
            MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
        ),
        default_online_meeting_provider=MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
    )


def _encode_event_remote_id(
    *,
    calendar_remote_id: str,
    event_remote_id: str,
) -> str:
    payload = {
        "schema_version": "msgraph.calendar.event-id.v1",
        "mailbox_user_id": _MAILBOX_USER_ID,
        "calendar_remote_id": calendar_remote_id,
        "event_remote_id": event_remote_id,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _response_status() -> MsGraphCalendarResponseStatus:
    return MsGraphCalendarResponseStatus(response=MsGraphCalendarResponseType.ORGANIZER)


def _active_event(
    *,
    calendar_remote_id: str,
    remote_id: str,
    change_key: str,
    has_attachments: bool = False,
) -> MsGraphCalendarEventChange:
    return MsGraphCalendarEventChange(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=calendar_remote_id,
        remote_id=remote_id,
        kind=MsGraphCalendarEventChangeKind.ACTIVE,
        change_key=change_key,
        event_type=MsGraphCalendarEventType.SINGLE_INSTANCE,
        start_at=_EVENT_START,
        end_at=_EVENT_END,
        last_modified_at=_LAST_MODIFIED_TS,
        is_all_day=False,
        is_cancelled=False,
        is_draft=False,
        has_attachments=has_attachments,
        is_online_meeting=False,
    )


def _removed_event(
    *,
    calendar_remote_id: str,
    remote_id: str,
) -> MsGraphCalendarEventChange:
    return MsGraphCalendarEventChange(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=calendar_remote_id,
        remote_id=remote_id,
        kind=MsGraphCalendarEventChangeKind.REMOVED,
        removed_reason="deleted",
    )


def _delta_page(
    *,
    calendar_remote_id: str,
    items: tuple[MsGraphCalendarEventChange, ...],
    continuation_kind: MsGraphKnowledgeContinuationKind,
    url: str,
) -> MsGraphCalendarEventDeltaPage:
    return MsGraphCalendarEventDeltaPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=calendar_remote_id,
        window=_window(),
        items=items,
        continuation=MsGraphKnowledgeContinuation(kind=continuation_kind, url=url),
    )


def _snapshot_page(
    *,
    calendar_remote_id: str,
    items: tuple[MsGraphCalendarEventChange, ...],
    continuation_url: str | None = None,
) -> MsGraphCalendarEventSnapshotPage:
    continuation = None
    if continuation_url is not None:
        continuation = MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=continuation_url,
        )
    return MsGraphCalendarEventSnapshotPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=calendar_remote_id,
        window=_window(),
        items=items,
        continuation=continuation,
    )


def _event_content(
    *,
    calendar_remote_id: str,
    remote_id: str,
    content_revision: str,
    subject: str,
    body_content: str,
    has_attachments: bool = False,
) -> MsGraphCalendarEventContent:
    return MsGraphCalendarEventContent(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=calendar_remote_id,
        remote_id=remote_id,
        content_revision=content_revision,
        event_type=MsGraphCalendarEventType.SINGLE_INSTANCE,
        subject=subject,
        body_kind=MsGraphCalendarBodyKind.TEXT,
        body_content=body_content,
        start_at=_EVENT_START,
        end_at=_EVENT_END,
        created_at=_CREATED_TS,
        last_modified_at=_LAST_MODIFIED_TS,
        importance=MsGraphCalendarImportance.NORMAL,
        sensitivity=MsGraphCalendarSensitivity.NORMAL,
        show_as=MsGraphCalendarShowAs.BUSY,
        response_status=_response_status(),
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


def _attachment_page(
    *,
    calendar_remote_id: str,
    event_remote_id: str,
    event_revision: str,
) -> MsGraphCalendarAttachmentPage:
    return MsGraphCalendarAttachmentPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=calendar_remote_id,
        event_remote_id=event_remote_id,
        event_revision=event_revision,
        items=(
            MsGraphCalendarAttachment(
                mailbox_user_id=_MAILBOX_USER_ID,
                calendar_remote_id=calendar_remote_id,
                event_remote_id=event_remote_id,
                event_revision=event_revision,
                remote_id=_ATTACHMENT_ID,
                kind=MsGraphCalendarAttachmentKind.FILE,
                name="report.pdf",
                content_type="application/pdf",
                size_bytes=42,
                is_inline=False,
                content_id=None,
                last_modified_at=_LAST_MODIFIED_TS,
            ),
        ),
        continuation=None,
    )


class _CalendarFakeCollaborationSuite(CollaborationSuite):
    def __init__(
        self,
        *,
        calendar: MsGraphCalendar,
        delta_pages: list[MsGraphCalendarEventDeltaPage] | None = None,
        snapshot_pages: list[MsGraphCalendarEventSnapshotPage] | None = None,
        incremental_delta_page: MsGraphCalendarEventDeltaPage | None = None,
        content: dict[tuple[str, str], MsGraphCalendarEventContent] | None = None,
        attachment_pages: dict[tuple[str, str], MsGraphCalendarAttachmentPage]
        | None = None,
        content_raises_changed: set[tuple[str, str]] | None = None,
    ) -> None:
        self.calendar = calendar
        self.delta_calls: list[dict[str, Any]] = []
        self.snapshot_calls: list[dict[str, Any]] = []
        self.content_calls: list[dict[str, Any]] = []
        self.attachment_calls: list[dict[str, Any]] = []
        self.forbidden_calls: list[str] = []
        self._delta_pages = list(delta_pages or [])
        self._delta_pages_backup = list(self._delta_pages)
        self._snapshot_pages = list(snapshot_pages or [])
        self._snapshot_pages_backup = list(self._snapshot_pages)
        self._incremental_delta_page = incremental_delta_page
        self._content = content or {}
        self._attachment_pages = attachment_pages or {}
        self._content_raises_changed = content_raises_changed or set()

    def _reset_delta_if_needed(self) -> None:
        if not self._delta_pages and self._delta_pages_backup:
            self._delta_pages = list(self._delta_pages_backup)

    def _reset_snapshot_if_needed(self) -> None:
        if not self._snapshot_pages and self._snapshot_pages_backup:
            self._snapshot_pages = list(self._snapshot_pages_backup)

    def read_calendar_events_delta_page_by_reference(
        self,
        *,
        calendar,
        window: MsGraphCalendarViewWindow,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int,
    ) -> MsGraphCalendarEventDeltaPage:
        if continuation is None:
            self._reset_delta_if_needed()
        if (
            continuation is not None
            and continuation.kind == MsGraphKnowledgeContinuationKind.DELTA
        ):
            if (
                continuation.url == _DELTA_CHECKPOINT_URL
                and self._incremental_delta_page is not None
            ):
                self.delta_calls.append(
                    {
                        "calendar": calendar,
                        "window": window,
                        "continuation": continuation,
                        "limit": limit,
                    }
                )
                return self._incremental_delta_page
        self.delta_calls.append(
            {
                "calendar": calendar,
                "window": window,
                "continuation": continuation,
                "limit": limit,
            }
        )
        if not self._delta_pages:
            raise AssertionError("unexpected delta page request")
        return self._delta_pages.pop(0)

    def read_calendar_events_snapshot_page_by_reference(
        self,
        *,
        calendar,
        window: MsGraphCalendarViewWindow,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int,
    ) -> MsGraphCalendarEventSnapshotPage:
        if continuation is None:
            self._reset_snapshot_if_needed()
        self.snapshot_calls.append(
            {
                "calendar": calendar,
                "window": window,
                "continuation": continuation,
                "limit": limit,
            }
        )
        if not self._snapshot_pages:
            raise AssertionError("unexpected snapshot page request")
        return self._snapshot_pages.pop(0)

    def read_calendar_event_content(
        self,
        *,
        event: MsGraphCalendarEventChange,
        max_chars: int,
    ) -> MsGraphCalendarEventContent:
        self.content_calls.append({"event": event, "max_chars": max_chars})
        key = (event.remote_id, event.change_key or "")
        if key in self._content_raises_changed:
            raise MsGraphCalendarEventChanged()
        return self._content[key]

    def read_calendar_attachments_page(
        self,
        *,
        event: MsGraphCalendarEventChange,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphCalendarAttachmentPage:
        self.attachment_calls.append(
            {
                "event": event,
                "continuation": continuation,
                "limit": limit,
            }
        )
        key = (event.remote_id, event.change_key or "")
        return self._attachment_pages[key]

    def read_calendar_file_attachment_content(self, **kwargs: Any):
        self.forbidden_calls.append("attachment_bytes")
        raise AssertionError("attachment bytes must not be read")

    def read_calendars_page(self, **kwargs: Any):
        self.forbidden_calls.append("calendar_inventory")
        raise AssertionError("calendar inventory must not be called")

    def read_teams_chats_page(self, **kwargs: Any):
        self.forbidden_calls.append("teams_chat")
        raise AssertionError("teams chat must not be called")

    def read_mail_messages_delta_page(self, **kwargs: Any):
        self.forbidden_calls.append("mail")
        raise AssertionError("mail must not be called")

    def get_message(self, user_id: str, message_id: str):
        raise NotImplementedError

    def list_messages(self, user_id: str, *, folder: str = "inbox", limit: int = 25):
        raise NotImplementedError

    def send_mail(self, user_id: str, *, subject: str, body: str, to):
        raise NotImplementedError

    def list_calendar_events(
        self, user_id: str, *, start: str, end: str, limit: int = 50
    ):
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


@dataclass
class _GraphResolver:
    integration: Ms365GraphCollaborationSuiteIntegration

    def resolve(self, *, source) -> Ms365GraphCollaborationSuiteIntegration:
        return self.integration


class _FailingItemStateRepository:
    def __init__(self, delegate: object, *, fail_times: int = 1) -> None:
        self._delegate = delegate
        self.fail_times = fail_times

    def __getattr__(self, name: str) -> object:
        return attribute_access.optional(self._delegate, name)

    def apply_batch(self, **kwargs: Any) -> None:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("item-state boom")
        self._delegate.apply_batch(**kwargs)  # type: ignore[attr-defined]


class _FailingCheckpointRepository:
    def __init__(self, delegate: object, *, fail_times: int = 1) -> None:
        self._delegate = delegate
        self.fail_times = fail_times
        self.successful_commits = 0

    def __getattr__(self, name: str) -> object:
        return attribute_access.optional(self._delegate, name)

    def commit(
        self,
        checkpoint: object,
        *,
        expected_previous: object,
        expected_publication_fence: object | None = None,
        publication_permit: object | None = None,
    ) -> None:
        _ = expected_publication_fence, publication_permit
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("checkpoint boom")
        self._delegate.commit(  # type: ignore[attr-defined]
            checkpoint,
            expected_previous=expected_previous,
            expected_publication_fence=expected_publication_fence,
            publication_permit=publication_permit,
        )
        self.successful_commits += 1


class _SinkBackedReceiptInspector:
    def __init__(self, sink: IdempotentRecordingSink) -> None:
        self._sink = sink

    def inspect_receipt(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        prepared_batch_payload_fingerprint: str,
    ) -> KnowledgeSyncSinkReceipt:
        if delivery_id not in self._sink.durable_delivery_ids:
            return KnowledgeSyncSinkReceipt(
                status=KnowledgeSyncSinkReceiptStatus.ABSENT
            )
        return KnowledgeSyncSinkReceipt(
            status=KnowledgeSyncSinkReceiptStatus.APPLIED,
            delivery_id=delivery_id,
            prepared_batch_payload_fingerprint=prepared_batch_payload_fingerprint,
        )


def _public_blob(value: object) -> str:
    return json.dumps(value, default=str)


def _assert_no_secrets(blob: str) -> None:
    forbidden = (
        _SECRET_SKIP,
        _SECRET_DELTA,
        _SECRET_DELTA_2,
        _DELTA_NEXT_URL,
        _DELTA_CHECKPOINT_URL,
        _INCREMENTAL_DELTA_URL,
        _SNAPSHOT_NEXT_URL,
        _CK_1,
        _CK_2,
        _CK_3,
        _CK_2_UPDATED,
        "Authorization",
        "skiptoken",
        "deltatoken",
    )
    for item in forbidden:
        assert item not in blob


def _primary_content_map() -> dict[tuple[str, str], MsGraphCalendarEventContent]:
    calendar_remote_id = _DEFAULT_CALENDAR_ID
    return {
        (_EVT_1, _CK_1): _event_content(
            calendar_remote_id=calendar_remote_id,
            remote_id=_EVT_1,
            content_revision=_CK_1,
            subject="Event one",
            body_content="body-one",
        ),
        (_EVT_2, _CK_2): _event_content(
            calendar_remote_id=calendar_remote_id,
            remote_id=_EVT_2,
            content_revision=_CK_2,
            subject="Event two",
            body_content="body-two",
        ),
        (_EVT_2, _CK_2_UPDATED): _event_content(
            calendar_remote_id=calendar_remote_id,
            remote_id=_EVT_2,
            content_revision=_CK_2_UPDATED,
            subject="Event two updated",
            body_content="body-two-updated",
        ),
    }


def _non_primary_content_map() -> dict[tuple[str, str], MsGraphCalendarEventContent]:
    calendar_remote_id = _OTHER_CALENDAR_ID
    return {
        (_EVT_1, _CK_1): _event_content(
            calendar_remote_id=calendar_remote_id,
            remote_id=_EVT_1,
            content_revision=_CK_1,
            subject="Event one",
            body_content="body-one",
        ),
        (_EVT_2, _CK_2): _event_content(
            calendar_remote_id=calendar_remote_id,
            remote_id=_EVT_2,
            content_revision=_CK_2,
            subject="Event two",
            body_content="body-two",
        ),
        (_EVT_3, _CK_3): _event_content(
            calendar_remote_id=calendar_remote_id,
            remote_id=_EVT_3,
            content_revision=_CK_3,
            subject="Event three",
            body_content="body-three",
        ),
    }


def _build_coordinator(
    fake: _CalendarFakeCollaborationSuite,
    *,
    calendar: MsGraphCalendar,
    binding_id: str,
    safe_display_name: str,
    document_store: InMemoryDocumentStore | None = None,
    checkpoint_repository: object | None = None,
    item_state_repository: object | None = None,
    sink: IdempotentRecordingSink | None = None,
    sink_receipt_inspector: object | None = None,
):
    integration = _CalendarTestIntegration.from_client(fake, enabled=True)
    registry = KnowledgeAdapterRegistry()
    register_msgraph_calendar_knowledge_adapter(registry)
    facade = VendorKnowledgeFacadeService(
        tenant_id="tenant-1",
        resolver=_GraphResolver(integration=integration),
        adapter_registry=registry,
    )
    if document_store is None:
        document_store = InMemoryDocumentStore()
    lease_repo = DocumentStoreKnowledgeSourceLeaseRepository(document_store)
    checkpoint_repo = (
        checkpoint_repository
        or DocumentStoreKnowledgeSyncCheckpointRepository(document_store)
    )
    state_repo = (
        item_state_repository
        or DocumentStoreKnowledgeRemoteItemStateRepository(document_store)
    )
    sink = sink or IdempotentRecordingSink()
    binding = KnowledgeSourceBinding(
        binding_id=binding_id,
        tenant_id="tenant-1",
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_CALENDAR_SOURCE_KIND,
        connection_ref="conn-1",
        safe_display_name="Calendar Binding",
        scope=KnowledgeSourceScope(
            remote_scope_id=encode_msgraph_calendar_scope_id(
                calendar=calendar,
                window=_window(),
            ),
            remote_scope_type=MSGRAPH_CALENDAR_SCOPE_TYPE,
            safe_display_name=safe_display_name,
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
    )
    reconciliation_kwargs = durable_reconciliation_coordinator_kwargs(
        state_repository=InMemoryRemoteItemStateRepository(),
        document_store=document_store,
    )
    if sink_receipt_inspector is not None:
        reconciliation_kwargs["sink_receipt_inspector"] = sink_receipt_inspector
    coordinator = VendorKnowledgeSyncCoordinator(
        tenant_id="tenant-1",
        owner_id="owner-1",
        binding_service=RecordingBindingService(binding=binding),  # type: ignore[arg-type]
        facade=facade,
        lease_repository=lease_repo,
        checkpoint_repository=checkpoint_repo,
        item_state_repository=state_repo,
        sink=sink,
        lease_ttl_seconds=30,
        **reconciliation_kwargs,
    )
    return coordinator, sink, checkpoint_repo, state_repo, fake, integration


async def _reconcile_until_complete(
    coordinator: VendorKnowledgeSyncCoordinator,
    *,
    binding_id: str,
    operation_id: str,
    call_trace: list[tuple[bool, str | None]] | None = None,
) -> list:
    results = []
    restart = True
    trigger_delivery_id: str | None = None
    while True:
        if call_trace is not None:
            call_trace.append((restart, trigger_delivery_id))
        result = await coordinator.reconcile_once(
            binding_id=binding_id,
            restart=restart,
            operation_id=operation_id,
            trigger_delivery_id=trigger_delivery_id,
        )
        results.append(result)
        assert result.status is KnowledgeSyncRunStatus.COMPLETED
        if not result.has_more:
            break
        restart = False
        trigger_delivery_id = result.delivery_id
    return results


def _primary_fake(**kwargs: Any) -> _CalendarFakeCollaborationSuite:
    calendar_remote_id = _DEFAULT_CALENDAR_ID
    defaults: dict[str, Any] = {
        "calendar": _default_calendar(),
        "delta_pages": [
            _delta_page(
                calendar_remote_id=calendar_remote_id,
                items=(
                    _active_event(
                        calendar_remote_id=calendar_remote_id,
                        remote_id=_EVT_1,
                        change_key=_CK_1,
                    ),
                    _active_event(
                        calendar_remote_id=calendar_remote_id,
                        remote_id=_EVT_2,
                        change_key=_CK_2,
                    ),
                ),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_DELTA_NEXT_URL,
            ),
            _delta_page(
                calendar_remote_id=calendar_remote_id,
                items=(
                    _removed_event(
                        calendar_remote_id=calendar_remote_id,
                        remote_id=_EVT_3,
                    ),
                ),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_CHECKPOINT_URL,
            ),
        ],
        "incremental_delta_page": _delta_page(
            calendar_remote_id=calendar_remote_id,
            items=(
                _active_event(
                    calendar_remote_id=calendar_remote_id,
                    remote_id=_EVT_2,
                    change_key=_CK_2_UPDATED,
                ),
            ),
            continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=_INCREMENTAL_DELTA_URL,
        ),
        "content": _primary_content_map(),
    }
    defaults.update(kwargs)
    return _CalendarFakeCollaborationSuite(**defaults)


def _non_primary_fake(**kwargs: Any) -> _CalendarFakeCollaborationSuite:
    calendar_remote_id = _OTHER_CALENDAR_ID
    defaults: dict[str, Any] = {
        "calendar": _other_calendar(),
        "snapshot_pages": [
            _snapshot_page(
                calendar_remote_id=calendar_remote_id,
                items=(
                    _active_event(
                        calendar_remote_id=calendar_remote_id,
                        remote_id=_EVT_1,
                        change_key=_CK_1,
                    ),
                    _active_event(
                        calendar_remote_id=calendar_remote_id,
                        remote_id=_EVT_2,
                        change_key=_CK_2,
                    ),
                ),
                continuation_url=_SNAPSHOT_NEXT_URL,
            ),
            _snapshot_page(
                calendar_remote_id=calendar_remote_id,
                items=(
                    _active_event(
                        calendar_remote_id=calendar_remote_id,
                        remote_id=_EVT_3,
                        change_key=_CK_3,
                    ),
                ),
            ),
        ],
        "content": _non_primary_content_map(),
    }
    defaults.update(kwargs)
    return _CalendarFakeCollaborationSuite(**defaults)


@pytest.mark.asyncio
async def test_msgraph_calendar_primary_facade_coordinator_reconciliation_and_incremental() -> (
    None
):
    fake = _primary_fake()
    coordinator, sink, checkpoint_repo, state_repo, fake, integration = (
        _build_coordinator(
            fake,
            calendar=_default_calendar(),
            binding_id=_PRIMARY_BINDING_ID,
            safe_display_name="Primary Calendar",
        )
    )
    integration_id = id(integration)

    call_trace: list[tuple[bool, str | None]] = []
    results = await _reconcile_until_complete(
        coordinator,
        binding_id=_PRIMARY_BINDING_ID,
        operation_id="msgraph-calendar-primary",
        call_trace=call_trace,
    )
    assert len(results) == 2
    assert all(result.has_more for result in results[:-1])
    assert results[-1].has_more is False
    assert len(sink.calls) == 2
    assert len({batch.delivery_id for batch in sink.calls}) == 2
    assert call_trace == [
        (True, None),
        (False, results[0].delivery_id),
    ]

    traversal: list[tuple[str, str | None]] = []
    for batch in sink.calls:
        assert batch.mode.value == "reconciliation"
        for envelope in batch.envelopes:
            _assert_no_secrets(_public_blob(envelope.model_dump(mode="json")))
            assert envelope.permissions is None
            if envelope.change_kind.value == "deleted":
                assert envelope.descriptor is None
                assert envelope.content is None
                traversal.append(
                    (
                        "deleted",
                        _encode_event_remote_id(
                            calendar_remote_id=_DEFAULT_CALENDAR_ID,
                            event_remote_id=_EVT_3,
                        ),
                    )
                )
                continue
            descriptor = envelope.descriptor
            assert descriptor is not None
            assert descriptor.item_type == "msgraph_calendar_event"
            assert envelope.content is not None
            assert envelope.content.mode is KnowledgeContentMode.STRUCTURED_RECORD
            record = envelope.content.structured_record
            assert record is not None
            assert record["schema"] == "msgraph.calendar.event.knowledge.v1"
            traversal.append(("active", record["subject"]))

    assert traversal == [
        ("active", "Event one"),
        ("active", "Event two"),
        (
            "deleted",
            _encode_event_remote_id(
                calendar_remote_id=_DEFAULT_CALENDAR_ID,
                event_remote_id=_EVT_3,
            ),
        ),
    ]

    checkpoint = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id=_PRIMARY_BINDING_ID,
    )
    assert checkpoint is not None
    assert checkpoint.cursor is not None
    assert checkpoint.cursor.version == MSGRAPH_CALENDAR_CURSOR_VERSION
    assert _SECRET_DELTA not in _public_blob(checkpoint.model_dump(mode="json"))

    assert fake.delta_calls[0]["continuation"] is None
    assert fake.delta_calls[1]["continuation"] is not None
    assert fake.delta_calls[1]["continuation"].url == _DELTA_NEXT_URL
    assert fake.forbidden_calls == []

    for event_id in (_EVT_1, _EVT_2):
        state = state_repo.get(
            tenant_id="tenant-1",
            binding_id=_PRIMARY_BINDING_ID,
            remote_id=_encode_event_remote_id(
                calendar_remote_id=_DEFAULT_CALENDAR_ID,
                event_remote_id=event_id,
            ),
        )
        assert state is not None
    deleted_state = state_repo.get(
        tenant_id="tenant-1",
        binding_id=_PRIMARY_BINDING_ID,
        remote_id=_encode_event_remote_id(
            calendar_remote_id=_DEFAULT_CALENDAR_ID,
            event_remote_id=_EVT_3,
        ),
    )
    assert deleted_state is not None
    assert deleted_state.status is KnowledgeRemoteItemStatus.DELETED

    incremental = await coordinator.sync_once(binding_id=_PRIMARY_BINDING_ID)
    assert incremental.status is KnowledgeSyncRunStatus.COMPLETED
    assert fake.delta_calls[-1]["continuation"] is not None
    assert (
        fake.delta_calls[-1]["continuation"].kind
        == MsGraphKnowledgeContinuationKind.DELTA
    )
    assert fake.delta_calls[-1]["continuation"].url == _DELTA_CHECKPOINT_URL

    incremental_batch = sink.calls[-1]
    updated_envelope = next(
        envelope
        for envelope in incremental_batch.envelopes
        if envelope.change_kind.value == "upsert"
    )
    assert updated_envelope.content is not None
    assert updated_envelope.content.structured_record is not None
    assert updated_envelope.content.structured_record["subject"] == "Event two updated"

    updated_checkpoint = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id=_PRIMARY_BINDING_ID,
    )
    assert updated_checkpoint is not None
    assert updated_checkpoint.cursor is not None
    assert updated_checkpoint.cursor.version == MSGRAPH_CALENDAR_CURSOR_VERSION

    public_proof = _public_blob(
        {
            "results": [result.model_dump(mode="json") for result in results],
            "incremental": incremental.model_dump(mode="json"),
            "sink": [batch.model_dump(mode="json") for batch in sink.calls],
            "integration_id": integration_id,
        }
    )
    _assert_no_secrets(public_proof)
    assert id(integration) == integration_id


@pytest.mark.asyncio
async def test_msgraph_calendar_non_primary_facade_coordinator_reconciliation() -> None:
    fake = _non_primary_fake()
    coordinator, sink, checkpoint_repo, state_repo, fake, integration = (
        _build_coordinator(
            fake,
            calendar=_other_calendar(),
            binding_id=_NON_PRIMARY_BINDING_ID,
            safe_display_name="Other Calendar",
        )
    )
    integration_id = id(integration)

    call_trace: list[tuple[bool, str | None]] = []
    results = await _reconcile_until_complete(
        coordinator,
        binding_id=_NON_PRIMARY_BINDING_ID,
        operation_id="msgraph-calendar-non-primary",
        call_trace=call_trace,
    )
    assert len(results) == 2
    assert all(result.has_more for result in results[:-1])
    assert results[-1].has_more is False
    assert len(sink.calls) == 2
    assert call_trace == [
        (True, None),
        (False, results[0].delivery_id),
    ]

    subjects = []
    for batch in sink.calls:
        assert batch.mode.value == "reconciliation"
        for envelope in batch.envelopes:
            assert envelope.content is not None
            record = envelope.content.structured_record
            assert record is not None
            subjects.append(record["subject"])
    assert subjects == ["Event one", "Event two", "Event three"]

    checkpoint = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id=_NON_PRIMARY_BINDING_ID,
    )
    assert checkpoint is not None
    assert checkpoint.cursor is not None
    assert checkpoint.cursor.version == MSGRAPH_CALENDAR_CURSOR_VERSION

    assert fake.snapshot_calls[0]["continuation"] is None
    assert fake.snapshot_calls[1]["continuation"] is not None
    assert fake.snapshot_calls[1]["continuation"].url == _SNAPSHOT_NEXT_URL
    assert fake.delta_calls == []
    assert fake.forbidden_calls == []

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id=_NON_PRIMARY_BINDING_ID)
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY

    active_remote_id = _encode_event_remote_id(
        calendar_remote_id=_OTHER_CALENDAR_ID,
        event_remote_id=_EVT_1,
    )
    active_state = state_repo.get(
        tenant_id="tenant-1",
        binding_id=_NON_PRIMARY_BINDING_ID,
        remote_id=active_remote_id,
    )
    assert active_state is not None
    assert active_state.status is KnowledgeRemoteItemStatus.ACTIVE

    prior_sink_calls = len(sink.calls)
    fake._snapshot_pages = [
        _snapshot_page(
            calendar_remote_id=_OTHER_CALENDAR_ID,
            items=(
                _active_event(
                    calendar_remote_id=_OTHER_CALENDAR_ID,
                    remote_id=_EVT_2,
                    change_key=_CK_2,
                ),
            ),
            continuation_url=_SNAPSHOT_NEXT_URL,
        ),
        _snapshot_page(
            calendar_remote_id=_OTHER_CALENDAR_ID,
            items=(
                _active_event(
                    calendar_remote_id=_OTHER_CALENDAR_ID,
                    remote_id=_EVT_2,
                    change_key=_CK_2,
                ),
            ),
        ),
    ]
    absence_operation_id = "msgraph-calendar-non-primary-absence"
    restart_result = await coordinator.reconcile_once(
        binding_id=_NON_PRIMARY_BINDING_ID,
        restart=True,
        operation_id=absence_operation_id,
    )
    assert restart_result.has_more is True
    assert len(sink.calls) == prior_sink_calls + 1
    latest_batch = sink.calls[-1]
    assert all(
        envelope.change_kind.value != "deleted" for envelope in latest_batch.envelopes
    )
    assert active_remote_id not in {
        envelope.remote_id for envelope in latest_batch.envelopes
    }
    still_active = state_repo.get(
        tenant_id="tenant-1",
        binding_id=_NON_PRIMARY_BINDING_ID,
        remote_id=active_remote_id,
    )
    assert still_active is not None
    assert still_active.status is KnowledgeRemoteItemStatus.ACTIVE

    final_result = await coordinator.reconcile_once(
        binding_id=_NON_PRIMARY_BINDING_ID,
        restart=False,
        operation_id=absence_operation_id,
        trigger_delivery_id=restart_result.delivery_id,
    )
    assert final_result.has_more is False
    assert final_result.delivery_id != restart_result.delivery_id
    final_batch = sink.calls[-1]
    final_deleted_ids = {
        envelope.remote_id
        for envelope in final_batch.envelopes
        if envelope.change_kind.value == "deleted"
    }
    assert active_remote_id in final_deleted_ids
    assert all(
        envelope.change_kind.value == "deleted"
        or envelope.remote_id
        == _encode_event_remote_id(
            calendar_remote_id=_OTHER_CALENDAR_ID,
            event_remote_id=_EVT_2,
        )
        for envelope in final_batch.envelopes
    )
    deleted_state = state_repo.get(
        tenant_id="tenant-1",
        binding_id=_NON_PRIMARY_BINDING_ID,
        remote_id=active_remote_id,
    )
    assert deleted_state is not None
    assert deleted_state.status is KnowledgeRemoteItemStatus.DELETED
    assert deleted_state.tenant_id == "tenant-1"
    assert deleted_state.binding_id == _NON_PRIMARY_BINDING_ID
    assert deleted_state.binding_configuration_version == 1
    completed_checkpoint = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id=_NON_PRIMARY_BINDING_ID,
    )
    assert completed_checkpoint is not None
    assert completed_checkpoint.cursor is not None
    assert completed_checkpoint.cursor.version == MSGRAPH_CALENDAR_CURSOR_VERSION
    assert fake.snapshot_calls[-1]["continuation"] is not None
    assert fake.snapshot_calls[-1]["continuation"].url == _SNAPSHOT_NEXT_URL

    public_proof = _public_blob(
        {
            "results": [result.model_dump(mode="json") for result in results],
            "absence": [
                restart_result.model_dump(mode="json"),
                final_result.model_dump(mode="json"),
            ],
            "integration_id": integration_id,
        }
    )
    _assert_no_secrets(public_proof)
    assert id(integration) == integration_id


@pytest.mark.asyncio
async def test_msgraph_calendar_primary_revision_race_does_not_commit_checkpoint() -> (
    None
):
    fake = _primary_fake(
        content_raises_changed={(_EVT_1, _CK_1)},
        delta_pages=[
            _delta_page(
                calendar_remote_id=_DEFAULT_CALENDAR_ID,
                items=(
                    _active_event(
                        calendar_remote_id=_DEFAULT_CALENDAR_ID,
                        remote_id=_EVT_1,
                        change_key=_CK_1,
                    ),
                ),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_DELTA_NEXT_URL,
            ),
        ],
    )
    coordinator, sink, checkpoint_repo, state_repo, fake, integration = (
        _build_coordinator(
            fake,
            calendar=_default_calendar(),
            binding_id=_PRIMARY_BINDING_ID,
            safe_display_name="Primary Calendar",
        )
    )
    integration_id = id(integration)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id=_PRIMARY_BINDING_ID,
            restart=True,
            operation_id="msgraph-calendar-revision-race",
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert (
        checkpoint_repo.get(
            tenant_id="tenant-1",
            binding_id=_PRIMARY_BINDING_ID,
        )
        is None
    )
    assert (
        state_repo.get(
            tenant_id="tenant-1",
            binding_id=_PRIMARY_BINDING_ID,
            remote_id=_encode_event_remote_id(
                calendar_remote_id=_DEFAULT_CALENDAR_ID,
                event_remote_id=_EVT_1,
            ),
        )
        is None
    )
    assert len(sink.calls) == 0
    assert len(fake.delta_calls) == 1
    assert id(integration) == integration_id


@pytest.mark.asyncio
async def test_msgraph_calendar_primary_malformed_page_does_not_commit_checkpoint() -> (
    None
):
    malformed_page = MsGraphCalendarEventDeltaPage.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        calendar_remote_id=_DEFAULT_CALENDAR_ID,
        window=_window(),
        items=(
            _active_event(
                calendar_remote_id="wrong-calendar",
                remote_id=_EVT_1,
                change_key=_CK_1,
            ),
        ),
        continuation=MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=_DELTA_CHECKPOINT_URL,
        ),
    )
    fake = _primary_fake(delta_pages=[malformed_page])
    coordinator, sink, checkpoint_repo, state_repo, fake, integration = (
        _build_coordinator(
            fake,
            calendar=_default_calendar(),
            binding_id=_PRIMARY_BINDING_ID,
            safe_display_name="Primary Calendar",
        )
    )
    integration_id = id(integration)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id=_PRIMARY_BINDING_ID,
            restart=True,
            operation_id="msgraph-calendar-malformed-page",
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert exc_info.value.retryable is False
    assert (
        checkpoint_repo.get(
            tenant_id="tenant-1",
            binding_id=_PRIMARY_BINDING_ID,
        )
        is None
    )
    assert (
        state_repo.get(
            tenant_id="tenant-1",
            binding_id=_PRIMARY_BINDING_ID,
            remote_id=_encode_event_remote_id(
                calendar_remote_id=_DEFAULT_CALENDAR_ID,
                event_remote_id=_EVT_1,
            ),
        )
        is None
    )
    assert len(sink.calls) == 0
    assert len(fake.delta_calls) == 1
    assert id(integration) == integration_id


@pytest.mark.asyncio
async def test_msgraph_calendar_primary_sink_failure_does_not_commit_checkpoint_or_state() -> (
    None
):
    fake = _primary_fake()
    coordinator, sink, checkpoint_repo, state_repo, fake, integration = (
        _build_coordinator(
            fake,
            calendar=_default_calendar(),
            binding_id=_PRIMARY_BINDING_ID,
            safe_display_name="Primary Calendar",
        )
    )
    integration_id = id(integration)
    sink.fail_times = 1
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id=_PRIMARY_BINDING_ID,
            restart=True,
            operation_id="msgraph-calendar-sink-failure",
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert len(sink.calls) == 1
    assert sink.durable_delivery_ids == []
    assert (
        checkpoint_repo.get(
            tenant_id="tenant-1",
            binding_id=_PRIMARY_BINDING_ID,
        )
        is None
    )
    for event_id in (_EVT_1, _EVT_2):
        assert (
            state_repo.get(
                tenant_id="tenant-1",
                binding_id=_PRIMARY_BINDING_ID,
                remote_id=_encode_event_remote_id(
                    calendar_remote_id=_DEFAULT_CALENDAR_ID,
                    event_remote_id=event_id,
                ),
            )
            is None
        )
    assert len(fake.delta_calls) == 1
    assert fake.delta_calls[0]["continuation"] is None
    assert id(integration) == integration_id


@pytest.mark.asyncio
async def test_msgraph_calendar_primary_retry_same_page_after_sink_failure() -> None:
    fake = _primary_fake(
        delta_pages=[
            _delta_page(
                calendar_remote_id=_DEFAULT_CALENDAR_ID,
                items=(
                    _active_event(
                        calendar_remote_id=_DEFAULT_CALENDAR_ID,
                        remote_id=_EVT_1,
                        change_key=_CK_1,
                    ),
                ),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_CHECKPOINT_URL,
            ),
        ],
    )
    single_page = fake._delta_pages_backup[0]
    fake._delta_pages = [single_page]
    fake._delta_pages_backup = [single_page]
    coordinator, sink, checkpoint_repo, state_repo, fake, integration = (
        _build_coordinator(
            fake,
            calendar=_default_calendar(),
            binding_id=_PRIMARY_BINDING_ID,
            safe_display_name="Primary Calendar",
        )
    )
    integration_id = id(integration)
    operation_id = "msgraph-calendar-sink-retry"
    sink.fail_times = 1
    with pytest.raises(VendorKnowledgeError):
        await coordinator.reconcile_once(
            binding_id=_PRIMARY_BINDING_ID,
            restart=True,
            operation_id=operation_id,
        )
    first_delivery = sink.calls[0].delivery_id
    first_delta_calls = len(fake.delta_calls)
    result = await coordinator.reconcile_once(
        binding_id=_PRIMARY_BINDING_ID,
        restart=True,
        operation_id=operation_id,
    )
    assert result.delivery_id == first_delivery
    assert sink.calls[1].delivery_id == first_delivery
    assert len(sink.durable_delivery_ids) == 1
    assert sink.durable_delivery_ids[0] == first_delivery
    assert len(fake.delta_calls) == first_delta_calls + 1
    assert fake.delta_calls[0] == fake.delta_calls[first_delta_calls]
    checkpoint = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id=_PRIMARY_BINDING_ID,
    )
    assert checkpoint is not None
    assert checkpoint.cursor is not None
    assert (
        state_repo.get(
            tenant_id="tenant-1",
            binding_id=_PRIMARY_BINDING_ID,
            remote_id=_encode_event_remote_id(
                calendar_remote_id=_DEFAULT_CALENDAR_ID,
                event_remote_id=_EVT_1,
            ),
        )
        is not None
    )
    assert id(integration) == integration_id


@pytest.mark.asyncio
async def test_msgraph_calendar_primary_item_state_failure_retries_without_duplicate_sink_effect() -> (
    None
):
    fake = _primary_fake(
        delta_pages=[
            _delta_page(
                calendar_remote_id=_DEFAULT_CALENDAR_ID,
                items=(
                    _active_event(
                        calendar_remote_id=_DEFAULT_CALENDAR_ID,
                        remote_id=_EVT_1,
                        change_key=_CK_1,
                    ),
                ),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_CHECKPOINT_URL,
            )
        ]
    )
    state_inner = InMemoryRemoteItemStateRepository()
    state_repo = _FailingItemStateRepository(state_inner)
    sink = IdempotentRecordingSink()
    coordinator, sink, checkpoint_repo, _state_repo, fake, integration = (
        _build_coordinator(
            fake,
            calendar=_default_calendar(),
            binding_id=_PRIMARY_BINDING_ID,
            safe_display_name="Primary Calendar",
            item_state_repository=state_repo,
            sink=sink,
            sink_receipt_inspector=_SinkBackedReceiptInspector(sink),
        )
    )
    integration_id = id(integration)
    operation_id = "msgraph-calendar-item-state-retry"
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id=_PRIMARY_BINDING_ID,
            restart=True,
            operation_id=operation_id,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    first_delivery = sink.calls[0].delivery_id
    assert sink.durable_delivery_ids == [first_delivery]
    assert (
        checkpoint_repo.get(
            tenant_id="tenant-1",
            binding_id=_PRIMARY_BINDING_ID,
        )
        is None
    )
    assert state_inner.applied_delivery_ids == set()

    result = await coordinator.reconcile_once(
        binding_id=_PRIMARY_BINDING_ID,
        restart=True,
        operation_id=operation_id,
    )
    assert result.status is KnowledgeSyncRunStatus.COMPLETED
    assert result.delivery_id == first_delivery
    assert sink.durable_delivery_ids == [first_delivery]
    assert len(sink.calls) == 1
    assert sink.calls[0].delivery_id == result.delivery_id
    assert len(fake.delta_calls) == 1
    assert state_inner.applied_delivery_ids == {first_delivery}
    assert (
        checkpoint_repo.get(
            tenant_id="tenant-1",
            binding_id=_PRIMARY_BINDING_ID,
        )
        is not None
    )
    assert id(integration) == integration_id


@pytest.mark.asyncio
async def test_msgraph_calendar_checkpoint_failure_retries_finalizing_without_duplicate_effects() -> (
    None
):
    fake = _primary_fake(
        delta_pages=[
            _delta_page(
                calendar_remote_id=_DEFAULT_CALENDAR_ID,
                items=(
                    _active_event(
                        calendar_remote_id=_DEFAULT_CALENDAR_ID,
                        remote_id=_EVT_1,
                        change_key=_CK_1,
                    ),
                ),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_CHECKPOINT_URL,
            )
        ]
    )
    document_store = InMemoryDocumentStore()
    checkpoint_inner = InMemoryCheckpointRepository()
    checkpoint_repo = _FailingCheckpointRepository(checkpoint_inner)
    coordinator, sink, _checkpoint_repo, state_repo, fake, integration = (
        _build_coordinator(
            fake,
            calendar=_default_calendar(),
            binding_id=_PRIMARY_BINDING_ID,
            safe_display_name="Primary Calendar",
            document_store=document_store,
            checkpoint_repository=checkpoint_repo,
        )
    )
    integration_id = id(integration)
    operation_id = "msgraph-calendar-checkpoint-retry"
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id=_PRIMARY_BINDING_ID,
            restart=True,
            operation_id=operation_id,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    first_delivery = sink.calls[0].delivery_id
    assert checkpoint_repo.successful_commits == 0
    assert checkpoint_repo.fail_times == 0
    assert sink.durable_delivery_ids == [first_delivery]
    assert (
        checkpoint_repo.get(
            tenant_id="tenant-1",
            binding_id=_PRIMARY_BINDING_ID,
        )
        is None
    )
    assert fake.delta_calls

    result = await coordinator.reconcile_once(
        binding_id=_PRIMARY_BINDING_ID,
        restart=True,
        operation_id=operation_id,
    )
    assert result.status is KnowledgeSyncRunStatus.COMPLETED
    assert result.delivery_id == first_delivery
    assert checkpoint_repo.successful_commits == 1
    assert sink.durable_delivery_ids == [first_delivery]
    assert len(sink.calls) == 1
    assert sink.calls[0].delivery_id == result.delivery_id
    assert len(fake.delta_calls) == 1
    assert (
        state_repo.get(
            tenant_id="tenant-1",
            binding_id=_PRIMARY_BINDING_ID,
            remote_id=_encode_event_remote_id(
                calendar_remote_id=_DEFAULT_CALENDAR_ID,
                event_remote_id=_EVT_1,
            ),
        )
        is not None
    )
    assert id(integration) == integration_id


@pytest.mark.asyncio
async def test_msgraph_calendar_primary_no_attachment_bytes_in_sink() -> None:
    calendar_remote_id = _DEFAULT_CALENDAR_ID
    content = _event_content(
        calendar_remote_id=calendar_remote_id,
        remote_id=_EVT_1,
        content_revision=_CK_1,
        subject="Event with attachment",
        body_content="body-with-attachment",
        has_attachments=True,
    )
    fake = _primary_fake(
        delta_pages=[
            _delta_page(
                calendar_remote_id=calendar_remote_id,
                items=(
                    _active_event(
                        calendar_remote_id=calendar_remote_id,
                        remote_id=_EVT_1,
                        change_key=_CK_1,
                        has_attachments=True,
                    ),
                ),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_CHECKPOINT_URL,
            ),
        ],
        incremental_delta_page=None,
        content={(_EVT_1, _CK_1): content},
        attachment_pages={
            (_EVT_1, _CK_1): _attachment_page(
                calendar_remote_id=calendar_remote_id,
                event_remote_id=_EVT_1,
                event_revision=_CK_1,
            ),
        },
    )
    coordinator, sink, _checkpoint_repo, _state_repo, fake, integration = (
        _build_coordinator(
            fake,
            calendar=_default_calendar(),
            binding_id=_PRIMARY_BINDING_ID,
            safe_display_name="Primary Calendar",
        )
    )
    await coordinator.reconcile_once(
        binding_id=_PRIMARY_BINDING_ID,
        restart=True,
        operation_id="msgraph-calendar-attachments",
    )
    assert len(fake.attachment_calls) == 1
    assert "attachment_bytes" not in fake.forbidden_calls

    envelope = sink.calls[0].envelopes[0]
    assert envelope.content is not None
    record = envelope.content.structured_record
    assert record is not None
    attachments = record["attachments"]
    assert attachments["attachment_inventory_included"] is True
    assert attachments["attachment_binary_content_included"] is False
    assert len(attachments["items"]) == 1
    assert attachments["items"][0]["attachment_remote_id"] == _ATTACHMENT_ID

    sink_blob = _public_blob(sink.calls)
    assert "attachment_bytes" not in sink_blob
    assert "b64" not in sink_blob.lower()
    _assert_no_secrets(sink_blob)


@pytest.mark.asyncio
async def test_msgraph_calendar_incomplete_attachment_inventory_is_not_claimed_complete() -> (
    None
):
    calendar_remote_id = _DEFAULT_CALENDAR_ID
    content = _event_content(
        calendar_remote_id=calendar_remote_id,
        remote_id=_EVT_1,
        content_revision=_CK_1,
        subject="Event with paged attachments",
        body_content="body-with-paged-attachments",
        has_attachments=True,
    )
    attachment_page = _attachment_page(
        calendar_remote_id=calendar_remote_id,
        event_remote_id=_EVT_1,
        event_revision=_CK_1,
    ).model_copy(
        update={
            "continuation": MsGraphKnowledgeContinuation(
                kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=(
                    f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
                    f"{quote(_DEFAULT_CALENDAR_ID, safe='')}/events/{quote(_EVT_1, safe='')}"
                    f"/attachments?$skiptoken={_SECRET_SKIP}"
                ),
            )
        }
    )
    fake = _primary_fake(
        delta_pages=[
            _delta_page(
                calendar_remote_id=calendar_remote_id,
                items=(
                    _active_event(
                        calendar_remote_id=calendar_remote_id,
                        remote_id=_EVT_1,
                        change_key=_CK_1,
                        has_attachments=True,
                    ),
                ),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_CHECKPOINT_URL,
            )
        ],
        incremental_delta_page=None,
        content={(_EVT_1, _CK_1): content},
        attachment_pages={(_EVT_1, _CK_1): attachment_page},
    )
    coordinator, sink, checkpoint_repo, state_repo, fake, integration = (
        _build_coordinator(
            fake,
            calendar=_default_calendar(),
            binding_id=_PRIMARY_BINDING_ID,
            safe_display_name="Primary Calendar",
        )
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id=_PRIMARY_BINDING_ID,
            restart=True,
            operation_id="msgraph-calendar-incomplete-attachments",
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert exc_info.value.retryable is False
    assert sink.calls == []
    assert (
        checkpoint_repo.get(
            tenant_id="tenant-1",
            binding_id=_PRIMARY_BINDING_ID,
        )
        is None
    )
    assert (
        state_repo.get(
            tenant_id="tenant-1",
            binding_id=_PRIMARY_BINDING_ID,
            remote_id=_encode_event_remote_id(
                calendar_remote_id=calendar_remote_id,
                event_remote_id=_EVT_1,
            ),
        )
        is None
    )
    assert "attachment_bytes" not in fake.forbidden_calls
    assert integration is not None
