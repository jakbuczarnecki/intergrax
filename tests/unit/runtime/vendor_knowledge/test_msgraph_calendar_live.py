from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.live import (
    EffectiveLiveCallBudgetV1,
    KnowledgeQueryAudienceV1,
    LiveCapabilityExecutionContextV1,
    LiveCapabilityExecutionResultV1,
    LiveExecutionOutcomeV1,
    LiveResultRetentionV1,
    ValidatedLiveCapabilityCallV1,
)
from intergrax.runtime.vendor_knowledge.live.identity import (
    LiveOperationV1,
    parse_capability_id,
)
from intergrax.runtime.vendor_knowledge.live.ms365_graph import (
    MSGRAPH_CALENDAR_LIST_CAPABILITY_ID,
    MSGRAPH_CALENDAR_LIST_REQUEST_SCHEMA_REF,
    MSGRAPH_CALENDAR_LIST_RESULT_SCHEMA_REF,
    MsGraphCalendarListLiveHandlerV1,
    MsGraphCalendarListLiveRequestV1,
    build_msgraph_calendar_list_descriptor,
    build_msgraph_live_registration_bundles,
)
from intergrax.runtime.vendor_knowledge.live.registration import (
    publish_live_registration_bundles,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeChange,
    KnowledgeChangeKind,
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgeItemIdentity,
    KnowledgeItemProvenance,
    KnowledgeItemRevision,
    KnowledgePage,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)
_SOURCE_KIND = "calendar"
_DEFAULT_SCOPE = "opaque-default-calendar-scope"
_NON_DEFAULT_SCOPE = "opaque-non-default-calendar-scope"
_REMOVAL_SEMANTICS = "removed_from_synchronized_calendar_window_view"


class _RoutingCalendarAdapter:
    def __init__(self, pages: dict[str, KnowledgePage]) -> None:
        self.pages = pages
        self.calls: list[dict[str, object]] = []

    async def read_page(
        self,
        *,
        integration: object,
        source: object,
        cursor: object,
        limit: int,
    ) -> KnowledgePage:
        scope_id = source.scope.remote_scope_id  # type: ignore[union-attr]
        self.calls.append(
            {
                "integration": integration,
                "source": source,
                "cursor": cursor,
                "limit": limit,
                "route": "primary-delta"
                if scope_id == _DEFAULT_SCOPE
                else "snapshot",
            }
        )
        return self.pages[scope_id]


class _ErrorCalendarAdapter:
    async def read_page(self, **_kwargs: object) -> KnowledgePage:
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.AUTHORIZATION_DENIED,
            safe_message="safe",
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=_SOURCE_KIND,
            retryable=False,
        )


def _integration() -> Ms365GraphCollaborationSuiteIntegration:
    return Ms365GraphCollaborationSuiteIntegration.from_client(object())


def _request(page_size: int = 25) -> MsGraphCalendarListLiveRequestV1:
    return MsGraphCalendarListLiveRequestV1(page_size=page_size)


def _call(
    request: object | None = None,
    *,
    remote_resource_id: str | None = _DEFAULT_SCOPE,
    max_result_items: int = 200,
    max_upstream_items: int = 200,
    max_provider_page_size: int = 100,
    max_result_bytes: int = 10_000,
    max_content_bytes_per_item: int = 4_096,
    **identity_overrides: object,
) -> ValidatedLiveCapabilityCallV1:
    values: dict[str, Any] = {
        "call_id": "call-calendar-1",
        "capability_id": MSGRAPH_CALENDAR_LIST_CAPABILITY_ID,
        "contract_version": "1",
        "connection_ref": "connection-1",
        "live_access_binding_id": "binding-1",
        "remote_resource_id": remote_resource_id,
        "validated_request": request or _request(),
        "effective_budget": EffectiveLiveCallBudgetV1(
            max_live_calls=1,
            max_total_duration_ms=10_000,
            max_result_items=max_result_items,
            max_result_bytes=max_result_bytes,
            max_provider_pages=1,
            max_provider_requests=1,
            max_upstream_items=max_upstream_items,
            max_provider_page_size=max_provider_page_size,
            max_content_bytes_per_item=max_content_bytes_per_item,
        ),
        "audience_context_ref": None,
        "provider_id": MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        "integration_kind": IntegrationCategory.COLLABORATION_SUITE,
        "source_kind": _SOURCE_KIND,
    }
    values.update(identity_overrides)
    return ValidatedLiveCapabilityCallV1(**values)


def _context() -> LiveCapabilityExecutionContextV1:
    return LiveCapabilityExecutionContextV1(
        run_id="run-1",
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        audience=KnowledgeQueryAudienceV1.PERSONAL,
        started_at=_NOW,
        deadline_monotonic=100,
        retention=LiveResultRetentionV1.RECEIPT_ONLY,
    )


def _active_change(remote_id: str = "opaque-event-1") -> KnowledgeChange:
    descriptor = KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id=remote_id),
        revision=KnowledgeItemRevision(version="revision-1", updated_at=_NOW),
        title="Calendar event",
        item_type="msgraph_calendar_event",
        content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=_SOURCE_KIND,
            remote_id=remote_id,
            web_url="https://example.test/events/1",
        ),
        metadata={
            "event_state": "active",
            "event_type": "single_instance",
            "start_at": "2026-08-05T10:00:00+00:00",
            "end_at": "2026-08-05T11:00:00+00:00",
            "original_start_at": None,
            "last_modified_at": "2026-08-05T11:00:00+00:00",
            "series_master_id": None,
            "i_cal_uid": "ical-1",
            "is_all_day": False,
            "is_cancelled": False,
            "is_draft": False,
            "has_attachments": True,
            "is_online_meeting": True,
            "removal_semantics": _REMOVAL_SEMANTICS,
        },
    )
    return KnowledgeChange(
        kind=KnowledgeChangeKind.UPSERT,
        remote_id=remote_id,
        descriptor=descriptor,
    )


def _removed_change(remote_id: str = "opaque-event-removed") -> KnowledgeChange:
    return KnowledgeChange(kind=KnowledgeChangeKind.DELETED, remote_id=remote_id)


def test_calendar_registration_is_canonical_and_ordered() -> None:
    bundles = build_msgraph_live_registration_bundles()
    published = publish_live_registration_bundles(bundles)
    assert tuple(bundle.descriptor.capability_id for bundle in bundles) == (
        "vendor.ms365_graph.drive.list",
        "vendor.ms365_graph.mail.list",
        "vendor.ms365_graph.teams_channel.list",
        "vendor.ms365_graph.teams_chat.list",
        MSGRAPH_CALENDAR_LIST_CAPABILITY_ID,
    )
    provider_id, source_kind, operation = parse_capability_id(
        MSGRAPH_CALENDAR_LIST_CAPABILITY_ID
    )
    assert provider_id == MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    assert source_kind == _SOURCE_KIND
    assert operation is LiveOperationV1.LIST
    assert published.schemas.resolve_request(
        MSGRAPH_CALENDAR_LIST_REQUEST_SCHEMA_REF, "1"
    ) is MsGraphCalendarListLiveRequestV1
    assert published.schemas.resolve_result(
        MSGRAPH_CALENDAR_LIST_RESULT_SCHEMA_REF, "1"
    ) is LiveCapabilityExecutionResultV1
    assert len(published.handlers) == 5
    for operation_name in ("search", "read", "content.read", "thread.read"):
        with pytest.raises(LookupError, match="live_capability_unavailable"):
            published.resolve_handler(
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                capability_id=f"vendor.ms365_graph.calendar.{operation_name}",
                contract_version="1",
            )


def test_calendar_request_and_descriptor_reject_scope_strategy_controls() -> None:
    request = _request(200)
    descriptor = build_msgraph_calendar_list_descriptor()
    assert request.page_size == 200
    assert descriptor.max_result_items == 200
    assert descriptor.max_upstream_items == 200
    assert descriptor.max_provider_page_size == 200
    with pytest.raises(ValidationError):
        MsGraphCalendarListLiveRequestV1.model_validate(
            {"page_size": 201, "sync_strategy": "snapshot"}
        )
    with pytest.raises(ValidationError):
        MsGraphCalendarListLiveRequestV1.model_validate(
            {"calendar_remote_id": "other", "window_start_at": "secret"}
        )
    with pytest.raises(ValidationError):
        request.page_size = 25


@pytest.mark.asyncio
async def test_calendar_handler_preserves_opaque_scope_and_adapter_strategy() -> None:
    adapter = _RoutingCalendarAdapter(
        {
            _DEFAULT_SCOPE: KnowledgePage(
                changes=(_active_change(),),
                next_cursor=KnowledgeCursor(value="delta-secret", version="calendar"),
                has_more=True,
            ),
            _NON_DEFAULT_SCOPE: KnowledgePage(
                changes=(_removed_change(),),
                has_more=False,
            ),
        }
    )
    handler = MsGraphCalendarListLiveHandlerV1(adapter=adapter)
    primary = await handler.execute(
        integration=_integration(),
        call=_call(_request(200), max_result_items=100, max_upstream_items=50, max_provider_page_size=32),
        context=_context(),
    )
    non_primary = await handler.execute(
        integration=_integration(),
        call=_call(_request(), remote_resource_id=_NON_DEFAULT_SCOPE),
        context=_context(),
    )
    assert primary.normalized_outcome is LiveExecutionOutcomeV1.TRUNCATED
    assert primary.truncated is True
    assert non_primary.normalized_outcome is LiveExecutionOutcomeV1.COMPLETED
    assert non_primary.truncated is False
    assert len(adapter.calls) == 2
    assert adapter.calls[0]["route"] == "primary-delta"
    assert adapter.calls[1]["route"] == "snapshot"
    assert adapter.calls[0]["cursor"] is None
    assert adapter.calls[1]["cursor"] is None
    assert adapter.calls[0]["limit"] == 32
    assert adapter.calls[0]["source"].scope.remote_scope_id == _DEFAULT_SCOPE  # type: ignore[union-attr]
    assert adapter.calls[1]["source"].scope.remote_scope_id == _NON_DEFAULT_SCOPE  # type: ignore[union-attr]
    assert primary.items[0].safe_locator == "https://example.test/events/1"
    assert non_primary.items[0].content == (
        '{"change_kind":"deleted","content_available":false,'
        '"event_state":"removed","item_type":"deleted",'
        f'"removal_semantics":"{_REMOVAL_SEMANTICS}"}}'
    )
    assert "delta-secret" not in primary.model_dump_json()


@pytest.mark.asyncio
async def test_calendar_mapping_is_metadata_only_deterministic_and_hashable() -> None:
    page = KnowledgePage(changes=(_active_change(),), has_more=False)
    first = await MsGraphCalendarListLiveHandlerV1(
        adapter=_RoutingCalendarAdapter({_DEFAULT_SCOPE: page})
    ).execute(
        integration=_integration(),
        call=_call(),
        context=_context(),
    )
    second = await MsGraphCalendarListLiveHandlerV1(
        adapter=_RoutingCalendarAdapter({_DEFAULT_SCOPE: page})
    ).execute(
        integration=_integration(),
        call=_call(),
        context=_context(),
    )
    assert first.normalized_outcome is LiveExecutionOutcomeV1.COMPLETED
    assert first.items[0].content == second.items[0].content
    assert first.items[0].content_hash == hashlib.sha256(
        first.items[0].content.encode("utf-8")
    ).hexdigest()
    assert "body" not in first.items[0].content
    assert "attendees" not in first.items[0].content
    assert '"attachments":' not in first.items[0].content
    assert "join_url" not in first.items[0].content
    assert "strategy" not in first.items[0].content
    assert "cursor" not in first.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "page",
    [
        KnowledgePage(changes=(_removed_change("same"), _removed_change("same"))),
        KnowledgePage(changes=tuple(_removed_change(str(index)) for index in range(6))),
    ],
)
async def test_calendar_handler_rejects_duplicate_or_oversized_pages(
    page: KnowledgePage,
) -> None:
    result = await MsGraphCalendarListLiveHandlerV1(
        adapter=_RoutingCalendarAdapter({_DEFAULT_SCOPE: page})
    ).execute(
        integration=_integration(),
        call=_call(_request(5)),
        context=_context(),
    )
    assert result.error_code == "live_provider_contract_violation"
    assert result.items == ()


@pytest.mark.asyncio
async def test_calendar_handler_maps_error_and_does_not_convert_absence_to_deletion() -> None:
    error = await MsGraphCalendarListLiveHandlerV1(
        adapter=_ErrorCalendarAdapter()
    ).execute(
        integration=_integration(),
        call=_call(),
        context=_context(),
    )
    empty = await MsGraphCalendarListLiveHandlerV1(
        adapter=_RoutingCalendarAdapter({_DEFAULT_SCOPE: KnowledgePage()})
    ).execute(
        integration=_integration(),
        call=_call(),
        context=_context(),
    )
    assert error.error_code == "live_provider_forbidden"
    assert empty.normalized_outcome is LiveExecutionOutcomeV1.COMPLETED
    assert empty.items == ()
