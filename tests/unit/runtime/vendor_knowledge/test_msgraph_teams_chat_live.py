from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from local_workspace_application.workspaces.hybrid_ask_policy import (
    ExecutableLiveCallV1,
    ResolvedLiveResourceScopeV1,
)

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
)
from intergrax.runtime.vendor_knowledge.live.identity import (
    LiveOperationV1,
    parse_capability_id,
)
from intergrax.runtime.vendor_knowledge.live.ms365_graph import (
    MSGRAPH_TEAMS_CHAT_LIST_CAPABILITY_ID,
    MSGRAPH_TEAMS_CHAT_LIST_REQUEST_SCHEMA_REF,
    MSGRAPH_TEAMS_CHAT_LIST_RESULT_SCHEMA_REF,
    MsGraphTeamsChatListLiveHandlerV1,
    MsGraphTeamsChatListLiveRequestV1,
    build_msgraph_live_registration_bundles,
    build_msgraph_teams_chat_list_descriptor,
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
_SOURCE_KIND = "teams_chat"
_SCOPE = "opaque-fixed-chat-window-scope"


class _FakeTeamsChatAdapter:
    def __init__(self, page: KnowledgePage) -> None:
        self.page = page
        self.calls: list[dict[str, object]] = []

    async def read_page(
        self,
        *,
        integration: object,
        source: object,
        cursor: object,
        limit: int,
    ) -> KnowledgePage:
        self.calls.append(
            {
                "integration": integration,
                "source": source,
                "cursor": cursor,
                "limit": limit,
            }
        )
        return self.page


class _ErrorTeamsChatAdapter:
    async def read_page(self, **_kwargs: object) -> KnowledgePage:
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.RATE_LIMITED,
            safe_message="safe",
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=_SOURCE_KIND,
            retryable=False,
        )


def _integration() -> Ms365GraphCollaborationSuiteIntegration:
    return Ms365GraphCollaborationSuiteIntegration.from_client(object())


def _request(page_size: int = 25) -> MsGraphTeamsChatListLiveRequestV1:
    return MsGraphTeamsChatListLiveRequestV1(page_size=page_size)


def _call(
    request: object | None = None,
    *,
    remote_resource_id: str | None = _SCOPE,
    scope_token: str | None = _SCOPE,
    max_result_items: int = 10,
    max_upstream_items: int = 10,
    max_provider_page_size: int = 10,
    max_result_bytes: int = 10_000,
    max_content_bytes_per_item: int = 4_096,
    **identity_overrides: object,
) -> ExecutableLiveCallV1:
    values: dict[str, Any] = {
        "call_id": "call-teams-chat-1",
        "capability_id": MSGRAPH_TEAMS_CHAT_LIST_CAPABILITY_ID,
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
    return ExecutableLiveCallV1(
        **values,
        resolved_resource_scope=ResolvedLiveResourceScopeV1(
            remote_resource_id=remote_resource_id,
            scope_token=scope_token,
        ),
    )


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


def _active_change(remote_id: str = "opaque-message-1") -> KnowledgeChange:
    descriptor = KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id=remote_id),
        revision=KnowledgeItemRevision(version="revision-1", updated_at=_NOW),
        title="Sprint planning",
        item_type="msgraph_teams_chat_message",
        content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=_SOURCE_KIND,
            remote_id=remote_id,
            web_url="https://example.test/messages/1",
        ),
        metadata={
            "message_state": "active",
            "message_type": "message",
            "importance": "normal",
            "body_kind": "text",
            "has_attachments": True,
            "created_at": "2026-08-05T10:00:00+00:00",
            "last_modified_at": "2026-08-05T11:00:00+00:00",
            "last_edited_at": None,
            "event_detail_type": None,
            "locale": None,
            "attachment_inventory_in_content": True,
            "attachment_binary_content_included": False,
            "hosted_content_included": False,
            "reference_urls_included": False,
        },
    )
    return KnowledgeChange(
        kind=KnowledgeChangeKind.UPSERT,
        remote_id=remote_id,
        descriptor=descriptor,
    )


def _deleted_change(remote_id: str = "opaque-message-deleted") -> KnowledgeChange:
    return KnowledgeChange(kind=KnowledgeChangeKind.DELETED, remote_id=remote_id)


def test_teams_chat_registration_is_canonical_and_ordered() -> None:
    bundles = build_msgraph_live_registration_bundles()
    published = publish_live_registration_bundles(bundles)
    assert tuple(bundle.descriptor.capability_id for bundle in bundles) == (
        "vendor.ms365_graph.drive.list",
        "vendor.ms365_graph.mail.list",
        "vendor.ms365_graph.teams_channel.list",
        MSGRAPH_TEAMS_CHAT_LIST_CAPABILITY_ID,
        "vendor.ms365_graph.calendar.list",
    )
    provider_id, source_kind, operation = parse_capability_id(
        MSGRAPH_TEAMS_CHAT_LIST_CAPABILITY_ID
    )
    assert provider_id == MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    assert source_kind == _SOURCE_KIND
    assert operation is LiveOperationV1.LIST
    assert published.schemas.resolve_request(
        MSGRAPH_TEAMS_CHAT_LIST_REQUEST_SCHEMA_REF, "1"
    ) is MsGraphTeamsChatListLiveRequestV1
    assert published.schemas.resolve_result(
        MSGRAPH_TEAMS_CHAT_LIST_RESULT_SCHEMA_REF, "1"
    ) is LiveCapabilityExecutionResultV1
    assert len(published.handlers) == 5
    for operation_name in ("search", "read", "thread.read", "content.read"):
        with pytest.raises(LookupError, match="live_capability_unavailable"):
            published.resolve_handler(
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                capability_id=f"vendor.ms365_graph.teams_chat.{operation_name}",
                contract_version="1",
            )


def test_teams_chat_request_and_descriptor_are_bounded() -> None:
    request = _request(50)
    assert request.page_size == 50
    assert build_msgraph_teams_chat_list_descriptor().max_result_items == 50
    with pytest.raises(ValidationError):
        MsGraphTeamsChatListLiveRequestV1.model_validate(
            {"page_size": 51, "chat_remote_id": "other"}
        )
    with pytest.raises(ValidationError):
        MsGraphTeamsChatListLiveRequestV1.model_validate({"cursor": "secret"})
    with pytest.raises(ValidationError):
        request.page_size = 25


@pytest.mark.asyncio
async def test_teams_chat_handler_uses_opaque_scope_once_and_maps_metadata() -> None:
    page = KnowledgePage(
        changes=(_active_change(), _deleted_change()),
        next_cursor=KnowledgeCursor(value="must-not-leak", version="chat"),
        proposed_checkpoint=KnowledgeCursor(value="must-not-leak", version="chat"),
        has_more=True,
    )
    adapter = _FakeTeamsChatAdapter(page)
    result = await MsGraphTeamsChatListLiveHandlerV1(adapter=adapter).execute(
        integration=_integration(),
        call=_call(_request(50), max_result_items=10, max_upstream_items=8, max_provider_page_size=7),
        context=_context(),
    )
    assert result.normalized_outcome is LiveExecutionOutcomeV1.TRUNCATED
    assert result.truncated is True
    assert result.item_count == 2
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["cursor"] is None
    assert adapter.calls[0]["limit"] == 7
    assert adapter.calls[0]["source"].scope.remote_scope_id == _SCOPE  # type: ignore[union-attr]
    assert result.items[0].safe_locator == "https://example.test/messages/1"
    assert result.items[0].retrieved_at == result.items[1].retrieved_at
    assert result.items[0].content_hash == hashlib.sha256(
        result.items[0].content.encode("utf-8")
    ).hexdigest()
    assert "body_content" not in result.items[0].content
    assert "mentions" not in result.items[0].content
    assert "reactions" not in result.items[0].content
    assert '"attachments":' not in result.items[0].content
    assert "opaque-message" in result.items[0].remote_item_id
    assert "must-not-leak" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_teams_chat_handler_prefers_resolved_scope_token_over_legacy_remote_id() -> None:
    scope_token = "resolved-scope-token"
    adapter = _FakeTeamsChatAdapter(KnowledgePage(changes=(_active_change(),)))
    result = await MsGraphTeamsChatListLiveHandlerV1(adapter=adapter).execute(
        integration=_integration(),
        call=_call(scope_token=scope_token),
        context=_context(),
    )
    assert result.normalized_outcome is LiveExecutionOutcomeV1.COMPLETED
    assert adapter.calls[0]["source"].scope.remote_scope_id == scope_token  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_teams_chat_handler_is_deterministic_and_completed_without_continuation() -> None:
    page = KnowledgePage(changes=(_active_change(),), has_more=False)
    first = await MsGraphTeamsChatListLiveHandlerV1(
        adapter=_FakeTeamsChatAdapter(page)
    ).execute(
        integration=_integration(),
        call=_call(_request()),
        context=_context(),
    )
    second = await MsGraphTeamsChatListLiveHandlerV1(
        adapter=_FakeTeamsChatAdapter(page)
    ).execute(
        integration=_integration(),
        call=_call(_request()),
        context=_context(),
    )
    assert first.normalized_outcome is LiveExecutionOutcomeV1.COMPLETED
    assert first.truncated is False
    assert first.items[0].content == second.items[0].content
    assert first.items[0].content_hash == second.items[0].content_hash
    assert "next_cursor" not in LiveCapabilityExecutionResultV1.model_fields


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "page",
    [
        KnowledgePage(changes=(_deleted_change("same"), _deleted_change("same"))),
        KnowledgePage(changes=tuple(_deleted_change(str(index)) for index in range(8))),
    ],
)
async def test_teams_chat_handler_rejects_duplicate_or_oversized_pages(
    page: KnowledgePage,
) -> None:
    result = await MsGraphTeamsChatListLiveHandlerV1(
        adapter=_FakeTeamsChatAdapter(page)
    ).execute(
        integration=_integration(),
        call=_call(_request(5)),
        context=_context(),
    )
    assert result.error_code == "live_provider_contract_violation"
    assert result.items == ()


@pytest.mark.asyncio
async def test_teams_chat_handler_maps_shared_errors_and_rejects_scope_before_adapter() -> None:
    error = await MsGraphTeamsChatListLiveHandlerV1(
        adapter=_ErrorTeamsChatAdapter()
    ).execute(
        integration=_integration(),
        call=_call(),
        context=_context(),
    )
    missing_scope = _FakeTeamsChatAdapter(KnowledgePage())
    invalid = await MsGraphTeamsChatListLiveHandlerV1(adapter=missing_scope).execute(
        integration=_integration(),
        call=_call(remote_resource_id=None),
        context=_context(),
    )
    assert error.error_code == "live_provider_throttled"
    assert invalid.error_code == "live_resource_scope_invalid"
    assert missing_scope.calls == []
