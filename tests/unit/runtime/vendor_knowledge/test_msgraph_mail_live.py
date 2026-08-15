from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

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
from intergrax.runtime.vendor_knowledge.live.ms365_graph import (
    MSGRAPH_MAIL_LIST_CAPABILITY_ID,
    MSGRAPH_MAIL_LIST_REQUEST_SCHEMA_REF,
    MSGRAPH_MAIL_LIST_RESULT_SCHEMA_REF,
    MsGraphMailListLiveHandlerV1,
    MsGraphMailListLiveRequestV1,
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

_SOURCE_KIND = "mail"
_SCOPE = "opaque-mailbox-folder-scope"
_STARTED_AT = datetime(2026, 1, 1, tzinfo=UTC)
_UPDATED_AT = datetime(2026, 1, 2, tzinfo=UTC)


class _WrongRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    value: int = 1


class _FakeMailAdapter:
    def __init__(self, page: KnowledgePage) -> None:
        self.page = page
        self.calls: list[tuple[object, object, object, object]] = []

    async def read_page(self, *, integration, source, cursor, limit):
        self.calls.append((integration, source, cursor, limit))
        return self.page


class _ErrorMailAdapter:
    def __init__(self, error_code: VendorKnowledgeErrorCode) -> None:
        self.error_code = error_code

    async def read_page(self, **_kwargs):
        raise VendorKnowledgeError(
            code=self.error_code,
            safe_message="safe",
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=_SOURCE_KIND,
            retryable=False,
        )


class _CancelledMailAdapter:
    async def read_page(self, **_kwargs):
        raise asyncio.CancelledError


def _integration() -> Ms365GraphCollaborationSuiteIntegration:
    return Ms365GraphCollaborationSuiteIntegration.from_client(object())


def _request(page_size: int = 25) -> MsGraphMailListLiveRequestV1:
    return MsGraphMailListLiveRequestV1(page_size=page_size)


def _call(
    request: object,
    *,
    remote_resource_id: str | None = _SCOPE,
    **identity_overrides: object,
) -> ValidatedLiveCapabilityCallV1:
    values = {
        "call_id": "call-mail-1",
        "capability_id": MSGRAPH_MAIL_LIST_CAPABILITY_ID,
        "contract_version": "1",
        "connection_ref": "connection-1",
        "live_access_binding_id": "binding-1",
        "remote_resource_id": remote_resource_id,
        "validated_request": request,
        "effective_budget": EffectiveLiveCallBudgetV1(
            max_live_calls=1,
            max_total_duration_ms=10_000,
            max_result_items=10,
            max_result_bytes=10_000,
            max_provider_pages=1,
            max_provider_requests=1,
            max_upstream_items=10,
            max_provider_page_size=20,
            max_content_bytes_per_item=4_096,
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
        started_at=_STARTED_AT,
        deadline_monotonic=100,
        retention=LiveResultRetentionV1.RECEIPT_ONLY,
    )


def _active_change(remote_id: str = "opaque-message-1") -> KnowledgeChange:
    descriptor = KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id=remote_id),
        revision=KnowledgeItemRevision(
            version="opaque-revision-1",
            updated_at=_UPDATED_AT,
        ),
        title="Quarterly report",
        item_type="msgraph_mail_message",
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
            "is_read": True,
            "is_draft": False,
            "has_attachments": True,
            "importance": "normal",
            "attachment_inventory_included": False,
            "attachment_content_included": False,
            "removal_semantics": "removed_from_synchronized_folder_view",
            "created_at": "2026-01-01T00:00:00+00:00",
            "received_at": "2026-01-01T01:00:00+00:00",
            "sent_at": "2026-01-01T01:00:00+00:00",
            "last_modified_at": "2026-01-02T00:00:00+00:00",
        },
    )
    return KnowledgeChange(
        kind=KnowledgeChangeKind.UPSERT,
        remote_id=remote_id,
        descriptor=descriptor,
    )


def _removed_change(remote_id: str = "opaque-message-removed") -> KnowledgeChange:
    return KnowledgeChange(kind=KnowledgeChangeKind.DELETED, remote_id=remote_id)


@pytest.mark.asyncio
async def test_mail_registration_is_canonical_combined_and_duplicate_safe() -> None:
    bundles = build_msgraph_live_registration_bundles()
    assert len(bundles) == 5
    published = publish_live_registration_bundles(bundles)
    assert {key[2] for key in published.descriptors} == {
        "vendor.ms365_graph.drive.list",
        MSGRAPH_MAIL_LIST_CAPABILITY_ID,
        "vendor.ms365_graph.teams_channel.list",
        "vendor.ms365_graph.teams_chat.list",
        "vendor.ms365_graph.calendar.list",
    }
    descriptor = published.resolve_handler(
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        capability_id=MSGRAPH_MAIL_LIST_CAPABILITY_ID,
        contract_version="1",
    )
    assert isinstance(descriptor, MsGraphMailListLiveHandlerV1)
    assert (
        published.schemas.resolve_request(MSGRAPH_MAIL_LIST_REQUEST_SCHEMA_REF, "1")
        is MsGraphMailListLiveRequestV1
    )
    assert (
        published.schemas.resolve_result(MSGRAPH_MAIL_LIST_RESULT_SCHEMA_REF, "1")
        is LiveCapabilityExecutionResultV1
    )
    assert bundles[1].descriptor.max_provider_pages == 1
    assert bundles[1].descriptor.max_provider_requests == 1
    with pytest.raises(ValueError, match="duplicate_live_capability_identity"):
        publish_live_registration_bundles(bundles + bundles)
    for operation in ("search", "read", "thread.read", "child.read", "content.read"):
        with pytest.raises(LookupError, match="live_capability_unavailable"):
            published.resolve_handler(
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                capability_id=f"vendor.ms365_graph.mail.{operation}",
                contract_version="1",
            )


def test_mail_request_is_strict_immutable_and_accepts_page_size_only() -> None:
    request = MsGraphMailListLiveRequestV1(page_size=10)
    with pytest.raises(ValidationError):
        MsGraphMailListLiveRequestV1.model_validate(
            {"page_size": 10, "mailbox_user_id": "user"}
        )
    with pytest.raises(ValidationError):
        MsGraphMailListLiveRequestV1.model_validate(
            {"page_size": 10, "remote_resource_id": _SCOPE}
        )
    with pytest.raises(ValidationError):
        MsGraphMailListLiveRequestV1.model_validate({"page_size": "10"})
    with pytest.raises(ValidationError):
        MsGraphMailListLiveRequestV1.model_validate({"page_size": 0})
    with pytest.raises(ValidationError):
        request.page_size = 11


@pytest.mark.asyncio
async def test_mail_handler_reuses_adapter_scope_and_maps_one_page() -> None:
    page = KnowledgePage(
        changes=(_active_change(), _removed_change()),
        next_cursor=KnowledgeCursor(value="must-not-leak", version="mail"),
        proposed_checkpoint=KnowledgeCursor(value="must-not-leak", version="mail"),
        has_more=True,
    )
    adapter = _FakeMailAdapter(page)
    result = await MsGraphMailListLiveHandlerV1(adapter=adapter).execute(
        integration=_integration(),
        call=_call(_request(page_size=50)),
        context=_context(),
    )
    assert result.normalized_outcome is LiveExecutionOutcomeV1.TRUNCATED
    assert result.truncated is True
    assert result.item_count == 2
    assert len(adapter.calls) == 1
    integration, source, cursor, limit = adapter.calls[0]
    assert integration is not None
    assert source.scope.remote_scope_id == _SCOPE
    assert cursor is None
    assert limit == 10
    assert result.items[0].retrieved_at == result.items[1].retrieved_at
    assert result.items[0].safe_locator == "https://example.test/messages/1"
    assert "body" not in result.items[0].content
    assert '"has_attachments":true' in result.items[0].content
    assert '"attachment_inventory_included":false' in result.items[0].content
    assert '"attachment_content_included":false' in result.items[0].content
    assert "global" not in result.items[1].content
    assert _REMOVAL_SEMANTICS in result.items[1].content
    assert result.items[0].content_hash == _sha256(result.items[0].content)


@pytest.mark.asyncio
async def test_mail_handler_completed_page_and_fail_closed_duplicates() -> None:
    completed = KnowledgePage(changes=(_active_change(),), has_more=False)
    result = await MsGraphMailListLiveHandlerV1(
        adapter=_FakeMailAdapter(completed)
    ).execute(
        integration=_integration(),
        call=_call(_request()),
        context=_context(),
    )
    assert result.normalized_outcome is LiveExecutionOutcomeV1.COMPLETED
    assert result.truncated is False

    duplicate_page = KnowledgePage(
        changes=(_removed_change("same"), _removed_change("same"))
    )
    duplicate = await MsGraphMailListLiveHandlerV1(
        adapter=_FakeMailAdapter(duplicate_page)
    ).execute(
        integration=_integration(),
        call=_call(_request()),
        context=_context(),
    )
    assert duplicate.error_code == "live_provider_contract_violation"


@pytest.mark.asyncio
async def test_mail_scope_and_request_are_rejected_before_provider() -> None:
    adapter = _FakeMailAdapter(KnowledgePage())
    handler = MsGraphMailListLiveHandlerV1(adapter=adapter)
    missing = await handler.execute(
        integration=_integration(),
        call=_call(_request(), remote_resource_id=None),
        context=_context(),
    )
    wrong_request = await handler.execute(
        integration=_integration(),
        call=_call(_WrongRequest()),
        context=_context(),
    )
    wrong_integration = await handler.execute(
        integration=object(),
        call=_call(_request()),
        context=_context(),
    )
    mismatch = await handler.execute(
        integration=_integration(),
        call=_call(
            _request(),
            provider_id="other_provider",
        ),
        context=_context(),
    )
    assert missing.error_code == "live_resource_scope_invalid"
    assert wrong_request.error_code == "live_request_invalid"
    assert wrong_integration.error_code == "live_provider_contract_violation"
    assert mismatch.error_code == "live_resource_scope_invalid"
    assert adapter.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vendor_code", "live_code"),
    (
        ("AUTHENTICATION_FAILED", "live_provider_unauthorized"),
        ("AUTHORIZATION_DENIED", "live_provider_forbidden"),
        ("REMOTE_ITEM_NOT_FOUND", "live_provider_not_found"),
        ("REMOTE_ITEM_REVOKED", "live_provider_not_found"),
        ("RATE_LIMITED", "live_provider_throttled"),
        ("DEPENDENCY_UNAVAILABLE", "live_provider_temporarily_unavailable"),
        ("INVALID_PROVIDER_RESPONSE", "live_provider_contract_violation"),
    ),
)
async def test_mail_vendor_errors_use_shared_live_taxonomy(
    vendor_code: str,
    live_code: str,
) -> None:
    handler = MsGraphMailListLiveHandlerV1(
        adapter=_ErrorMailAdapter(VendorKnowledgeErrorCode[vendor_code])
    )
    result = await handler.execute(
        integration=_integration(),
        call=_call(_request()),
        context=_context(),
    )
    assert result.error_code == live_code
    assert result.items == ()


@pytest.mark.asyncio
async def test_mail_handler_propagates_cancellation() -> None:
    with pytest.raises(asyncio.CancelledError):
        await MsGraphMailListLiveHandlerV1(adapter=_CancelledMailAdapter()).execute(
            integration=_integration(),
            call=_call(_request()),
            context=_context(),
        )


def _sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_REMOVAL_SEMANTICS = "removed_from_synchronized_folder_view"
