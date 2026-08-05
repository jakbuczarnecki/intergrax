from __future__ import annotations

import asyncio
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
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.common import (
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.drive import (
    MSGRAPH_DRIVE_SOURCE_KIND,
    MsGraphDriveDeltaPage,
    MsGraphDriveItem,
    MsGraphDriveItemKind,
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
    MSGRAPH_DRIVE_LIST_CAPABILITY_ID,
    MSGRAPH_DRIVE_LIST_REQUEST_SCHEMA_REF,
    MSGRAPH_DRIVE_LIST_RESULT_SCHEMA_REF,
    MsGraphDriveListLiveHandlerV1,
    MsGraphDriveListLiveRequestV1,
    build_msgraph_drive_live_registration_bundles,
)
from intergrax.runtime.vendor_knowledge.live.registration import (
    publish_live_registration_bundles,
)

_NOW = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)
_DRIVE_ID = "drive-1"


class _FakeDriveClient:
    def __init__(self, page: MsGraphDriveDeltaPage) -> None:
        self.page = page
        self.calls: list[dict[str, Any]] = []

    def read_drive_delta_page(
        self,
        *,
        drive_id: str,
        continuation: object = None,
        limit: int,
    ) -> MsGraphDriveDeltaPage:
        self.calls.append(
            {
                "drive_id": drive_id,
                "continuation": continuation,
                "limit": limit,
            }
        )
        return self.page


class _Resolver:
    def __init__(self, integration: object) -> None:
        self.integration = integration
        self.calls: list[dict[str, str]] = []

    def resolve(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
    ) -> object:
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "connection_ref": connection_ref,
                "provider_id": provider_id,
                "integration_kind": integration_kind.value,
            }
        )
        return self.integration


class _StubAdapter:
    def __init__(self, error: VendorKnowledgeError | None = None) -> None:
        self.error = error

    async def read_page(self, **_: object) -> object:
        if self.error is not None:
            raise self.error
        raise AssertionError("stub page is not configured")


def _item(
    *,
    remote_id: str,
    kind: MsGraphDriveItemKind,
    name: str,
    web_url: str | None = None,
) -> MsGraphDriveItem:
    return MsGraphDriveItem(
        remote_id=remote_id,
        drive_id=_DRIVE_ID,
        parent_remote_id=None,
        kind=kind,
        name=name,
        c_tag='"ctag-1"' if kind is MsGraphDriveItemKind.FILE else None,
        size_bytes=12 if kind is MsGraphDriveItemKind.FILE else None,
        mime_type="application/pdf" if kind is MsGraphDriveItemKind.FILE else None,
        created_at=_NOW,
        last_modified_at=_NOW,
        web_url=web_url,
        is_root=kind is MsGraphDriveItemKind.FOLDER,
    )


def _page(
    items: tuple[MsGraphDriveItem, ...],
    *,
    has_more: bool = False,
) -> MsGraphDriveDeltaPage:
    return MsGraphDriveDeltaPage(
        items=items,
        continuation=MsGraphKnowledgeContinuation(
            kind=(
                MsGraphKnowledgeContinuationKind.NEXT_PAGE
                if has_more
                else MsGraphKnowledgeContinuationKind.DELTA
            ),
            url="https://graph.example/continuation",
        ),
    )


def _integration(client: _FakeDriveClient) -> Ms365GraphCollaborationSuiteIntegration:
    return Ms365GraphCollaborationSuiteIntegration.from_client(client, enabled=True)


def _call(
    *,
    request: MsGraphDriveListLiveRequestV1 | None = None,
    remote_resource_id: str | None = _DRIVE_ID,
    max_result_items: int = 10,
    max_upstream_items: int = 10,
    max_provider_page_size: int = 10,
) -> ValidatedLiveCapabilityCallV1:
    return ValidatedLiveCapabilityCallV1(
        call_id="call-1",
        capability_id=MSGRAPH_DRIVE_LIST_CAPABILITY_ID,
        contract_version="1",
        connection_ref="connection-1",
        live_access_binding_id="binding-1",
        remote_resource_id=remote_resource_id,
        validated_request=(
            request
            if isinstance(request, MsGraphDriveListLiveRequestV1)
            else MsGraphDriveListLiveRequestV1(page_size=25)
        ),
        effective_budget=EffectiveLiveCallBudgetV1(
            max_live_calls=1,
            max_total_duration_ms=1_000,
            max_result_items=max_result_items,
            max_result_bytes=10_000,
            max_provider_pages=1,
            max_provider_requests=1,
            max_upstream_items=max_upstream_items,
            max_provider_page_size=max_provider_page_size,
            max_content_bytes_per_item=4_096,
        ),
        audience_context_ref="personal",
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_DRIVE_SOURCE_KIND,
    )


def _context() -> LiveCapabilityExecutionContextV1:
    return LiveCapabilityExecutionContextV1(
        run_id="run-1",
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        audience=KnowledgeQueryAudienceV1.PERSONAL,
        started_at=_NOW,
        deadline_monotonic=100.0,
        retention=LiveResultRetentionV1.EPHEMERAL,
    )


@pytest.mark.unit
def test_drive_registration_is_one_complete_canonical_bundle() -> None:
    bundles = build_msgraph_drive_live_registration_bundles()
    published = publish_live_registration_bundles(bundles)
    provider_id, source_kind, operation = parse_capability_id(
        MSGRAPH_DRIVE_LIST_CAPABILITY_ID
    )

    assert len(bundles) == 1
    assert len(published.descriptors) == 1
    assert len(published.handlers) == 1
    assert provider_id == MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    assert source_kind == MSGRAPH_DRIVE_SOURCE_KIND
    assert operation is LiveOperationV1.LIST
    assert published.schemas.resolve_request(
        MSGRAPH_DRIVE_LIST_REQUEST_SCHEMA_REF, "1"
    ) is MsGraphDriveListLiveRequestV1
    assert published.schemas.resolve_result(
        MSGRAPH_DRIVE_LIST_RESULT_SCHEMA_REF, "1"
    ).__name__ == "LiveCapabilityExecutionResultV1"
    for unsupported_capability_id in (
        "vendor.ms365_graph.drive.search",
        "vendor.ms365_graph.drive.read",
        "vendor.ms365_graph.drive.child.read",
        "vendor.ms365_graph.drive.content.read",
    ):
        with pytest.raises(LookupError):
            published.resolve_handler(
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                capability_id=unsupported_capability_id,
                contract_version="1",
            )
    with pytest.raises(ValueError):
        publish_live_registration_bundles(bundles + bundles)


@pytest.mark.unit
def test_drive_request_is_strict_immutable_and_has_no_scope_controls() -> None:
    request = MsGraphDriveListLiveRequestV1(page_size=10)
    with pytest.raises(ValidationError):
        MsGraphDriveListLiveRequestV1.model_validate({"page_size": 10, "drive_id": "other"})
    with pytest.raises(ValidationError):
        MsGraphDriveListLiveRequestV1.model_validate({"page_size": 10, "continuation": "x"})
    with pytest.raises(ValidationError):
        MsGraphDriveListLiveRequestV1.model_validate({"page_size": "10"})
    with pytest.raises(ValidationError):
        request.page_size = 20


@pytest.mark.asyncio
@pytest.mark.unit
async def test_drive_list_reuses_injected_integration_once_and_maps_metadata() -> None:
    page = _page(
        (
            _item(
                remote_id="file-1",
                kind=MsGraphDriveItemKind.FILE,
                name="Report.pdf",
                web_url="https://graph.example/items/file-1?cursor=secret",
            ),
            _item(
                remote_id="folder-1",
                kind=MsGraphDriveItemKind.FOLDER,
                name="Reports",
            ),
        ),
        has_more=True,
    )
    client = _FakeDriveClient(page)
    integration = _integration(client)
    result = await MsGraphDriveListLiveHandlerV1().execute(
        integration=integration,
        call=_call(max_result_items=4, max_upstream_items=3, max_provider_page_size=2),
        context=_context(),
    )

    assert result.normalized_outcome is LiveExecutionOutcomeV1.TRUNCATED
    assert result.truncated is True
    assert [item.remote_item_id for item in result.items] == ["file-1", "folder-1"]
    assert result.items[0].safe_locator is None
    assert '"drive_item_kind":"file"' in result.items[0].content
    assert '"drive_item_kind":"folder"' in result.items[1].content
    assert result.items[0].content_hash == hashlib.sha256(
        result.items[0].content.encode("utf-8")
    ).hexdigest()
    assert client.calls == [
        {
            "drive_id": _DRIVE_ID,
            "continuation": None,
            "limit": 2,
        }
    ]
    assert "next_cursor" not in LiveCapabilityExecutionResultV1.model_fields


@pytest.mark.asyncio
@pytest.mark.unit
async def test_drive_scope_and_integration_rejections_happen_before_provider_call() -> None:
    client = _FakeDriveClient(_page(()))
    integration = _integration(client)
    handler = MsGraphDriveListLiveHandlerV1()

    missing = await handler.execute(
        integration=integration,
        call=_call(remote_resource_id=None),
        context=_context(),
    )
    invalid = await handler.execute(
        integration=integration,
        call=_call(remote_resource_id="\x00invalid"),
        context=_context(),
    )
    wrong_integration = await handler.execute(
        integration=object(),
        call=_call(),
        context=_context(),
    )

    assert missing.error_code == "live_resource_scope_invalid"
    assert invalid.error_code == "live_resource_scope_invalid"
    assert wrong_integration.error_code == "live_provider_contract_violation"
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(
    ("vendor_code", "live_code"),
    [
        (VendorKnowledgeErrorCode.INVALID_SCOPE, "live_resource_scope_invalid"),
        (VendorKnowledgeErrorCode.CONFIGURATION_ERROR, "live_request_invalid"),
        (VendorKnowledgeErrorCode.AUTHENTICATION_FAILED, "live_provider_unauthorized"),
        (VendorKnowledgeErrorCode.AUTHORIZATION_DENIED, "live_provider_forbidden"),
        (VendorKnowledgeErrorCode.REMOTE_ITEM_NOT_FOUND, "live_provider_not_found"),
        (VendorKnowledgeErrorCode.RATE_LIMITED, "live_provider_throttled"),
        (
            VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
            "live_provider_temporarily_unavailable",
        ),
        (
            VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            "live_provider_contract_violation",
        ),
    ],
)
async def test_drive_vendor_errors_use_shared_live_taxonomy(
    vendor_code: VendorKnowledgeErrorCode,
    live_code: str,
) -> None:
    error = VendorKnowledgeError(
        code=vendor_code,
        safe_message="safe provider failure",
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        source_kind=MSGRAPH_DRIVE_SOURCE_KIND,
    )
    result = await MsGraphDriveListLiveHandlerV1(
        adapter=_StubAdapter(error)  # type: ignore[arg-type]
    ).execute(
        integration=_integration(_FakeDriveClient(_page(()))),
        call=_call(),
        context=_context(),
    )

    assert result.normalized_outcome is LiveExecutionOutcomeV1.FAILED
    assert result.error_code == live_code


@pytest.mark.asyncio
@pytest.mark.unit
async def test_drive_cancellation_is_not_swallowed() -> None:
    class _CancelledAdapter:
        async def read_page(self, **_: object) -> object:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await MsGraphDriveListLiveHandlerV1(
            adapter=_CancelledAdapter()  # type: ignore[arg-type]
        ).execute(
            integration=_integration(_FakeDriveClient(_page(()))),
            call=_call(),
            context=_context(),
        )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_real_executor_uses_published_drive_handler_and_receipt_only() -> None:
    from local_workspace_application.workspaces.hybrid_ask_execution import (
        LiveCapabilityExecutorV1,
    )

    page = _page(
        (
            _item(
                remote_id="file-1",
                kind=MsGraphDriveItemKind.FILE,
                name="Report.pdf",
            ),
        )
    )
    client = _FakeDriveClient(page)
    integration = _integration(client)
    resolver = _Resolver(integration)
    published = publish_live_registration_bundles(
        build_msgraph_drive_live_registration_bundles()
    )
    result = await LiveCapabilityExecutorV1(
        published_registration=published,
        integration_resolver=resolver,
        clock=lambda: _NOW,
        monotonic=lambda: 100.0,
        id_factory=lambda: "receipt-1",
    ).execute(
        run_id="run-1",
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        call=_call(request=MsGraphDriveListLiveRequestV1(page_size=5)),
        audience=KnowledgeQueryAudienceV1.PERSONAL,
        retention=LiveResultRetentionV1.RECEIPT_ONLY,
    )

    assert result.normalized_outcome is LiveExecutionOutcomeV1.COMPLETED
    assert result.provider_id == MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    assert result.source_kind == MSGRAPH_DRIVE_SOURCE_KIND
    assert result.remote_resource_id == _DRIVE_ID
    assert result.receipt is not None
    assert result.receipt.item_count == 1
    assert resolver.calls[0]["connection_ref"] == "connection-1"
    assert len(client.calls) == 1
