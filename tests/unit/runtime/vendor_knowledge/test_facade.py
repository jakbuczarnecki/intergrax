# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for VendorKnowledgeFacadeService."""

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.base import IntegrationDependencyError
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeAdapterCapabilities,
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgePermissions,
    KnowledgeScopeInfo,
    KnowledgeVisibility,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from tests.unit.runtime.vendor_knowledge._fakes import (
    FakeAdapter,
    FakeIntegration,
    RecordingResolver,
    make_content,
    make_descriptor,
    make_page,
    make_source,
)


def _facade(
    *,
    tenant_id: str = "tenant-1",
    integration: object | None = None,
    adapter: FakeAdapter | None = None,
    resolver: RecordingResolver | None = None,
    registry: KnowledgeAdapterRegistry | None = None,
) -> tuple[VendorKnowledgeFacadeService, RecordingResolver, FakeAdapter]:
    resolved_integration = integration if integration is not None else FakeIntegration()
    resolved_adapter = adapter or FakeAdapter()
    resolved_resolver = resolver or RecordingResolver(integration=resolved_integration)
    resolved_registry = registry or KnowledgeAdapterRegistry()
    if not resolved_registry.registered_keys():
        resolved_registry.register(resolved_adapter)
    service = VendorKnowledgeFacadeService(
        tenant_id=tenant_id,
        resolver=resolved_resolver,
        adapter_registry=resolved_registry,
    )
    return service, resolved_resolver, resolved_adapter


@pytest.mark.unit
@pytest.mark.asyncio
async def test_full_inspect_flow() -> None:
    source = make_source()
    service, resolver, adapter = _facade()

    result = await service.inspect_source(source=source)

    assert result.source == source
    assert resolver.calls == [source]
    assert adapter.inspect_calls[0]["integration"] is resolver.integration


@pytest.mark.unit
@pytest.mark.asyncio
async def test_full_initial_page_read_flow() -> None:
    source = make_source()
    service, resolver, adapter = _facade()

    page = await service.read_page(source=source, cursor=None, limit=10)

    assert len(page.changes) == 1
    assert adapter.read_calls[0]["cursor"] is None
    assert adapter.read_calls[0]["limit"] == 10
    assert adapter.read_calls[0]["integration"] is resolver.integration


@pytest.mark.unit
@pytest.mark.asyncio
async def test_incremental_read_with_cursor() -> None:
    source = make_source()
    cursor = KnowledgeCursor(value="cursor-1")
    service, _resolver, adapter = _facade()

    await service.read_page(source=source, cursor=cursor, limit=5)

    assert adapter.read_calls[0]["cursor"] == cursor


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        KnowledgeContentMode.BINARY,
        KnowledgeContentMode.RICH_TEXT,
        KnowledgeContentMode.STRUCTURED_RECORD,
    ],
)
async def test_content_fetch_modes(mode: KnowledgeContentMode) -> None:
    source = make_source()
    item = make_descriptor(source=source, content_mode=mode)
    adapter = FakeAdapter(content=make_content(mode=mode))
    service, _resolver, _adapter = _facade(adapter=adapter)

    content = await service.fetch_content(source=source, item=item)

    assert content.mode is mode


@pytest.mark.unit
@pytest.mark.asyncio
async def test_permission_fetch() -> None:
    source = make_source()
    item = make_descriptor(source=source)
    permissions = KnowledgePermissions(visibility=KnowledgeVisibility.RESTRICTED)
    adapter = FakeAdapter(permissions=permissions)
    service, _resolver, _adapter = _facade(adapter=adapter)

    result = await service.fetch_permissions(source=source, item=item)

    assert result.visibility is KnowledgeVisibility.RESTRICTED


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tenant_mismatch_before_dependency_calls() -> None:
    resolver = RecordingResolver(integration=FakeIntegration())
    registry = KnowledgeAdapterRegistry()
    adapter = FakeAdapter()
    registry.register(adapter)
    service = VendorKnowledgeFacadeService(
        tenant_id="tenant-1",
        resolver=resolver,
        adapter_registry=registry,
    )

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.inspect_source(source=make_source(tenant_id="other"))

    assert exc_info.value.code is VendorKnowledgeErrorCode.TENANT_MISMATCH
    assert resolver.calls == []
    assert adapter.inspect_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_identity_mismatch() -> None:
    source = make_source(provider_id="example")
    adapter = FakeAdapter(provider_id="example")
    registry = KnowledgeAdapterRegistry()
    registry.register(adapter)
    adapter._provider_id = "other"
    service, _resolver, _adapter = _facade(adapter=adapter, registry=registry)

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.inspect_source(source=source)

    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert adapter.inspect_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unsupported_inventory_incremental_read() -> None:
    adapter = FakeAdapter(
        capabilities=KnowledgeAdapterCapabilities(
            full_inventory=False,
            incremental_changes=False,
        )
    )
    service, _resolver, resolved = _facade(adapter=adapter)

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.read_page(source=make_source(), cursor=None, limit=10)

    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert resolved.read_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unsupported_incremental_when_cursor_present() -> None:
    adapter = FakeAdapter(
        capabilities=KnowledgeAdapterCapabilities(
            full_inventory=True,
            incremental_changes=False,
        )
    )
    service, _resolver, resolved = _facade(adapter=adapter)

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.read_page(
            source=make_source(),
            cursor=KnowledgeCursor(value="c1"),
            limit=10,
        )

    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert resolved.read_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unsupported_content_mode() -> None:
    adapter = FakeAdapter(
        capabilities=KnowledgeAdapterCapabilities(
            content_fetch=True,
            binary_content=False,
            rich_text_content=False,
            structured_content=False,
        )
    )
    service, _resolver, resolved = _facade(adapter=adapter)
    item = make_descriptor(content_mode=KnowledgeContentMode.BINARY)

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.fetch_content(source=make_source(), item=item)

    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert resolved.content_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unsupported_permissions() -> None:
    adapter = FakeAdapter(
        capabilities=KnowledgeAdapterCapabilities(permissions=False),
    )
    service, _resolver, resolved = _facade(adapter=adapter)

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.fetch_permissions(
            source=make_source(),
            item=make_descriptor(),
        )

    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert resolved.permissions_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalid_limit() -> None:
    service, resolver, adapter = _facade()

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.read_page(source=make_source(), cursor=None, limit=0)

    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert resolver.calls == []
    assert adapter.read_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mismatched_scope_result() -> None:
    source = make_source()
    other = make_source(remote_scope_id="other-scope")
    adapter = FakeAdapter(
        scope_info=KnowledgeScopeInfo(
            source=other,
            capabilities=KnowledgeAdapterCapabilities(full_inventory=True),
            safe_display_name="Other",
        )
    )
    service, _resolver, _adapter = _facade(adapter=adapter)

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.inspect_source(source=source)

    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mismatched_provenance_provider_source() -> None:
    source = make_source()
    adapter = FakeAdapter(
        page=make_page(source=source, provider_id="other", source_kind="issues"),
    )
    service, _resolver, _adapter = _facade(adapter=adapter)

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.read_page(source=source, cursor=None, limit=10)

    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mismatched_returned_content_mode() -> None:
    source = make_source()
    item = make_descriptor(source=source, content_mode=KnowledgeContentMode.BINARY)
    adapter = FakeAdapter(
        content=KnowledgeContent(mode=KnowledgeContentMode.RICH_TEXT, rich_text="x"),
    )
    service, _resolver, _adapter = _facade(adapter=adapter)

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.fetch_content(source=source, item=item)

    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


@pytest.mark.unit
@pytest.mark.asyncio
async def test_vendor_knowledge_error_passthrough() -> None:
    original = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.REMOTE_ITEM_NOT_FOUND,
        safe_message="item missing",
        retryable=False,
    )
    adapter = FakeAdapter(inspect_error=original)
    service, _resolver, _adapter = _facade(adapter=adapter)

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.inspect_source(source=make_source())

    assert exc_info.value is original


@pytest.mark.unit
@pytest.mark.asyncio
async def test_integration_dependency_error_normalization() -> None:
    adapter = FakeAdapter(
        inspect_error=IntegrationDependencyError(
            "timeout token=leaked-secret",
            integration_name="example",
        )
    )
    service, _resolver, _adapter = _facade(adapter=adapter)

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.inspect_source(source=make_source())

    error = exc_info.value
    assert error.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert error.retryable is True
    assert "leaked-secret" not in error.safe_message
    assert "token=" not in error.safe_message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unexpected_adapter_error_normalization() -> None:
    adapter = FakeAdapter(inspect_error=RuntimeError("body https://x?api_key=1"))
    service, _resolver, _adapter = _facade(adapter=adapter)

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await service.inspect_source(source=make_source())

    error = exc_info.value
    assert error.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert error.retryable is False
    assert "api_key" not in error.safe_message
    assert "https://" not in error.safe_message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_same_resolved_integration_instance_passed_to_adapter() -> None:
    integration = FakeIntegration()
    service, resolver, adapter = _facade(integration=integration)
    source = make_source()
    item = make_descriptor(source=source)

    await service.inspect_source(source=source)
    await service.read_page(source=source, cursor=None, limit=3)
    await service.fetch_content(source=source, item=item)
    await service.fetch_permissions(source=source, item=item)

    assert adapter.inspect_calls[0]["integration"] is integration
    assert adapter.read_calls[0]["integration"] is integration
    assert adapter.content_calls[0]["integration"] is integration
    assert adapter.permissions_calls[0]["integration"] is integration
    assert resolver.integration is integration


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_network_or_persistence_required() -> None:
    service, _resolver, _adapter = _facade()
    source = make_source()

    scope = await service.inspect_source(source=source)
    page = await service.read_page(source=source, cursor=None, limit=1)
    content = await service.fetch_content(
        source=source,
        item=make_descriptor(source=source),
    )
    permissions = await service.fetch_permissions(
        source=source,
        item=make_descriptor(source=source),
    )

    assert scope.safe_display_name
    assert page.changes
    assert content.mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert permissions.visibility is KnowledgeVisibility.TENANT
