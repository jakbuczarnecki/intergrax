# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for vendor-knowledge facade protocols."""

from __future__ import annotations

import inspect

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime import vendor_knowledge as package
from intergrax.runtime.vendor_knowledge.contracts import (
    VendorIntegrationResolver,
    VendorKnowledgeAdapter,
    VendorKnowledgeFacade,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeAdapterCapabilities,
    KnowledgeChange,
    KnowledgeChangeKind,
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgeItemIdentity,
    KnowledgeItemProvenance,
    KnowledgeItemRevision,
    KnowledgePage,
    KnowledgePermissions,
    KnowledgeScopeInfo,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
    KnowledgeVisibility,
)


def _source() -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id="example",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
        scope=KnowledgeSourceScope(
            remote_scope_id="proj-1",
            remote_scope_type="project",
            safe_display_name="Project",
            parameters={},
        ),
    )


def _item() -> KnowledgeItemDescriptor:
    return KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id="item-1"),
        revision=KnowledgeItemRevision(),
        title="Issue",
        item_type="issue",
        content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id="example",
            source_kind="issues",
            remote_id="item-1",
        ),
        metadata={},
    )


class _FakeResolver:
    def __init__(self, integration: object) -> None:
        self._integration = integration
        self.calls: list[KnowledgeSourceRef] = []

    def resolve(self, *, source: KnowledgeSourceRef) -> object:
        self.calls.append(source)
        return self._integration


class _FakeAdapter:
    def __init__(self) -> None:
        self.seen_integrations: list[object] = []

    @property
    def provider_id(self) -> str:
        return "example"

    @property
    def integration_kind(self) -> IntegrationCategory:
        return IntegrationCategory.ISSUE_TRACKER

    @property
    def source_kind(self) -> str:
        return "issues"

    @property
    def capabilities(self) -> KnowledgeAdapterCapabilities:
        return KnowledgeAdapterCapabilities(
            content_fetch=True,
            permissions=True,
            incremental_changes=True,
        )

    async def inspect_scope(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
    ) -> KnowledgeScopeInfo:
        self.seen_integrations.append(integration)
        return KnowledgeScopeInfo(
            source=source,
            capabilities=self.capabilities,
            safe_display_name=source.scope.safe_display_name,
        )

    async def read_page(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        cursor: KnowledgeCursor | None,
        limit: int,
    ) -> KnowledgePage:
        self.seen_integrations.append(integration)
        _ = source, cursor, limit
        return KnowledgePage(
            changes=(
                KnowledgeChange(
                    kind=KnowledgeChangeKind.UPSERT,
                    remote_id="item-1",
                    descriptor=_item(),
                ),
            ),
            has_more=False,
        )

    async def fetch_content(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeContent:
        self.seen_integrations.append(integration)
        _ = source, item
        return KnowledgeContent(
            mode=KnowledgeContentMode.STRUCTURED_RECORD,
            structured_record={"id": "item-1"},
        )

    async def fetch_permissions(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgePermissions:
        self.seen_integrations.append(integration)
        _ = source, item
        return KnowledgePermissions(visibility=KnowledgeVisibility.UNKNOWN)


class _FakeFacade:
    def __init__(self, resolver: _FakeResolver, adapter: _FakeAdapter) -> None:
        self._resolver = resolver
        self._adapter = adapter

    async def inspect_source(self, *, source: KnowledgeSourceRef) -> KnowledgeScopeInfo:
        integration = self._resolver.resolve(source=source)
        return await self._adapter.inspect_scope(integration=integration, source=source)

    async def read_page(
        self,
        *,
        source: KnowledgeSourceRef,
        cursor: KnowledgeCursor | None,
        limit: int,
    ) -> KnowledgePage:
        integration = self._resolver.resolve(source=source)
        return await self._adapter.read_page(
            integration=integration,
            source=source,
            cursor=cursor,
            limit=limit,
        )

    async def fetch_content(
        self,
        *,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeContent:
        integration = self._resolver.resolve(source=source)
        return await self._adapter.fetch_content(
            integration=integration,
            source=source,
            item=item,
        )

    async def fetch_permissions(
        self,
        *,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgePermissions:
        integration = self._resolver.resolve(source=source)
        return await self._adapter.fetch_permissions(
            integration=integration,
            source=source,
            item=item,
        )


@pytest.mark.unit
def test_fake_resolver_satisfies_protocol() -> None:
    resolver = _FakeResolver(integration=object())
    assert isinstance(resolver, VendorIntegrationResolver)


@pytest.mark.unit
def test_fake_adapter_satisfies_protocol() -> None:
    adapter = _FakeAdapter()
    assert isinstance(adapter, VendorKnowledgeAdapter)


@pytest.mark.unit
def test_fake_facade_satisfies_protocol() -> None:
    facade = _FakeFacade(_FakeResolver(object()), _FakeAdapter())
    assert isinstance(facade, VendorKnowledgeFacade)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_receives_external_integration_instance() -> None:
    integration = object()
    adapter = _FakeAdapter()
    source = _source()
    await adapter.inspect_scope(integration=integration, source=source)
    await adapter.read_page(integration=integration, source=source, cursor=None, limit=10)
    assert adapter.seen_integrations == [integration, integration]


@pytest.mark.unit
def test_contracts_are_vendor_neutral() -> None:
    annotations = (
        str(inspect.signature(VendorKnowledgeAdapter.inspect_scope)),
        str(inspect.signature(VendorKnowledgeFacade.read_page)),
        str(VendorIntegrationResolver.resolve.__annotations__),
    )
    joined = " ".join(annotations).lower()
    assert "jira" not in joined
    assert "confluence" not in joined
    assert "ms365" not in joined
    assert "databricks" not in joined


@pytest.mark.unit
def test_package_does_not_import_lkw() -> None:
    module_names = set(getattr(package, "__all__", ()))
    assert "local_workspace_application" not in repr(package)
    assert "ManagedWorkspace" not in module_names
    import sys

    loaded = " ".join(sys.modules)
    # Importing the contract package must not pull LKW modules.
    assert "local_workspace_application" not in loaded


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_facade_flow_on_fake_data() -> None:
    integration = {"provider": "example"}
    resolver = _FakeResolver(integration)
    adapter = _FakeAdapter()
    facade = _FakeFacade(resolver, adapter)
    source = _source()
    item = _item()

    scope_info = await facade.inspect_source(source=source)
    page = await facade.read_page(source=source, cursor=None, limit=5)
    content = await facade.fetch_content(source=source, item=item)
    permissions = await facade.fetch_permissions(source=source, item=item)

    assert scope_info.source.tenant_id == "tenant-1"
    assert page.changes[0].remote_id == "item-1"
    assert content.structured_record == {"id": "item-1"}
    assert permissions.visibility is KnowledgeVisibility.UNKNOWN
    assert resolver.calls
    assert all(seen is integration for seen in adapter.seen_integrations)
