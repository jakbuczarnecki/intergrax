# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared test fakes for vendor knowledge facade unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.document_store import (
    DocumentQueryResult,
    DocumentRecord,
)
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError
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


@dataclass
class FakeIntegration:
    """Neutral PlatformIntegration-like object for resolver/facade tests."""

    provider_id: str = "example"
    integration_kind: str = IntegrationCategory.ISSUE_TRACKER.value
    integration_id: str = "example:issue_tracker"
    constructed: bool = False


@dataclass
class RecordingResolver:
    """Records resolve calls without constructing integrations."""

    integration: object
    calls: list[KnowledgeSourceRef] = field(default_factory=list)
    error: Exception | None = None

    def resolve(self, *, source: KnowledgeSourceRef) -> object:
        self.calls.append(source)
        if self.error is not None:
            raise self.error
        return self.integration


class FakeAdapter:
    """Configurable vendor-neutral adapter with recording hooks."""

    def __init__(
        self,
        *,
        provider_id: str = "example",
        integration_kind: IntegrationCategory = IntegrationCategory.ISSUE_TRACKER,
        source_kind: str = "issues",
        capabilities: KnowledgeAdapterCapabilities | None = None,
        scope_info: KnowledgeScopeInfo | None = None,
        page: KnowledgePage | None = None,
        content: KnowledgeContent | None = None,
        permissions: KnowledgePermissions | None = None,
        inspect_error: Exception | None = None,
        read_error: Exception | None = None,
        content_error: Exception | None = None,
        permissions_error: Exception | None = None,
    ) -> None:
        self._provider_id = provider_id
        self._integration_kind = integration_kind
        self._source_kind = source_kind
        self._capabilities = capabilities or KnowledgeAdapterCapabilities(
            full_inventory=True,
            incremental_changes=True,
            content_fetch=True,
            binary_content=True,
            rich_text_content=True,
            structured_content=True,
            permissions=True,
        )
        self._scope_info = scope_info
        self._page = page
        self._content = content
        self._permissions = permissions
        self._inspect_error = inspect_error
        self._read_error = read_error
        self._content_error = content_error
        self._permissions_error = permissions_error
        self.inspect_calls: list[dict[str, Any]] = []
        self.read_calls: list[dict[str, Any]] = []
        self.content_calls: list[dict[str, Any]] = []
        self.permissions_calls: list[dict[str, Any]] = []

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def integration_kind(self) -> IntegrationCategory:
        return self._integration_kind

    @property
    def source_kind(self) -> str:
        return self._source_kind

    @property
    def capabilities(self) -> KnowledgeAdapterCapabilities:
        return self._capabilities

    async def inspect_scope(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
    ) -> KnowledgeScopeInfo:
        self.inspect_calls.append({"integration": integration, "source": source})
        if self._inspect_error is not None:
            raise self._inspect_error
        if self._scope_info is not None:
            return self._scope_info
        return KnowledgeScopeInfo(
            source=source,
            capabilities=self._capabilities,
            safe_display_name="Example scope",
        )

    async def read_page(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        cursor: KnowledgeCursor | None,
        limit: int,
    ) -> KnowledgePage:
        self.read_calls.append(
            {
                "integration": integration,
                "source": source,
                "cursor": cursor,
                "limit": limit,
            }
        )
        if self._read_error is not None:
            raise self._read_error
        if self._page is not None:
            return self._page
        return make_page(source=source)

    async def fetch_content(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeContent:
        self.content_calls.append(
            {"integration": integration, "source": source, "item": item}
        )
        if self._content_error is not None:
            raise self._content_error
        if self._content is not None:
            return self._content
        return make_content(mode=item.content_mode)

    async def fetch_permissions(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgePermissions:
        self.permissions_calls.append(
            {"integration": integration, "source": source, "item": item}
        )
        if self._permissions_error is not None:
            raise self._permissions_error
        if self._permissions is not None:
            return self._permissions
        return KnowledgePermissions(visibility=KnowledgeVisibility.TENANT)


def make_source(
    *,
    tenant_id: str = "tenant-1",
    provider_id: str = "example",
    integration_kind: IntegrationCategory = IntegrationCategory.ISSUE_TRACKER,
    source_kind: str = "issues",
    connection_ref: str | None = None,
    remote_scope_id: str = "scope-1",
) -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id=tenant_id,
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        connection_ref=connection_ref,
        scope=KnowledgeSourceScope(
            remote_scope_id=remote_scope_id,
            remote_scope_type="project",
            safe_display_name="Example Project",
            parameters={},
        ),
    )


def make_descriptor(
    *,
    source: KnowledgeSourceRef | None = None,
    remote_id: str = "item-1",
    content_mode: KnowledgeContentMode = KnowledgeContentMode.STRUCTURED_RECORD,
    provider_id: str | None = None,
    source_kind: str | None = None,
) -> KnowledgeItemDescriptor:
    resolved_source = source or make_source()
    return KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id=remote_id),
        revision=KnowledgeItemRevision(version="1"),
        title="Example item",
        item_type="record",
        content_mode=content_mode,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id=provider_id or resolved_source.provider_id,
            source_kind=source_kind or resolved_source.source_kind,
            remote_id=remote_id,
        ),
        metadata={},
    )


def make_page(
    *,
    source: KnowledgeSourceRef | None = None,
    remote_id: str = "item-1",
    provider_id: str | None = None,
    source_kind: str | None = None,
    next_cursor: KnowledgeCursor | None = None,
    has_more: bool = False,
) -> KnowledgePage:
    descriptor = make_descriptor(
        source=source,
        remote_id=remote_id,
        provider_id=provider_id,
        source_kind=source_kind,
    )
    return KnowledgePage(
        changes=(
            KnowledgeChange(
                kind=KnowledgeChangeKind.UPSERT,
                descriptor=descriptor,
                remote_id=remote_id,
            ),
        ),
        next_cursor=next_cursor,
        proposed_checkpoint=None,
        has_more=has_more,
    )


def make_content(*, mode: KnowledgeContentMode) -> KnowledgeContent:
    if mode is KnowledgeContentMode.BINARY:
        return KnowledgeContent(mode=mode, binary=b"bytes")
    if mode is KnowledgeContentMode.RICH_TEXT:
        return KnowledgeContent(mode=mode, rich_text="hello")
    return KnowledgeContent(mode=mode, structured_record={"id": "item-1"})


def raise_vendor_error(
    *,
    code: Any,
    safe_message: str = "adapter domain failure",
) -> VendorKnowledgeError:
    return VendorKnowledgeError(code=code, safe_message=safe_message)


@dataclass
class FakeConnectionIntegration:
    """Neutral integration instance for connection-registry tests."""

    provider_id: str = "ms365_graph"
    integration_kind: str = IntegrationCategory.COLLABORATION_SUITE.value
    label: str = "connection-a"


class InMemoryDocumentStore:
    """Minimal DocumentStore fake for binding repository tests."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], DocumentRecord] = {}
        self.closed = False
        self.close_calls = 0

    def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
        return self._rows.get((partition_key, row_key))

    def put(self, document: DocumentRecord) -> None:
        self._rows[(document.partition_key, document.row_key)] = document

    def delete(self, partition_key: str, row_key: str) -> None:
        self._rows.pop((partition_key, row_key), None)

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: Optional[str] = None,
    ) -> DocumentQueryResult:
        matches: list[DocumentRecord] = []
        for (pk, rk), document in sorted(self._rows.items(), key=lambda item: item[0][1]):
            if pk != partition_key:
                continue
            if row_key_prefix is not None and not rk.startswith(row_key_prefix):
                continue
            matches.append(document)
            if len(matches) >= limit:
                break
        return DocumentQueryResult(documents=matches, total=len(matches))

    def close(self) -> None:
        self.closed = True
        self.close_calls += 1
