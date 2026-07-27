# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Stateless Vendor Knowledge Facade core service."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.runtime.vendor_knowledge.contracts import (
    VendorIntegrationResolver,
    VendorKnowledgeAdapter,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgePage,
    KnowledgePermissions,
    KnowledgeScopeInfo,
    KnowledgeSourceRef,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry

_T = TypeVar("_T")


class VendorKnowledgeFacadeService:
    """Application-facing vendor-neutral knowledge access boundary.

    Dependencies are injected; the service performs no I/O of its own.
    """

    def __init__(
        self,
        *,
        tenant_id: str,
        resolver: VendorIntegrationResolver,
        adapter_registry: KnowledgeAdapterRegistry,
    ) -> None:
        cleaned_tenant = str(tenant_id).strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")
        self._tenant_id = cleaned_tenant
        self._resolver = resolver
        self._adapter_registry = adapter_registry

    async def inspect_source(self, *, source: KnowledgeSourceRef) -> KnowledgeScopeInfo:
        integration, adapter = self._prepare(source=source)
        result = await self._invoke_adapter(
            adapter.inspect_scope,
            integration=integration,
            source=source,
            provider_id=source.provider_id,
            source_kind=source.source_kind,
        )
        if result.source != source:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Adapter inspect result source does not match the request",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        return result

    async def read_page(
        self,
        *,
        source: KnowledgeSourceRef,
        cursor: KnowledgeCursor | None,
        limit: int,
    ) -> KnowledgePage:
        if limit <= 0:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Page limit must be greater than zero",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

        integration, adapter = self._prepare(source=source)
        capabilities = adapter.capabilities

        if not (
            capabilities.full_inventory
            or capabilities.incremental_changes
            or capabilities.reconciliation
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
                safe_message="Page reads are not supported for this adapter",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

        page = await self._invoke_adapter(
            adapter.read_page,
            integration=integration,
            source=source,
            cursor=cursor,
            limit=limit,
            provider_id=source.provider_id,
            source_kind=source.source_kind,
        )
        self._validate_page_provenance(page, source=source)
        return page

    async def fetch_content(
        self,
        *,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeContent:
        integration, adapter = self._prepare(source=source)
        self._validate_item_provenance(item, source=source)
        capabilities = adapter.capabilities

        if not capabilities.content_fetch:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
                safe_message="Content fetch is not supported for this adapter",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

        if item.content_mode is KnowledgeContentMode.BINARY:
            mode_supported = capabilities.binary_content
        elif item.content_mode is KnowledgeContentMode.RICH_TEXT:
            mode_supported = capabilities.rich_text_content
        elif item.content_mode is KnowledgeContentMode.STRUCTURED_RECORD:
            mode_supported = capabilities.structured_content
        else:
            mode_supported = False

        if not mode_supported:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
                safe_message="Requested content mode is not supported for this adapter",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

        content = await self._invoke_adapter(
            adapter.fetch_content,
            integration=integration,
            source=source,
            item=item,
            provider_id=source.provider_id,
            source_kind=source.source_kind,
        )
        if content.mode != item.content_mode:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Adapter content mode does not match the requested item",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        return content

    async def fetch_permissions(
        self,
        *,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgePermissions:
        integration, adapter = self._prepare(source=source)
        self._validate_item_provenance(item, source=source)
        if not adapter.capabilities.permissions:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
                safe_message="Permission fetch is not supported for this adapter",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

        return await self._invoke_adapter(
            adapter.fetch_permissions,
            integration=integration,
            source=source,
            item=item,
            provider_id=source.provider_id,
            source_kind=source.source_kind,
        )

    def _prepare(
        self, *, source: KnowledgeSourceRef
    ) -> tuple[object, VendorKnowledgeAdapter]:
        if source.tenant_id != self._tenant_id:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.TENANT_MISMATCH,
                safe_message="Knowledge source tenant does not match the configured facade tenant",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

        integration = self._resolver.resolve(source=source)
        adapter = self._adapter_registry.resolve(source=source)
        self._validate_adapter_identity(adapter, source=source)
        return integration, adapter

    def _validate_adapter_identity(
        self,
        adapter: VendorKnowledgeAdapter,
        *,
        source: KnowledgeSourceRef,
    ) -> None:
        if (
            adapter.provider_id != source.provider_id
            or adapter.integration_kind != source.integration_kind
            or adapter.source_kind != source.source_kind
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Resolved adapter identity does not match the requested source",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

    def _validate_item_provenance(
        self,
        item: KnowledgeItemDescriptor,
        *,
        source: KnowledgeSourceRef,
    ) -> None:
        provenance = item.provenance
        if (
            provenance.provider_id != source.provider_id
            or provenance.source_kind != source.source_kind
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Item provenance does not match the requested source",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

    def _validate_page_provenance(
        self,
        page: KnowledgePage,
        *,
        source: KnowledgeSourceRef,
    ) -> None:
        for change in page.changes:
            descriptor = change.descriptor
            if descriptor is None:
                continue
            provenance = descriptor.provenance
            if (
                provenance.provider_id != source.provider_id
                or provenance.source_kind != source.source_kind
            ):
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Page change provenance does not match the requested source",
                    provider_id=source.provider_id,
                    source_kind=source.source_kind,
                    retryable=False,
                )

    async def _invoke_adapter(
        self,
        operation: Callable[..., Awaitable[_T]],
        *,
        provider_id: str,
        source_kind: str,
        **kwargs: object,
    ) -> _T:
        try:
            return await operation(**kwargs)
        except VendorKnowledgeError:
            raise
        except IntegrationDependencyError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Vendor knowledge dependency is currently unavailable",
                provider_id=provider_id,
                source_kind=source_kind,
                retryable=True,
            ) from None
        except IntegrationConfigurationError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Vendor knowledge adapter configuration is invalid",
                provider_id=provider_id,
                source_kind=source_kind,
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Vendor knowledge adapter returned an unexpected failure",
                provider_id=provider_id,
                source_kind=source_kind,
                retryable=False,
            ) from None
