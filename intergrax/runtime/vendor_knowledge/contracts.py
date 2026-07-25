# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Protocol ports for the Vendor Knowledge Facade (no runtime behavior)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeAdapterCapabilities,
    KnowledgeContent,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgePage,
    KnowledgePermissions,
    KnowledgeScopeInfo,
    KnowledgeSourceRef,
)


@runtime_checkable
class VendorIntegrationResolver(Protocol):
    """Resolve an existing provider/category integration for a knowledge source.

    Production implementations delegate to ``IntegrationProfile`` / catalog.
    This protocol does not perform resolution itself.
    """

    def resolve(
        self,
        *,
        source: KnowledgeSourceRef,
    ) -> object:
        """Return an already constructed integration instance."""
        ...


@runtime_checkable
class VendorKnowledgeAdapter(Protocol):
    """Thin mapping from a resolved integration into vendor-neutral knowledge models.

    Adapters receive an integration instance; they do not create clients,
    fetch secrets, persist checkpoints, invoke RAG, or import LKW.
    """

    @property
    def provider_id(self) -> str:
        ...

    @property
    def integration_kind(self) -> IntegrationCategory:
        ...

    @property
    def source_kind(self) -> str:
        ...

    @property
    def capabilities(self) -> KnowledgeAdapterCapabilities:
        ...

    async def inspect_scope(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
    ) -> KnowledgeScopeInfo:
        ...

    async def read_page(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        cursor: KnowledgeCursor | None,
        limit: int,
    ) -> KnowledgePage:
        ...

    async def fetch_content(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeContent:
        ...

    async def fetch_permissions(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgePermissions:
        ...


@runtime_checkable
class VendorKnowledgeFacade(Protocol):
    """Application-facing vendor-neutral knowledge access boundary."""

    async def inspect_source(
        self,
        *,
        source: KnowledgeSourceRef,
    ) -> KnowledgeScopeInfo:
        ...

    async def read_page(
        self,
        *,
        source: KnowledgeSourceRef,
        cursor: KnowledgeCursor | None,
        limit: int,
    ) -> KnowledgePage:
        ...

    async def fetch_content(
        self,
        *,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeContent:
        ...

    async def fetch_permissions(
        self,
        *,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgePermissions:
        ...
