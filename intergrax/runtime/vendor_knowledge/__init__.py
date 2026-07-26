# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vendor Knowledge Facade — contracts and stateless core services."""

from intergrax.runtime.vendor_knowledge.contracts import (
    VendorIntegrationResolver,
    VendorKnowledgeAdapter,
    VendorKnowledgeFacade,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
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
    KnowledgePrincipal,
    KnowledgeScopeInfo,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
    KnowledgeVisibility,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from intergrax.runtime.vendor_knowledge.resolver import IntegrationProfileVendorResolver

__all__ = [
    "IntegrationProfileVendorResolver",
    "KnowledgeAdapterCapabilities",
    "KnowledgeAdapterRegistry",
    "KnowledgeChange",
    "KnowledgeChangeKind",
    "KnowledgeContent",
    "KnowledgeContentMode",
    "KnowledgeCursor",
    "KnowledgeItemDescriptor",
    "KnowledgeItemIdentity",
    "KnowledgeItemProvenance",
    "KnowledgeItemRevision",
    "KnowledgePage",
    "KnowledgePermissions",
    "KnowledgePrincipal",
    "KnowledgeScopeInfo",
    "KnowledgeSourceRef",
    "KnowledgeSourceScope",
    "KnowledgeVisibility",
    "VendorIntegrationResolver",
    "VendorKnowledgeAdapter",
    "VendorKnowledgeError",
    "VendorKnowledgeErrorCode",
    "VendorKnowledgeFacade",
    "VendorKnowledgeFacadeService",
]
