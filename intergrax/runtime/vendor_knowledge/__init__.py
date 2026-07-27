# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vendor Knowledge Facade — contracts and core services."""

from intergrax.runtime.vendor_knowledge.binding_document_store import (
    DocumentStoreKnowledgeSourceBindingRepository,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingRepository,
    KnowledgeSourceBindingService,
    KnowledgeSourceBindingStatus,
    to_source_ref,
)
from intergrax.runtime.vendor_knowledge.connections import (
    ConnectionAwareVendorResolver,
    KnowledgeConnectionRegistry,
)
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
from intergrax.runtime.vendor_knowledge.sync_contracts import (
    KnowledgeRemoteItemStateRepository,
    KnowledgeSourceLeaseRepository,
    KnowledgeSyncCheckpointConflict,
    KnowledgeSyncCheckpointRepository,
    KnowledgeSyncCorruptState,
    KnowledgeSyncSink,
)
from intergrax.runtime.vendor_knowledge.sync_coordinator import VendorKnowledgeSyncCoordinator
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeRemoteItemState,
    KnowledgeRemoteItemStatus,
    KnowledgeSourceLeaseToken,
    KnowledgeSyncBatch,
    KnowledgeSyncCheckpoint,
    KnowledgeSyncEnvelope,
    KnowledgeSyncMode,
    KnowledgeSyncRunResult,
    KnowledgeSyncRunStatus,
)

__all__ = [
    "ConnectionAwareVendorResolver",
    "DocumentStoreKnowledgeSourceBindingRepository",
    "IntegrationProfileVendorResolver",
    "KnowledgeAdapterCapabilities",
    "KnowledgeAdapterRegistry",
    "KnowledgeChange",
    "KnowledgeChangeKind",
    "KnowledgeConnectionRegistry",
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
    "KnowledgeRemoteItemState",
    "KnowledgeRemoteItemStateRepository",
    "KnowledgeRemoteItemStatus",
    "KnowledgeScopeInfo",
    "KnowledgeSourceBinding",
    "KnowledgeSourceBindingRepository",
    "KnowledgeSourceBindingService",
    "KnowledgeSourceBindingStatus",
    "KnowledgeSourceLeaseRepository",
    "KnowledgeSourceLeaseToken",
    "KnowledgeSourceRef",
    "KnowledgeSourceScope",
    "KnowledgeSyncBatch",
    "KnowledgeSyncCheckpoint",
    "KnowledgeSyncCheckpointConflict",
    "KnowledgeSyncCheckpointRepository",
    "KnowledgeSyncCorruptState",
    "KnowledgeSyncEnvelope",
    "KnowledgeSyncMode",
    "KnowledgeSyncRunResult",
    "KnowledgeSyncRunStatus",
    "KnowledgeSyncSink",
    "KnowledgeVisibility",
    "VendorIntegrationResolver",
    "VendorKnowledgeAdapter",
    "VendorKnowledgeError",
    "VendorKnowledgeErrorCode",
    "VendorKnowledgeFacade",
    "VendorKnowledgeFacadeService",
    "VendorKnowledgeSyncCoordinator",
    "to_source_ref",
]
