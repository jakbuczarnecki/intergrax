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
from intergrax.runtime.vendor_knowledge.remote_resource_discovery import (
    RemoteResourceAvailabilityV1,
    RemoteResourceCandidatePageV1,
    RemoteResourceCandidateV1,
    RemoteResourceDescriptorV1,
    RemoteResourceDiscoveryPageV1,
    RemoteResourceDiscoveryProvider,
    RemoteResourceDiscoveryRegistry,
    TenantRemoteResourceDiscoveryService,
)
from intergrax.runtime.vendor_knowledge.resolver import IntegrationProfileVendorResolver
from intergrax.runtime.vendor_knowledge.sync_contracts import (
    KnowledgeRemoteItemStateRepository,
    KnowledgeSourceLeaseRepository,
    KnowledgeSyncCheckpointConflict,
    KnowledgeSyncCheckpointRepository,
    KnowledgeSyncCorruptState,
    KnowledgeSyncSink,
)
from intergrax.runtime.vendor_knowledge.sync_coordinator import (
    VendorKnowledgeSyncCoordinator,
)
from intergrax.runtime.vendor_knowledge.sync_document_store import (
    DocumentStoreKnowledgeRemoteItemStateRepository,
    DocumentStoreKnowledgeSourceLeaseRepository,
    DocumentStoreKnowledgeSyncCheckpointRepository,
)
from intergrax.runtime.vendor_knowledge.sync_jobs import (
    VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA,
    VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
    VendorKnowledgeSyncJob,
    VendorKnowledgeSyncScheduler,
    decode_vendor_knowledge_sync_job,
    encode_vendor_knowledge_sync_job,
    vendor_knowledge_sync_idempotency_key,
)
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
from intergrax.runtime.vendor_knowledge.sync_runtime import (
    VendorKnowledgeSyncRuntime,
    build_vendor_knowledge_sync_runtime,
)
from intergrax.runtime.vendor_knowledge.sync_task import (
    VendorKnowledgeSyncDispatcher,
    make_vendor_knowledge_sync_handler,
    owner_id_for_sync_run,
    register_vendor_knowledge_sync_handler,
)
from intergrax.runtime.vendor_knowledge.sync_worker import (
    VendorKnowledgeSyncWorkerOutput,
    make_vendor_knowledge_sync_worker_handler,
    register_vendor_knowledge_sync_worker_handler,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
    RepositoryTenantConnectionPort,
    TenantConnectionCapabilityReadService,
    TenantConnectionPort,
    TenantLiveCapabilityCatalog,
    TenantLiveCapabilityCatalogPort,
    is_bindable_read_only_capability,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
    DocumentStoreTenantConnectionRepository,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionIntegrationFactory,
    TenantConnectionRehydrationResult,
    TenantConnectionRehydrationStatus,
    TenantConnectionRehydrator,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnection,
    TenantConnectionAdministrativeStatus,
    TenantConnectionAlreadyExists,
    TenantConnectionCorruptRecord,
    TenantConnectionInvalidState,
    TenantConnectionNotFound,
    TenantConnectionRepository,
    TenantConnectionService,
    TenantConnectionVersionConflict,
    to_safe_tenant_connection,
)

__all__ = [
    "VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA",
    "VENDOR_KNOWLEDGE_SYNC_TASK_NAME",
    "CapabilityEffectV1",
    "ConnectionAwareVendorResolver",
    "DocumentStoreKnowledgeRemoteItemStateRepository",
    "DocumentStoreKnowledgeSourceBindingRepository",
    "DocumentStoreKnowledgeSourceLeaseRepository",
    "DocumentStoreKnowledgeSyncCheckpointRepository",
    "DocumentStoreTenantConnectionRepository",
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
    "LiveCapabilityDescriptorV1",
    "RemoteResourceAvailabilityV1",
    "RemoteResourceCandidatePageV1",
    "RemoteResourceCandidateV1",
    "RemoteResourceDescriptorV1",
    "RemoteResourceDiscoveryPageV1",
    "RemoteResourceDiscoveryProvider",
    "RemoteResourceDiscoveryRegistry",
    "RepositoryTenantConnectionPort",
    "SafeTenantConnectionV1",
    "TenantConnection",
    "TenantConnectionAdministrativeStatus",
    "TenantConnectionAlreadyExists",
    "TenantConnectionCapabilityReadService",
    "TenantConnectionCorruptRecord",
    "TenantConnectionIntegrationFactory",
    "TenantConnectionInvalidState",
    "TenantConnectionNotFound",
    "TenantConnectionPort",
    "TenantConnectionRehydrationResult",
    "TenantConnectionRehydrationStatus",
    "TenantConnectionRehydrator",
    "TenantConnectionRepository",
    "TenantConnectionService",
    "TenantConnectionVersionConflict",
    "TenantLiveCapabilityCatalog",
    "TenantLiveCapabilityCatalogPort",
    "TenantRemoteResourceDiscoveryService",
    "VendorIntegrationResolver",
    "VendorKnowledgeAdapter",
    "VendorKnowledgeError",
    "VendorKnowledgeErrorCode",
    "VendorKnowledgeFacade",
    "VendorKnowledgeFacadeService",
    "VendorKnowledgeSyncCoordinator",
    "VendorKnowledgeSyncDispatcher",
    "VendorKnowledgeSyncJob",
    "VendorKnowledgeSyncRuntime",
    "VendorKnowledgeSyncScheduler",
    "VendorKnowledgeSyncWorkerOutput",
    "build_vendor_knowledge_sync_runtime",
    "decode_vendor_knowledge_sync_job",
    "encode_vendor_knowledge_sync_job",
    "is_bindable_read_only_capability",
    "make_vendor_knowledge_sync_handler",
    "make_vendor_knowledge_sync_worker_handler",
    "owner_id_for_sync_run",
    "register_vendor_knowledge_sync_handler",
    "register_vendor_knowledge_sync_worker_handler",
    "to_safe_tenant_connection",
    "to_source_ref",
    "vendor_knowledge_sync_idempotency_key",
]
