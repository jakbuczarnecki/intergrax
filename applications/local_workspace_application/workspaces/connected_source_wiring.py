# © Artur Czarnecki. All rights reserved.

"""Composition helpers for connected workspace knowledge access."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.runtime.vendor_knowledge.adapters.slack_conversation import (
    register_slack_conversation_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.binding_document_store import (
    DocumentStoreKnowledgeSourceBindingRepository,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingService,
)
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.workspaces.connected_source_discovery import (
    WorkspaceRemoteResourceDiscoveryService,
)
from local_workspace_application.workspaces.connected_source_opaque_ref_codec import (
    RemoteResourceOpaqueRefCodec,
)
from local_workspace_application.workspaces.connected_source_sync_service import (
    ConnectedSourceSyncDependencies,
    ManagedWorkspaceConnectedSourceSyncService,
)
from local_workspace_application.workspaces.connected_source_tenant_binding import (
    WorkspaceConnectedSourceTenantBindingService,
)
from local_workspace_application.workspaces.document_indexing import WorkspaceDocumentIndexingService
from local_workspace_application.workspaces.knowledge_access_service import (
    WorkspaceKnowledgeAccessService,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from local_workspace_application.workspaces.sync_runtime import ManagedWorkspaceSyncRuntime


class _ConnectionAwareResolver:
    def __init__(self, registry: KnowledgeConnectionRegistry) -> None:
        self._registry = registry

    def resolve(self, *, source):
        return self._registry.resolve(
            tenant_id=source.tenant_id,
            connection_ref=source.connection_ref,
            provider_id=source.provider_id,
            integration_kind=source.integration_kind,
        )


class _TenantBindingPort:
    def __init__(
        self,
        binding_service_factory: Callable[[str], KnowledgeSourceBindingService],
    ) -> None:
        self._binding_service_factory = binding_service_factory

    def get_binding(self, *, tenant_id: str, binding_id: str) -> KnowledgeSourceBinding | None:
        return self._binding_service_factory(tenant_id).get(binding_id)


@dataclass(slots=True)
class ConnectedSourceWiring:
    connection_registry: KnowledgeConnectionRegistry
    opaque_ref_codec: RemoteResourceOpaqueRefCodec
    discovery_service: WorkspaceRemoteResourceDiscoveryService
    tenant_binding_service: WorkspaceConnectedSourceTenantBindingService
    knowledge_access_service: WorkspaceKnowledgeAccessService
    connected_source_sync_service: ManagedWorkspaceConnectedSourceSyncService


def build_connected_source_opaque_ref_codec(
    settings: LocalWorkspaceBackendSettings,
) -> RemoteResourceOpaqueRefCodec:
    return RemoteResourceOpaqueRefCodec.from_signing_key_material(
        settings.connected_source_opaque_ref_signing_key
    )


def build_connected_source_wiring(
    *,
    repository: ManagedWorkspaceRepository,
    workspace_service: ManagedWorkspaceService,
    configuration_service: WorkspaceKnowledgeConfigurationService,
    mutation_engine: WorkspaceKnowledgeConfigurationMutationEngine,
    indexing_service: WorkspaceDocumentIndexingService,
    settings: LocalWorkspaceBackendSettings,
    connection_registry: KnowledgeConnectionRegistry | None = None,
    opaque_ref_codec: RemoteResourceOpaqueRefCodec | None = None,
    sync_runtime: ManagedWorkspaceSyncRuntime | None = None,
) -> ConnectedSourceWiring:
    registry = connection_registry or KnowledgeConnectionRegistry()
    codec = opaque_ref_codec or build_connected_source_opaque_ref_codec(settings)
    discovery = WorkspaceRemoteResourceDiscoveryService(
        workspace_lookup=workspace_service,
        configuration_reader=configuration_service,
        connection_registry=registry,
        opaque_ref_codec=codec,
    )
    adapter_registry = KnowledgeAdapterRegistry()
    register_slack_conversation_knowledge_adapter(adapter_registry)
    binding_repo = DocumentStoreKnowledgeSourceBindingRepository(repository.document_store)
    resolver = _ConnectionAwareResolver(registry)

    def binding_service_factory(tenant_id: str) -> KnowledgeSourceBindingService:
        return KnowledgeSourceBindingService(
            tenant_id=tenant_id,
            repository=binding_repo,
            integration_resolver=resolver,
            adapter_registry=adapter_registry,
        )

    tenant_binding_service = WorkspaceConnectedSourceTenantBindingService(binding_service_factory)
    tenant_binding_port = _TenantBindingPort(binding_service_factory)

    def dependencies_factory(tenant_id: str) -> ConnectedSourceSyncDependencies:
        return ConnectedSourceSyncDependencies(
            binding_service=binding_service_factory(tenant_id),
            facade=VendorKnowledgeFacadeService(
                tenant_id=tenant_id,
                resolver=resolver,
                adapter_registry=adapter_registry,
            ),
            owner_id=f"lkw-connected-source:{tenant_id}",
        )

    continuation = None
    sync_enqueue_context = None
    if sync_runtime is not None:
        continuation = _SyncRuntimeContinuation(sync_runtime)
        sync_enqueue_context = sync_runtime.wiring_context

    connected_sync = ManagedWorkspaceConnectedSourceSyncService(
        repository=repository,
        indexing_service=indexing_service,
        configuration_reader=configuration_service,
        tenant_binding_port=tenant_binding_port,
        dependencies_factory=dependencies_factory,
        continuation=continuation,
        sync_enqueue_context=sync_enqueue_context,
    )
    knowledge_access = WorkspaceKnowledgeAccessService(
        discovery_service=discovery,
        tenant_binding_service=tenant_binding_service,
        mutation_engine=mutation_engine,
        workspace_service=workspace_service,
        tenant_binding_port=tenant_binding_port,
        opaque_ref_codec=codec,
    )
    return ConnectedSourceWiring(
        connection_registry=registry,
        opaque_ref_codec=codec,
        discovery_service=discovery,
        tenant_binding_service=tenant_binding_service,
        knowledge_access_service=knowledge_access,
        connected_source_sync_service=connected_sync,
    )


class _SyncRuntimeContinuation:
    def __init__(self, runtime: ManagedWorkspaceSyncRuntime) -> None:
        self._runtime = runtime

    def requeue(self, job) -> None:
        from local_workspace_application.workspaces.sync_enqueue import enqueue_managed_workspace_sync

        enqueue_managed_workspace_sync(self._runtime.wiring_context, job)


def register_slack_connection_integration(
    *,
    wiring: ConnectedSourceWiring,
    tenant_id: str,
    connection_ref: str,
    integration,
) -> None:
    wiring.connection_registry.register(
        tenant_id=tenant_id,
        connection_ref=connection_ref,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        integration=integration,
    )
