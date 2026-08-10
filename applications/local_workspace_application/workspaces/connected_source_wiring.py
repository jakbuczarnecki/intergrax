# © Artur Czarnecki. All rights reserved.

"""Composition helpers for connected workspace knowledge access."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.workspaces.connected_source_discovery import (
    WorkspaceRemoteResourceDiscoveryService,
)
from local_workspace_application.workspaces.connected_source_discovery_google_workspace import (
    GoogleWorkspaceKnownResourceCatalog,
)
from local_workspace_application.workspaces.connected_source_discovery_atlassian import (
    ConfluenceKnownSpaceCatalog,
    JiraKnownProjectCatalog,
)
from local_workspace_application.workspaces.connected_source_discovery_strategy import (
    RemoteResourceDiscoveryStrategyRegistry,
)
from local_workspace_application.workspaces.connected_source_materializer import (
    ConnectedSourceContentMaterializerRegistry,
)
from local_workspace_application.workspaces.connected_source_opaque_ref_codec import (
    RemoteResourceOpaqueRefCodec,
)
from local_workspace_application.workspaces.connected_source_sync_service import (
    ConnectedSourceSyncDependencies,
    ManagedWorkspaceConnectedSourceSyncService,
)
from local_workspace_application.workspaces.connected_source_tenant_binding import (
    ProviderNeutralConnectedSourceCandidateAdapter,
    SlackConnectedSourceCandidateAdapter,
    WorkspaceConnectedSourceTenantBindingService,
)
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingService,
)
from local_workspace_application.workspaces.knowledge_access_service import (
    TenantKnowledgeSourceBindingPort,
    WorkspaceKnowledgeAccessService,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.knowledge_indexed_source_lifecycle_service import (
    WorkspaceIndexedSourceLifecycleService,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from local_workspace_application.workspaces.sync_runtime import (
    ManagedWorkspaceSyncRuntime,
)

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.runtime.vendor_knowledge.contribution_catalog import (
    VendorKnowledgeContributionCatalog,
    build_vendor_knowledge_adapter_registry,
)
# Compatibility boundary retained for callers of build_default_vendor_knowledge_adapter_registry.
from intergrax.runtime.vendor_knowledge.binding_document_store import (
    DocumentStoreKnowledgeSourceBindingRepository,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingService,
)
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
from local_workspace_application.workspaces.vendor_knowledge_extension_composition import (
    VendorKnowledgeApplicationExtensionContext,
    build_default_vendor_knowledge_application_contribution_catalog,
)


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
    google_known_resource_catalog: GoogleWorkspaceKnownResourceCatalog
    jira_known_project_catalog: JiraKnownProjectCatalog
    confluence_known_space_catalog: ConfluenceKnownSpaceCatalog
    discovery_service: WorkspaceRemoteResourceDiscoveryService
    tenant_binding_service: WorkspaceConnectedSourceTenantBindingService
    tenant_binding_port: TenantKnowledgeSourceBindingPort
    indexed_source_lifecycle_service: WorkspaceIndexedSourceLifecycleService
    knowledge_access_service: WorkspaceKnowledgeAccessService
    connected_source_sync_service: ManagedWorkspaceConnectedSourceSyncService


def build_connected_source_opaque_ref_codec(
    settings: LocalWorkspaceBackendSettings,
) -> RemoteResourceOpaqueRefCodec:
    return RemoteResourceOpaqueRefCodec.from_signing_key_material(
        settings.connected_source_opaque_ref_signing_key
    )


def build_default_remote_resource_discovery_registry(
    *,
    connection_registry: KnowledgeConnectionRegistry,
    opaque_ref_codec: RemoteResourceOpaqueRefCodec,
    google_known_resource_catalog: GoogleWorkspaceKnownResourceCatalog,
    jira_known_project_catalog: JiraKnownProjectCatalog,
    confluence_known_space_catalog: ConfluenceKnownSpaceCatalog,
    msgraph_mailbox_user_id: str | None,
    msgraph_teams_channel_team_id: str | None = None,
    contribution_catalog: VendorKnowledgeContributionCatalog | None = None,
    discover_entry_points: bool = False,
) -> RemoteResourceDiscoveryStrategyRegistry:
    """Compose discovery strategies from application-owned contribution hooks."""
    context = VendorKnowledgeApplicationExtensionContext(
        connection_registry=connection_registry,
        opaque_ref_codec=opaque_ref_codec,
        google_known_resource_catalog=google_known_resource_catalog,
        jira_known_project_catalog=jira_known_project_catalog,
        confluence_known_space_catalog=confluence_known_space_catalog,
        msgraph_mailbox_user_id=msgraph_mailbox_user_id,
        msgraph_teams_channel_team_id=msgraph_teams_channel_team_id,
    )
    catalog = contribution_catalog or build_default_vendor_knowledge_application_contribution_catalog(
        context,
        discover_entry_points=discover_entry_points,
    )
    strategies = tuple(
        hook.factory(context)
        for contribution in catalog.list_contributions()
        for hook in contribution.discovery_contributions
    )
    return RemoteResourceDiscoveryStrategyRegistry(strategies)


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
    materializer_registry: ConnectedSourceContentMaterializerRegistry | None = None,
    google_known_resource_catalog: GoogleWorkspaceKnownResourceCatalog | None = None,
    jira_known_project_catalog: JiraKnownProjectCatalog | None = None,
    confluence_known_space_catalog: ConfluenceKnownSpaceCatalog | None = None,
    msgraph_mailbox_user_id: str | None = None,
    msgraph_teams_channel_team_id: str | None = None,
    discover_vendor_knowledge_entry_points: bool = False,
) -> ConnectedSourceWiring:
    registry = connection_registry or KnowledgeConnectionRegistry()
    codec = opaque_ref_codec or build_connected_source_opaque_ref_codec(settings)
    google_resources = google_known_resource_catalog or GoogleWorkspaceKnownResourceCatalog()
    jira_projects = jira_known_project_catalog or JiraKnownProjectCatalog()
    confluence_spaces = confluence_known_space_catalog or ConfluenceKnownSpaceCatalog()
    contribution_context = VendorKnowledgeApplicationExtensionContext(
        connection_registry=registry,
        opaque_ref_codec=codec,
        google_known_resource_catalog=google_resources,
        jira_known_project_catalog=jira_projects,
        confluence_known_space_catalog=confluence_spaces,
        msgraph_mailbox_user_id=msgraph_mailbox_user_id,
        msgraph_teams_channel_team_id=msgraph_teams_channel_team_id,
    )
    contribution_catalog = build_default_vendor_knowledge_application_contribution_catalog(
        contribution_context,
        discover_entry_points=discover_vendor_knowledge_entry_points,
    )
    discovery_strategy_registry = build_default_remote_resource_discovery_registry(
        connection_registry=registry,
        opaque_ref_codec=codec,
        google_known_resource_catalog=google_resources,
        jira_known_project_catalog=jira_projects,
        confluence_known_space_catalog=confluence_spaces,
        msgraph_mailbox_user_id=msgraph_mailbox_user_id,
        msgraph_teams_channel_team_id=msgraph_teams_channel_team_id,
        contribution_catalog=contribution_catalog,
        discover_entry_points=discover_vendor_knowledge_entry_points,
    )
    discovery = WorkspaceRemoteResourceDiscoveryService(
        workspace_lookup=workspace_service,
        configuration_reader=configuration_service,
        opaque_ref_codec=codec,
        strategy_registry=discovery_strategy_registry,
    )
    candidate_adapter = SlackConnectedSourceCandidateAdapter(
        codec=codec,
        discovery_service=discovery,
    )
    candidate_dispatcher = ProviderNeutralConnectedSourceCandidateAdapter(
        slack=candidate_adapter,
        codec=codec,
        discovery_service=discovery,
    )
    adapter_registry = build_vendor_knowledge_adapter_registry(contribution_catalog)
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
    indexed_source_lifecycle = WorkspaceIndexedSourceLifecycleService(
        repository=repository,
        configuration_service=configuration_service,
        mutation_engine=mutation_engine,
        tenant_binding_port=tenant_binding_port,
    )

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
        materializer_registry=materializer_registry,
    )
    knowledge_access = WorkspaceKnowledgeAccessService(
        discovery_service=discovery,
        tenant_binding_service=tenant_binding_service,
        indexed_source_lifecycle_service=indexed_source_lifecycle,
        workspace_service=workspace_service,
        tenant_binding_port=tenant_binding_port,
        candidate_adapter=candidate_dispatcher,
    )
    return ConnectedSourceWiring(
        connection_registry=registry,
        opaque_ref_codec=codec,
        google_known_resource_catalog=google_resources,
        jira_known_project_catalog=jira_projects,
        confluence_known_space_catalog=confluence_spaces,
        discovery_service=discovery,
        tenant_binding_service=tenant_binding_service,
        tenant_binding_port=tenant_binding_port,
        indexed_source_lifecycle_service=indexed_source_lifecycle,
        knowledge_access_service=knowledge_access,
        connected_source_sync_service=connected_sync,
    )


class _SyncRuntimeContinuation:
    def __init__(self, runtime: ManagedWorkspaceSyncRuntime) -> None:
        self._runtime = runtime

    def requeue(self, job) -> None:
        from local_workspace_application.workspaces.sync_enqueue import (
            enqueue_managed_workspace_sync,
        )

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


def register_msgraph_connection_integration(
    *,
    wiring: ConnectedSourceWiring,
    tenant_id: str,
    connection_ref: str,
    integration,
) -> None:
    wiring.connection_registry.register(
        tenant_id=tenant_id,
        connection_ref=connection_ref,
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        integration=integration,
    )
