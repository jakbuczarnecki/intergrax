# © Artur Czarnecki. All rights reserved.

"""Production host composition for shared Slack connected-source integration."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SlackConversationChannelIntegration,
)
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.workspaces.connected_source_wiring import (
    ConnectedSourceWiring,
    build_connected_source_wiring,
    register_slack_connection_integration,
)
from local_workspace_application.workspaces.document_indexing import WorkspaceDocumentIndexingService
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from local_workspace_application.workspaces.sync_runtime import ManagedWorkspaceSyncRuntime


@dataclass(frozen=True, slots=True)
class ConnectedSourceHostReadiness:
    signing_key_configured: bool
    slack_integration_available: bool
    mapping_complete: bool
    tenant_id: str | None
    connection_ref: str | None
    reason: str | None = None


@dataclass(slots=True)
class ConnectedSourceHostBundle:
    wiring: ConnectedSourceWiring | None
    slack_integration: SlackConversationChannelIntegration | None
    readiness: ConnectedSourceHostReadiness


def resolve_connected_source_host_mapping(
    settings: LocalWorkspaceBackendSettings,
) -> tuple[str | None, str | None]:
    tenant_id = settings.slack_tenant_id.strip() or None
    connection_ref = settings.connected_source_slack_connection_ref.strip() or None
    return tenant_id, connection_ref


def build_shared_slack_integration_for_host() -> SlackConversationChannelIntegration | None:
    try:
        platform_config = SlackConversationChannelIntegrationConfig.from_env(enabled=True)
        platform_config.validate_for_runtime()
        return SlackConversationChannelIntegration.from_config(platform_config)
    except Exception:
        return None


def build_connected_source_host_bundle(
    *,
    settings: LocalWorkspaceBackendSettings,
    repository: ManagedWorkspaceRepository,
    workspace_service: ManagedWorkspaceService,
    configuration_service: WorkspaceKnowledgeConfigurationService,
    mutation_engine: WorkspaceKnowledgeConfigurationMutationEngine,
    indexing_service: WorkspaceDocumentIndexingService,
    slack_integration: SlackConversationChannelIntegration | None = None,
    sync_runtime: ManagedWorkspaceSyncRuntime | None = None,
) -> ConnectedSourceHostBundle:
    signing_key_configured = bool(settings.connected_source_opaque_ref_signing_key.strip())
    if not signing_key_configured:
        return ConnectedSourceHostBundle(
            wiring=None,
            slack_integration=None,
            readiness=ConnectedSourceHostReadiness(
                signing_key_configured=False,
                slack_integration_available=False,
                mapping_complete=False,
                tenant_id=None,
                connection_ref=None,
                reason="connected_source_signing_key_missing",
            ),
        )

    tenant_id, connection_ref = resolve_connected_source_host_mapping(settings)
    integration = slack_integration or build_shared_slack_integration_for_host()
    if integration is None:
        return ConnectedSourceHostBundle(
            wiring=None,
            slack_integration=None,
            readiness=ConnectedSourceHostReadiness(
                signing_key_configured=True,
                slack_integration_available=False,
                mapping_complete=False,
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                reason="slack_integration_unavailable",
            ),
        )

    mapping_complete = bool(tenant_id and connection_ref)
    if not mapping_complete:
        return ConnectedSourceHostBundle(
            wiring=None,
            slack_integration=integration,
            readiness=ConnectedSourceHostReadiness(
                signing_key_configured=True,
                slack_integration_available=True,
                mapping_complete=False,
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                reason="connected_source_mapping_incomplete",
            ),
        )

    registry = KnowledgeConnectionRegistry()
    wiring = build_connected_source_wiring(
        repository=repository,
        workspace_service=workspace_service,
        configuration_service=configuration_service,
        mutation_engine=mutation_engine,
        indexing_service=indexing_service,
        settings=settings,
        connection_registry=registry,
        sync_runtime=sync_runtime,
    )
    register_slack_connection_integration(
        wiring=wiring,
        tenant_id=tenant_id,
        connection_ref=connection_ref,
        integration=integration,
    )
    return ConnectedSourceHostBundle(
        wiring=wiring,
        slack_integration=integration,
        readiness=ConnectedSourceHostReadiness(
            signing_key_configured=True,
            slack_integration_available=True,
            mapping_complete=True,
            tenant_id=tenant_id,
            connection_ref=connection_ref,
        ),
    )
