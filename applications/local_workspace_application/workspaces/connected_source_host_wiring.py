# © Artur Czarnecki. All rights reserved.

"""Production host composition for provider-neutral connected sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.live.bootstrap import (
    build_vendor_knowledge_live_registration_registry,
)
from intergrax.runtime.vendor_knowledge.plugin_composition import (
    build_default_vendor_knowledge_source_plugin_registry,
)
from intergrax.runtime.vendor_knowledge.provider_composition import (
    build_default_vendor_knowledge_connection_factory_registry,
    build_default_vendor_knowledge_legacy_local_bootstrap,
    VendorKnowledgeLegacyLocalBootstrap,
)
from intergrax.runtime.vendor_knowledge.source_catalog import (
    TenantVendorKnowledgeSourceCatalog,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    RepositoryTenantConnectionPort,
    TenantConnectionPort,
    TenantLiveCapabilityCatalog,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
    DocumentStoreTenantConnectionRepository,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionIntegrationFactory,
    TenantConnectionRehydrator,
    TenantConnectionRehydrationStatus,
)
from intergrax.integrations.contracts.secrets_store import SecretsStore
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceReadinessState,
)
from local_workspace_application.workspaces.connected_source_wiring import (
    ConnectedSourceWiring,
    build_connected_source_wiring,
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
    state: ConnectedSourceReadinessState
    signing_key_configured: bool
    # Deprecated internal compatibility alias; use connection_runtime_available.
    slack_integration_available: bool
    mapping_complete: bool
    tenant_id: str | None
    connection_ref: str | None
    reason: str | None = None
    tenant_connections_rehydrated: bool = False

    @property
    def connection_runtime_available(self) -> bool:
        """Provider-neutral readiness view retained over the legacy field."""
        return self.slack_integration_available


@dataclass(slots=True)
class ConnectedSourceHostBundle:
    wiring: ConnectedSourceWiring | None
    slack_integration: Any | None
    readiness: ConnectedSourceHostReadiness
    hybrid_ask_connection_registry: KnowledgeConnectionRegistry
    tenant_connection_port: TenantConnectionPort | None = None
    tenant_live_capability_catalog: TenantLiveCapabilityCatalog | None = None
    tenant_source_catalog: TenantVendorKnowledgeSourceCatalog | None = None

    @property
    def legacy_local_integration(self) -> Any | None:
        """Provider-neutral compatibility view of the legacy local runtime."""
        return self.slack_integration


def connected_source_considered_for_host(settings: LocalWorkspaceBackendSettings) -> bool:
    """Return whether generic or legacy connected-source bootstrap is configured."""
    return bool(
        settings.tenant_connection_bootstrap_tenant_ids
        or settings.connected_source_opaque_ref_signing_key.strip()
        or _legacy_local_bootstrap_considered(settings)
    )


# LEGACY_LOCAL_BOOTSTRAP: these settings are compatibility-only and are not
# consulted by the durable provider-neutral bootstrap below.
def _legacy_local_bootstrap_considered(settings: LocalWorkspaceBackendSettings) -> bool:
    return bool(
        settings.slack_companion_enabled
        or settings.slack_tenant_id.strip()
        or settings.connected_source_slack_connection_ref.strip()
    )


def resolve_connected_source_host_mapping(
    settings: LocalWorkspaceBackendSettings,
) -> tuple[str | None, str | None]:
    """Read the legacy local mapping; durable lifecycle remains repository-owned."""
    tenant_id = settings.slack_tenant_id.strip() or None
    connection_ref = settings.connected_source_slack_connection_ref.strip() or None
    return tenant_id, connection_ref


def _bootstrap_from_legacy_integration(
    integration: Any | None,
) -> VendorKnowledgeLegacyLocalBootstrap | None:
    if integration is None:
        return None
    provider_id = getattr(integration, "provider_id", None)
    integration_kind = getattr(integration, "integration_kind", None)
    if not isinstance(provider_id, str) or not provider_id.strip():
        return None
    if isinstance(integration_kind, str):
        try:
            integration_kind = IntegrationCategory(integration_kind)
        except ValueError:
            return None
    if not isinstance(integration_kind, IntegrationCategory):
        return None
    return VendorKnowledgeLegacyLocalBootstrap(
        provider_id=provider_id.strip(),
        integration_kind=integration_kind,
        integration=integration,
    )


def build_connected_source_host_bundle(
    *,
    settings: LocalWorkspaceBackendSettings,
    repository: ManagedWorkspaceRepository,
    workspace_service: ManagedWorkspaceService,
    configuration_service: WorkspaceKnowledgeConfigurationService,
    mutation_engine: WorkspaceKnowledgeConfigurationMutationEngine,
    indexing_service: WorkspaceDocumentIndexingService,
    legacy_local_bootstrap: VendorKnowledgeLegacyLocalBootstrap | None = None,
    legacy_local_integration: Any | None = None,
    # Compatibility alias for callers from the pre-neutral composition boundary.
    slack_integration: Any | None = None,
    sync_runtime: ManagedWorkspaceSyncRuntime | None = None,
    tenant_connection_secrets_store: SecretsStore | None = None,
    tenant_connection_factory_registry: TenantConnectionIntegrationFactory | None = None,
    msgraph_mailbox_user_id: str | None = None,
    msgraph_teams_channel_team_id: str | None = None,
    discover_vendor_knowledge_entry_points: bool = False,
) -> ConnectedSourceHostBundle:
    registry = KnowledgeConnectionRegistry()
    connection_repository = DocumentStoreTenantConnectionRepository(repository.document_store)
    connection_port = RepositoryTenantConnectionPort(connection_repository)
    live_capability_catalog = TenantLiveCapabilityCatalog(
        connection_port=connection_port,
    )
    build_vendor_knowledge_live_registration_registry(
        discover_entry_points=discover_vendor_knowledge_entry_points,
    ).publish_to_tenant_catalog(
        live_capability_catalog,
    )
    source_catalog = TenantVendorKnowledgeSourceCatalog(
        connection_port=connection_port,
        plugin_registry=build_default_vendor_knowledge_source_plugin_registry(
            discover_entry_points=discover_vendor_knowledge_entry_points,
        ),
    )
    factory_registry = (
        tenant_connection_factory_registry
        or build_default_vendor_knowledge_connection_factory_registry(
            discover_entry_points=discover_vendor_knowledge_entry_points,
        )
    )
    legacy_local_integration = legacy_local_integration or slack_integration
    legacy_local_bootstrap = legacy_local_bootstrap or _bootstrap_from_legacy_integration(
        legacy_local_integration
    )
    production_tenant_ids = tuple(
        dict.fromkeys(
            tenant_id.strip()
            for tenant_id in settings.tenant_connection_bootstrap_tenant_ids
            if tenant_id.strip()
        )
    )
    if production_tenant_ids:
        tenant_id = None
        connection_ref = None
    else:
        tenant_id, connection_ref = resolve_connected_source_host_mapping(settings)
    integration = legacy_local_integration
    registered_results = []

    if not connected_source_considered_for_host(settings):
        return ConnectedSourceHostBundle(
            wiring=None,
            slack_integration=None,
            readiness=ConnectedSourceHostReadiness(
                state=ConnectedSourceReadinessState.DISABLED,
                signing_key_configured=False,
                slack_integration_available=False,
                mapping_complete=False,
                tenant_id=None,
                connection_ref=None,
                reason="connected_source_disabled",
            ),
            hybrid_ask_connection_registry=registry,
        )

    signing_key_configured = bool(settings.connected_source_opaque_ref_signing_key.strip())
    if not signing_key_configured:
        return ConnectedSourceHostBundle(
            wiring=None,
            slack_integration=None,
            readiness=ConnectedSourceHostReadiness(
                state=ConnectedSourceReadinessState.SIGNING_KEY_MISSING,
                signing_key_configured=False,
                slack_integration_available=False,
                mapping_complete=False,
                tenant_id=None,
                connection_ref=None,
                reason="connected_source_signing_key_missing",
            ),
            hybrid_ask_connection_registry=registry,
        )

    if tenant_connection_secrets_store is not None:
        rehydrator = TenantConnectionRehydrator(
            repository=connection_repository,
            secrets_store=tenant_connection_secrets_store,
            integration_factory=factory_registry,
            connection_registry=registry,
        )
        if production_tenant_ids:
            for production_tenant_id in production_tenant_ids:
                try:
                    rehydration = rehydrator.rehydrate_tenant(
                        tenant_id=production_tenant_id,
                    )
                except Exception:
                    continue
                registered_results.extend(
                    result
                    for result in rehydration
                    if result.status is TenantConnectionRehydrationStatus.REGISTERED
                )
        elif tenant_id and connection_ref:
            persisted_connection = connection_repository.get(
                tenant_id=tenant_id,
                connection_ref=connection_ref,
            )
            if persisted_connection is not None:
                rehydration = rehydrator.rehydrate_tenant(tenant_id=tenant_id)
                target_rehydration = next(
                    (
                        result
                        for result in rehydration
                        if (
                            result.connection.tenant_id == tenant_id
                            and result.connection.connection_ref == connection_ref
                            and result.connection.provider_id == persisted_connection.provider_id
                            and result.connection.integration_kind
                            == persisted_connection.integration_kind
                            and result.status is TenantConnectionRehydrationStatus.REGISTERED
                        )
                    ),
                    None,
                )
                if target_rehydration is not None:
                    registered_results.append(target_rehydration)
                    integration = registry.resolve(
                        tenant_id=tenant_id,
                        connection_ref=connection_ref,
                        provider_id=persisted_connection.provider_id,
                        integration_kind=persisted_connection.integration_kind,
                    )

    rehydrated = bool(registered_results)
    if rehydrated:
        first_registered = registered_results[0].connection
        if production_tenant_ids:
            integration = None
            tenant_id = first_registered.tenant_id
            connection_ref = first_registered.connection_ref
        else:
            # A durable legacy target remains the compatibility projection.
            integration = integration or registry.resolve(
                tenant_id=first_registered.tenant_id,
                connection_ref=first_registered.connection_ref,
                provider_id=first_registered.provider_id,
                integration_kind=first_registered.integration_kind,
            )

    # LEGACY_LOCAL_BOOTSTRAP: local injected/env Slack compatibility fallback.
    if not rehydrated and not production_tenant_ids:
        legacy_local_bootstrap = (
            legacy_local_bootstrap
            or build_default_vendor_knowledge_legacy_local_bootstrap()
        )
        integration = integration or (
            legacy_local_bootstrap.integration
            if legacy_local_bootstrap is not None
            else None
        )

    if integration is None and not rehydrated:
        return ConnectedSourceHostBundle(
            wiring=None,
            slack_integration=None,
            readiness=ConnectedSourceHostReadiness(
                state=ConnectedSourceReadinessState.SLACK_INTEGRATION_UNAVAILABLE,
                signing_key_configured=True,
                slack_integration_available=False,
                mapping_complete=False,
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                reason=(
                    "tenant_connection_runtime_unavailable"
                    if production_tenant_ids
                    else "slack_integration_unavailable"
                ),
            ),
            hybrid_ask_connection_registry=registry,
        )

    mapping_complete = bool(tenant_id and connection_ref)
    if not rehydrated and not mapping_complete:
        return ConnectedSourceHostBundle(
            wiring=None,
            slack_integration=integration,
            readiness=ConnectedSourceHostReadiness(
                state=ConnectedSourceReadinessState.MAPPING_INCOMPLETE,
                signing_key_configured=True,
                slack_integration_available=True,
                mapping_complete=False,
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                reason="connected_source_mapping_incomplete",
            ),
            hybrid_ask_connection_registry=registry,
        )

    wiring = build_connected_source_wiring(
        repository=repository,
        workspace_service=workspace_service,
        configuration_service=configuration_service,
        mutation_engine=mutation_engine,
        indexing_service=indexing_service,
        settings=settings,
        connection_registry=registry,
        sync_runtime=sync_runtime,
        msgraph_mailbox_user_id=msgraph_mailbox_user_id,
        msgraph_teams_channel_team_id=msgraph_teams_channel_team_id,
        discover_vendor_knowledge_entry_points=discover_vendor_knowledge_entry_points,
    )
    if not rehydrated:
        legacy_local_bootstrap = legacy_local_bootstrap or _bootstrap_from_legacy_integration(
            integration
        )
        if legacy_local_bootstrap is None:
            return ConnectedSourceHostBundle(
                wiring=None,
                slack_integration=None,
                readiness=ConnectedSourceHostReadiness(
                    state=ConnectedSourceReadinessState.SLACK_INTEGRATION_UNAVAILABLE,
                    signing_key_configured=True,
                    slack_integration_available=False,
                    mapping_complete=False,
                    tenant_id=tenant_id,
                    connection_ref=connection_ref,
                    reason="connection_runtime_identity_unavailable",
                ),
                hybrid_ask_connection_registry=registry,
            )
        registry.register(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
            provider_id=legacy_local_bootstrap.provider_id,
            integration_kind=legacy_local_bootstrap.integration_kind,
            integration=integration,
        )
    return ConnectedSourceHostBundle(
        wiring=wiring,
        slack_integration=integration,
        readiness=ConnectedSourceHostReadiness(
            state=ConnectedSourceReadinessState.READY,
            signing_key_configured=True,
            slack_integration_available=rehydrated or integration is not None,
            mapping_complete=mapping_complete or rehydrated,
            tenant_id=tenant_id,
            connection_ref=connection_ref,
            tenant_connections_rehydrated=rehydrated,
        ),
        hybrid_ask_connection_registry=registry,
        tenant_connection_port=connection_port,
        tenant_live_capability_catalog=live_capability_catalog,
        tenant_source_catalog=source_catalog,
    )
