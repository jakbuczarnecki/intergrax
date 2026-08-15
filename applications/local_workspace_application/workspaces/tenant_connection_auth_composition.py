# © Artur Czarnecki. All rights reserved.

"""Composition root for tenant connection auth providers (PRODUCT-5B)."""

from __future__ import annotations

from intergrax.integrations.providers.collaboration_suite.google_workspace.tenant_connection_auth import (
    GoogleWorkspaceOAuthConfig,
    GoogleWorkspaceTenantConnectionAuthProvider,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.tenant_connection_auth import (
    Ms365GraphOAuthConfig,
    Ms365GraphTenantConnectionAuthProvider,
)
from intergrax.integrations.providers.conversation_channel.slack.tenant_connection_auth import (
    SlackTenantConnectionAuthProvider,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_auth import (
    TenantConnectionAuthProviderRegistry,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings


def build_tenant_connection_auth_provider_registry(
    settings: LocalWorkspaceBackendSettings,
) -> TenantConnectionAuthProviderRegistry:
    registry = TenantConnectionAuthProviderRegistry()
    registry.register(SlackTenantConnectionAuthProvider())

    google_config: GoogleWorkspaceOAuthConfig | None = None
    if settings.google_workspace_oauth_client_id.strip():
        google_config = GoogleWorkspaceOAuthConfig(
            client_id=settings.google_workspace_oauth_client_id.strip(),
            client_secret=settings.google_workspace_oauth_client_secret.strip(),
        )
    registry.register(GoogleWorkspaceTenantConnectionAuthProvider(google_config))

    ms365_config: Ms365GraphOAuthConfig | None = None
    if settings.ms365_oauth_client_id.strip() and settings.ms365_oauth_tenant_id.strip():
        ms365_config = Ms365GraphOAuthConfig(
            tenant_id=settings.ms365_oauth_tenant_id.strip(),
            client_id=settings.ms365_oauth_client_id.strip(),
            client_secret=settings.ms365_oauth_client_secret.strip(),
        )
    registry.register(Ms365GraphTenantConnectionAuthProvider(ms365_config))
    return registry


__all__ = ["build_tenant_connection_auth_provider_registry"]
