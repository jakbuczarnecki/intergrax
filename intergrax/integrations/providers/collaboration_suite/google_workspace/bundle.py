# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_google_workspace_collaboration_suite as _legacy_create_google_workspace_collaboration_suite

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
    GoogleWorkspaceCollaborationSuiteIntegrationConfig,
    GoogleWorkspaceCollaborationSuiteClient,
)

__all__ = [
    "create_google_workspace_collaboration_suite",
    "create_google_workspace_collaboration_suite_integration",
]


def create_google_workspace_collaboration_suite_integration(
    *,
    client: GoogleWorkspaceCollaborationSuiteClient | None = None,
    enabled: bool = False,
) -> GoogleWorkspaceCollaborationSuiteIntegration:
    """
    Build a contract-based Google Workspace collaboration suite integration.

    The legacy facade (create_google_workspace_collaboration_suite) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Google Workspace collaboration suite integration requires an injected client when enabled=True",
        )
    if client is not None:
        return GoogleWorkspaceCollaborationSuiteIntegration.from_client(client, enabled=enabled)
    return GoogleWorkspaceCollaborationSuiteIntegration.for_provider(
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        display_name="Google Workspace",
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(enabled=enabled),
    )


def create_google_workspace_collaboration_suite(**kwargs: object) -> GoogleWorkspaceCollaborationSuiteIntegration:
    """Compatibility shim — constructs GoogleWorkspaceCollaborationSuiteIntegration from legacy runtime."""
    runtime = _legacy_create_google_workspace_collaboration_suite(**kwargs)
    if isinstance(runtime, GoogleWorkspaceCollaborationSuiteIntegration):
        return runtime
    return GoogleWorkspaceCollaborationSuiteIntegration.from_runtime(runtime)
