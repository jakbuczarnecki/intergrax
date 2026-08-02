# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID",
    "GOOGLE_WORKSPACE_SUPPORTED_SOURCE_KINDS",
    "GoogleWorkspaceClientFactory",
    "GoogleWorkspaceCollaborationSuiteIntegration",
    "GoogleWorkspaceCollaborationSuiteIntegrationConfig",
    "GoogleWorkspaceCredentialResolver",
    "GoogleWorkspaceSourceKind",
    "create_google_workspace_collaboration_suite",
    "create_google_workspace_collaboration_suite_integration",
    "register_google_workspace_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_google_workspace_collaboration_suite",
        "create_google_workspace_collaboration_suite_integration",
    }
)

_FOUNDATION_EXPORTS = frozenset(
    {
        "GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID",
        "GOOGLE_WORKSPACE_SUPPORTED_SOURCE_KINDS",
        "GoogleWorkspaceClientFactory",
        "GoogleWorkspaceCollaborationSuiteIntegration",
        "GoogleWorkspaceCollaborationSuiteIntegrationConfig",
        "GoogleWorkspaceCredentialResolver",
        "GoogleWorkspaceSourceKind",
    }
)


def __getattr__(name: str):
    if name == "register_google_workspace_integration":
        from intergrax.integrations.providers.collaboration_suite.google_workspace.register import (
            register_google_workspace_integration,
        )

        return register_google_workspace_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.collaboration_suite.google_workspace import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _FOUNDATION_EXPORTS:
        if name == "GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID":
            from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
                GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            )

            return GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID
        if name == "GOOGLE_WORKSPACE_SUPPORTED_SOURCE_KINDS":
            from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
                GOOGLE_WORKSPACE_SUPPORTED_SOURCE_KINDS,
            )

            return GOOGLE_WORKSPACE_SUPPORTED_SOURCE_KINDS
        if name in {
            "GoogleWorkspaceClientFactory",
            "GoogleWorkspaceCredentialResolver",
            "GoogleWorkspaceSourceKind",
        }:
            from intergrax.integrations.providers.collaboration_suite.google_workspace import (
                contracts as _contracts,
            )

            return export_from_bundle(_contracts, name, _FOUNDATION_EXPORTS)
        if name in {
            "GoogleWorkspaceCollaborationSuiteIntegration",
            "GoogleWorkspaceCollaborationSuiteIntegrationConfig",
        }:
            from intergrax.integrations.providers.collaboration_suite.google_workspace import (
                integration as _integration,
            )

            return export_from_bundle(_integration, name, _FOUNDATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
