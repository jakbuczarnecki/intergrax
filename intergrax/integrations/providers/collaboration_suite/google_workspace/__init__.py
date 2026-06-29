# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID",
    "GoogleWorkspaceCollaborationSuiteIntegration",
    "GoogleWorkspaceCollaborationSuiteIntegrationConfig",
    "GoogleWorkspaceCollaborationSuiteClient",
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

_INTEGRATION_EXPORTS = frozenset(
    {
        "GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID",
        "GoogleWorkspaceCollaborationSuiteIntegration",
        "GoogleWorkspaceCollaborationSuiteIntegrationConfig",
        "GoogleWorkspaceCollaborationSuiteClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID",
        "GoogleWorkspaceCollaborationSuiteIntegration",
        "GoogleWorkspaceCollaborationSuiteIntegrationConfig",
        "GoogleWorkspaceCollaborationSuiteClient",
    }
)

def __getattr__(name: str):
    if name == "register_google_workspace_integration":
        from intergrax.integrations.providers.collaboration_suite.google_workspace.register import register_google_workspace_integration

        return register_google_workspace_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.collaboration_suite.google_workspace import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.collaboration_suite.google_workspace import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.collaboration_suite.google_workspace import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
