# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "ABSOLUTE_GOOGLE_DRIVE_CONTENT_MAX_BYTES",
    "DefaultGoogleWorkspaceClientFactory",
    "DEFAULT_GOOGLE_DRIVE_CONTENT_MAX_BYTES",
    "GOOGLE_DRIVE_NATIVE_EXPORT_MAX_BYTES",
    "GOOGLE_DRIVE_SOURCE_KIND",
    "GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID",
    "GOOGLE_WORKSPACE_SUPPORTED_SOURCE_KINDS",
    "GoogleDriveChange",
    "GoogleDriveChangePage",
    "GoogleDriveContentChanged",
    "GoogleDriveContentMode",
    "GoogleDriveContentReadClient",
    "GoogleDriveContentReader",
    "GoogleDriveContentTooLarge",
    "GoogleDriveContentUnavailable",
    "GoogleDriveFileContent",
    "GoogleDriveItem",
    "GoogleDriveItemKind",
    "GoogleDriveItemPage",
    "GoogleDriveKnowledgeReadClient",
    "GoogleDriveKnowledgeReader",
    "GoogleDriveScope",
    "GoogleDriveScopeKind",
    "GoogleDriveSharedDrive",
    "GoogleDriveSharedDrivePage",
    "GoogleDriveUnsupportedContent",
    "GoogleWorkspaceApiError",
    "GoogleWorkspaceBinaryPayload",
    "GoogleWorkspaceBinaryTransport",
    "GoogleWorkspaceClientFactory",
    "GoogleWorkspaceClientFamily",
    "GoogleWorkspaceCollectionPage",
    "GoogleWorkspaceCollaborationSuiteIntegration",
    "GoogleWorkspaceCollaborationSuiteIntegrationConfig",
    "GoogleWorkspaceCredentialResolver",
    "GoogleWorkspaceErrorKind",
    "GoogleWorkspaceHttpTransport",
    "GoogleWorkspacePageToken",
    "GoogleWorkspaceRequestExecutor",
    "GoogleWorkspaceRequestExecutorFactory",
    "GoogleWorkspaceRetryPolicy",
    "GoogleWorkspaceSourceKind",
    "GoogleWorkspaceTransport",
    "create_google_workspace_collaboration_suite",
    "create_google_workspace_collaboration_suite_integration",
    "parse_google_workspace_collection_page",
    "register_google_workspace_integration",
]

_DRIVE_EXPORTS = frozenset(
    {
        "ABSOLUTE_GOOGLE_DRIVE_CONTENT_MAX_BYTES",
        "DEFAULT_GOOGLE_DRIVE_CONTENT_MAX_BYTES",
        "GOOGLE_DRIVE_NATIVE_EXPORT_MAX_BYTES",
        "GOOGLE_DRIVE_SOURCE_KIND",
        "GoogleDriveChange",
        "GoogleDriveChangePage",
        "GoogleDriveContentChanged",
        "GoogleDriveContentMode",
        "GoogleDriveContentReadClient",
        "GoogleDriveContentReader",
        "GoogleDriveContentTooLarge",
        "GoogleDriveContentUnavailable",
        "GoogleDriveFileContent",
        "GoogleDriveItem",
        "GoogleDriveItemKind",
        "GoogleDriveItemPage",
        "GoogleDriveKnowledgeReadClient",
        "GoogleDriveKnowledgeReader",
        "GoogleDriveScope",
        "GoogleDriveScopeKind",
        "GoogleDriveSharedDrive",
        "GoogleDriveSharedDrivePage",
        "GoogleDriveUnsupportedContent",
    }
)

_BUNDLE_EXPORTS = frozenset(
    {
        "create_google_workspace_collaboration_suite",
        "create_google_workspace_collaboration_suite_integration",
    }
)

_FOUNDATION_EXPORTS = frozenset(
    {
        "DefaultGoogleWorkspaceClientFactory",
        "GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID",
        "GOOGLE_WORKSPACE_SUPPORTED_SOURCE_KINDS",
        "GoogleWorkspaceApiError",
        "GoogleWorkspaceBinaryPayload",
        "GoogleWorkspaceBinaryTransport",
        "GoogleWorkspaceClientFactory",
        "GoogleWorkspaceClientFamily",
        "GoogleWorkspaceCollectionPage",
        "GoogleWorkspaceCollaborationSuiteIntegration",
        "GoogleWorkspaceCollaborationSuiteIntegrationConfig",
        "GoogleWorkspaceCredentialResolver",
        "GoogleWorkspaceErrorKind",
        "GoogleWorkspaceHttpTransport",
        "GoogleWorkspacePageToken",
        "GoogleWorkspaceRequestExecutor",
        "GoogleWorkspaceRequestExecutorFactory",
        "GoogleWorkspaceRetryPolicy",
        "GoogleWorkspaceSourceKind",
        "GoogleWorkspaceTransport",
        "parse_google_workspace_collection_page",
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
    if name == "DefaultGoogleWorkspaceClientFactory":
        from intergrax.integrations.providers.collaboration_suite.google_workspace.client_family import (
            DefaultGoogleWorkspaceClientFactory,
        )

        return DefaultGoogleWorkspaceClientFactory
    if name in {
        "GoogleWorkspaceApiError",
        "GoogleWorkspaceCollectionPage",
        "GoogleWorkspaceErrorKind",
        "GoogleWorkspaceHttpTransport",
        "GoogleWorkspacePageToken",
        "GoogleWorkspaceRetryPolicy",
        "parse_google_workspace_collection_page",
    }:
        from intergrax.integrations.providers.collaboration_suite.google_workspace import (
            transport as _transport,
        )

        return export_from_bundle(_transport, name, _FOUNDATION_EXPORTS)
    if name in _DRIVE_EXPORTS:
        from intergrax.integrations.providers.collaboration_suite.google_workspace import (
            knowledge_read as _knowledge_read,
        )

        return export_from_bundle(_knowledge_read, name, _DRIVE_EXPORTS)
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
            "GoogleWorkspaceClientFamily",
            "GoogleWorkspaceCredentialResolver",
            "GoogleWorkspaceRequestExecutor",
            "GoogleWorkspaceRequestExecutorFactory",
            "GoogleWorkspaceSourceKind",
            "GoogleWorkspaceTransport",
            "GoogleWorkspaceBinaryPayload",
            "GoogleWorkspaceBinaryTransport",
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
