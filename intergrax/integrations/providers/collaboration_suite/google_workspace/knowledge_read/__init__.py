# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace provider-specific knowledge-read surfaces."""

from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
    GOOGLE_DRIVE_SOURCE_KIND,
    GoogleDriveChange,
    GoogleDriveChangePage,
    GoogleDriveItem,
    GoogleDriveItemKind,
    GoogleDriveItemPage,
    GoogleDriveKnowledgeReadClient,
    GoogleDriveKnowledgeReader,
    GoogleDriveScope,
    GoogleDriveScopeKind,
    GoogleDriveSharedDrive,
    GoogleDriveSharedDrivePage,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive_content import (
    ABSOLUTE_GOOGLE_DRIVE_CONTENT_MAX_BYTES,
    DEFAULT_GOOGLE_DRIVE_CONTENT_MAX_BYTES,
    GOOGLE_DRIVE_NATIVE_EXPORT_MAX_BYTES,
    GoogleDriveContentChanged,
    GoogleDriveContentMode,
    GoogleDriveContentReadClient,
    GoogleDriveContentReader,
    GoogleDriveContentTooLarge,
    GoogleDriveContentUnavailable,
    GoogleDriveFileContent,
    GoogleDriveUnsupportedContent,
)

__all__ = [
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
]
