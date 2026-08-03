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

__all__ = [
    "GOOGLE_DRIVE_SOURCE_KIND",
    "GoogleDriveChange",
    "GoogleDriveChangePage",
    "GoogleDriveItem",
    "GoogleDriveItemKind",
    "GoogleDriveItemPage",
    "GoogleDriveKnowledgeReadClient",
    "GoogleDriveKnowledgeReader",
    "GoogleDriveScope",
    "GoogleDriveScopeKind",
    "GoogleDriveSharedDrive",
    "GoogleDriveSharedDrivePage",
]
