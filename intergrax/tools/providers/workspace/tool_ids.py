# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral workspace catalog tool identifiers."""

from __future__ import annotations

from intergrax.contracts.workspace_tool_ids import (
    WORKSPACE_SEARCH_TOOL_ID,
    WORKSPACE_WRITE_FILE_TOOL_ID,
)

WORKSPACE_READ_FILE_TOOL_ID = "workspace.read_file"
WORKSPACE_LIST_FILES_TOOL_ID = "workspace.list_files"
WORKSPACE_SNAPSHOT_TOOL_ID = "workspace.snapshot"
WORKSPACE_DELETE_FILE_TOOL_ID = "workspace.delete_file"
WORKSPACE_EXPORT_ARTIFACT_TOOL_ID = "workspace.export_artifact"
WORKSPACE_IMPORT_ARTIFACT_TOOL_ID = "workspace.import_artifact"

__all__ = [
    "WORKSPACE_DELETE_FILE_TOOL_ID",
    "WORKSPACE_EXPORT_ARTIFACT_TOOL_ID",
    "WORKSPACE_IMPORT_ARTIFACT_TOOL_ID",
    "WORKSPACE_LIST_FILES_TOOL_ID",
    "WORKSPACE_READ_FILE_TOOL_ID",
    "WORKSPACE_SEARCH_TOOL_ID",
    "WORKSPACE_SNAPSHOT_TOOL_ID",
    "WORKSPACE_WRITE_FILE_TOOL_ID",
]
