# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.workspace.contracts import (
    WorkspaceArtifactOutput,
    WorkspaceDeleteFileInput,
    WorkspaceDeleteFileOutput,
    WorkspaceExportArtifactInput,
    WorkspaceExportArtifactOutput,
    WorkspaceImportArtifactInput,
    WorkspaceImportArtifactOutput,
    WorkspaceListFilesInput,
    WorkspaceListFilesOutput,
    WorkspaceReadFileInput,
    WorkspaceReadFileOutput,
    WorkspaceSearchInput,
    WorkspaceSearchOutput,
    WorkspaceSnapshotInput,
    WorkspaceSnapshotOutput,
    WorkspaceWriteFileInput,
)
from intergrax.tools.providers.workspace.service import (
    workspace_delete_file,
    workspace_export_artifact,
    workspace_import_artifact,
    workspace_list_files,
    workspace_read_file,
    workspace_search,
    workspace_snapshot,
    workspace_write_file,
)


class WorkspaceWriteFileHandler(ServiceToolHandler[WorkspaceWriteFileInput, WorkspaceArtifactOutput]):
    _service = workspace_write_file


class WorkspaceReadFileHandler(ServiceToolHandler[WorkspaceReadFileInput, WorkspaceReadFileOutput]):
    _service = workspace_read_file


class WorkspaceListFilesHandler(ServiceToolHandler[WorkspaceListFilesInput, WorkspaceListFilesOutput]):
    _service = workspace_list_files


class WorkspaceSnapshotHandler(ServiceToolHandler[WorkspaceSnapshotInput, WorkspaceSnapshotOutput]):
    _service = workspace_snapshot


class WorkspaceDeleteFileHandler(ServiceToolHandler[WorkspaceDeleteFileInput, WorkspaceDeleteFileOutput]):
    _service = workspace_delete_file


class WorkspaceSearchHandler(ServiceToolHandler[WorkspaceSearchInput, WorkspaceSearchOutput]):
    _service = workspace_search


class WorkspaceExportArtifactHandler(
    ServiceToolHandler[WorkspaceExportArtifactInput, WorkspaceExportArtifactOutput]
):
    _service = workspace_export_artifact


class WorkspaceImportArtifactHandler(
    ServiceToolHandler[WorkspaceImportArtifactInput, WorkspaceImportArtifactOutput]
):
    _service = workspace_import_artifact
