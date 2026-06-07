# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.workspace.contracts import (
    WorkspaceArtifactOutput,
    WorkspaceListFilesInput,
    WorkspaceListFilesOutput,
    WorkspaceReadFileInput,
    WorkspaceReadFileOutput,
    WorkspaceSnapshotInput,
    WorkspaceSnapshotOutput,
    WorkspaceWriteFileInput,
)
from intergrax.tools.providers.workspace.service import (
    workspace_list_files,
    workspace_read_file,
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
