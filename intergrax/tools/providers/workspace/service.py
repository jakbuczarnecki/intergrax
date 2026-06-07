# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.workspace.models import ShadowArtifact
from intergrax.runtime.workspace.shadow_workspace import ShadowWorkspace
from intergrax.tools.providers.workspace.contracts import (
    WorkspaceArtifactOutput,
    WorkspaceDeleteFileInput,
    WorkspaceDeleteFileOutput,
    WorkspaceListFilesInput,
    WorkspaceListFilesOutput,
    WorkspaceReadFileInput,
    WorkspaceReadFileOutput,
    WorkspaceSearchInput,
    WorkspaceSearchMatch,
    WorkspaceSearchOutput,
    WorkspaceSnapshotInput,
    WorkspaceSnapshotOutput,
    WorkspaceWriteFileInput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

WORKSPACE_WRITE_FILE_TOOL_ID = "workspace.write_file"
WORKSPACE_READ_FILE_TOOL_ID = "workspace.read_file"
WORKSPACE_LIST_FILES_TOOL_ID = "workspace.list_files"
WORKSPACE_SNAPSHOT_TOOL_ID = "workspace.snapshot"
WORKSPACE_DELETE_FILE_TOOL_ID = "workspace.delete_file"
WORKSPACE_SEARCH_TOOL_ID = "workspace.search"


def _require_workspace(ctx: ToolWiringContext) -> ShadowWorkspace:
    workspace = ctx.shadow_workspace
    if workspace is None:
        raise RuntimeError("shadow_workspace_not_configured")
    if not isinstance(workspace, ShadowWorkspace):
        raise RuntimeError("shadow_workspace_invalid_type")
    return workspace


def _artifact_output(artifact: ShadowArtifact, *, workspace_id: str) -> WorkspaceArtifactOutput:
    return WorkspaceArtifactOutput(
        artifact_id=artifact.artifact_id,
        relative_path=artifact.relative_path,
        size_bytes=artifact.size_bytes,
        content_type=artifact.content_type,
        sha256=artifact.sha256,
        workspace_id=workspace_id,
    )


def workspace_write_file(ctx: ToolWiringContext, params: WorkspaceWriteFileInput) -> WorkspaceArtifactOutput:
    workspace = _require_workspace(ctx)
    artifact = workspace.write_text(
        params.path.strip(),
        params.content,
        content_type=params.content_type,
    )
    return _artifact_output(artifact, workspace_id=workspace.workspace_id)


def workspace_read_file(ctx: ToolWiringContext, params: WorkspaceReadFileInput) -> WorkspaceReadFileOutput:
    workspace = _require_workspace(ctx)
    content = workspace.read_text(params.path.strip())
    return WorkspaceReadFileOutput(
        path=params.path.strip(),
        content=content,
        workspace_id=workspace.workspace_id,
    )


def workspace_list_files(ctx: ToolWiringContext, params: WorkspaceListFilesInput) -> WorkspaceListFilesOutput:
    workspace = _require_workspace(ctx)
    artifacts = [
        _artifact_output(item, workspace_id=workspace.workspace_id)
        for item in workspace.list_artifacts()
    ]
    return WorkspaceListFilesOutput(
        artifacts=artifacts,
        workspace_id=workspace.workspace_id,
        artifact_count=len(artifacts),
    )


def workspace_snapshot(ctx: ToolWiringContext, params: WorkspaceSnapshotInput) -> WorkspaceSnapshotOutput:
    workspace = _require_workspace(ctx)
    snapshot = workspace.snapshot()
    return WorkspaceSnapshotOutput(
        workspace_id=snapshot.workspace_id,
        created_at_utc=snapshot.created_at_utc,
        files=dict(snapshot.files),
        file_count=len(snapshot.files),
    )


def workspace_delete_file(ctx: ToolWiringContext, params: WorkspaceDeleteFileInput) -> WorkspaceDeleteFileOutput:
    workspace = _require_workspace(ctx)
    path = params.path.strip()
    deleted = workspace.delete_file(path)
    return WorkspaceDeleteFileOutput(path=path, deleted=deleted, workspace_id=workspace.workspace_id)


def workspace_search(ctx: ToolWiringContext, params: WorkspaceSearchInput) -> WorkspaceSearchOutput:
    workspace = _require_workspace(ctx)
    raw_matches = workspace.search_text(
        params.query,
        path_prefix=params.path_prefix,
        case_insensitive=params.case_insensitive,
        max_matches=params.max_matches,
    )
    matches = [
        WorkspaceSearchMatch(path=path, line_number=line_number, line=line)
        for path, line_number, line in raw_matches
    ]
    return WorkspaceSearchOutput(
        matches=matches,
        match_count=len(matches),
        workspace_id=workspace.workspace_id,
    )
