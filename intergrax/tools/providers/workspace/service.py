# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pathlib import Path

from intergrax.runtime.workspace.models import ShadowArtifact
from intergrax.runtime.workspace.shadow_workspace import ShadowWorkspace
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
WORKSPACE_EXPORT_ARTIFACT_TOOL_ID = "workspace.export_artifact"
WORKSPACE_IMPORT_ARTIFACT_TOOL_ID = "workspace.import_artifact"


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


def _require_object_storage(ctx: ToolWiringContext):
    storage = ctx.object_storage
    if storage is None:
        raise RuntimeError("object_storage_not_configured")
    return storage


def _safe_relative_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {relative_path}")
    return path


def workspace_export_artifact(
    ctx: ToolWiringContext,
    params: WorkspaceExportArtifactInput,
) -> WorkspaceExportArtifactOutput:
    workspace = _require_workspace(ctx)
    object_storage = _require_object_storage(ctx)
    rel = _safe_relative_path(params.path.strip())
    target = workspace.root / rel
    if not target.is_file():
        return WorkspaceExportArtifactOutput(
            exported=False,
            path=params.path.strip(),
            storage_key=params.storage_key.strip(),
            workspace_id=workspace.workspace_id,
            reason="artifact_not_found",
        )
    body = target.read_bytes()
    content_type = params.content_type.strip() or "application/octet-stream"
    for artifact in workspace.list_artifacts():
        if artifact.relative_path == rel.as_posix() and artifact.content_type:
            content_type = artifact.content_type
            break
    storage_key = params.storage_key.strip()
    object_storage.put(storage_key, body, content_type=content_type)
    return WorkspaceExportArtifactOutput(
        exported=True,
        path=rel.as_posix(),
        storage_key=storage_key,
        size_bytes=len(body),
        workspace_id=workspace.workspace_id,
        content_type=content_type,
        reason="ok",
    )


def workspace_import_artifact(
    ctx: ToolWiringContext,
    params: WorkspaceImportArtifactInput,
) -> WorkspaceImportArtifactOutput:
    workspace = _require_workspace(ctx)
    object_storage = _require_object_storage(ctx)
    storage_key = params.storage_key.strip()
    stored = object_storage.get(storage_key)
    if stored is None:
        return WorkspaceImportArtifactOutput(
            imported=False,
            path=params.path.strip(),
            storage_key=storage_key,
            workspace_id=workspace.workspace_id,
            reason="object_not_found",
        )
    content_type = params.content_type.strip() or stored.content_type
    if content_type.startswith("text/") or content_type in {"application/json", "application/xml"}:
        workspace.write_text(params.path.strip(), stored.body.decode("utf-8"), content_type=content_type)
    else:
        rel = _safe_relative_path(params.path.strip())
        target = workspace.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(stored.body)
    return WorkspaceImportArtifactOutput(
        imported=True,
        path=params.path.strip(),
        storage_key=storage_key,
        size_bytes=len(stored.body),
        workspace_id=workspace.workspace_id,
        reason="ok",
    )


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
