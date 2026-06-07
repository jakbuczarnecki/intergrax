# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class WorkspaceWriteFileInput(BaseModel):
    path: str = Field(..., min_length=1, description="Relative path inside the shadow workspace.")
    content: str = Field(..., description="UTF-8 text content to write.")
    content_type: str = Field(default="text/plain", description="MIME type for the artifact.")


class WorkspaceArtifactOutput(BaseModel):
    artifact_id: str
    relative_path: str
    size_bytes: int
    content_type: str
    sha256: str
    workspace_id: str = ""


class WorkspaceReadFileInput(BaseModel):
    path: str = Field(..., min_length=1, description="Relative path inside the shadow workspace.")


class WorkspaceReadFileOutput(BaseModel):
    path: str
    content: str
    workspace_id: str = ""


class WorkspaceListFilesInput(BaseModel):
    pass


class WorkspaceListFilesOutput(BaseModel):
    artifacts: list[WorkspaceArtifactOutput] = Field(default_factory=list)
    workspace_id: str = ""
    artifact_count: int = 0


class WorkspaceSnapshotInput(BaseModel):
    pass


class WorkspaceSnapshotOutput(BaseModel):
    workspace_id: str
    created_at_utc: str
    files: dict[str, str] = Field(default_factory=dict)
    file_count: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class WorkspaceDeleteFileInput(BaseModel):
    path: str = Field(..., min_length=1, description="Relative path inside the shadow workspace.")


class WorkspaceDeleteFileOutput(BaseModel):
    path: str
    deleted: bool = False
    workspace_id: str = ""


class WorkspaceSearchInput(BaseModel):
    query: str = Field(..., min_length=1, description="Substring to search for in workspace files.")
    path_prefix: str = Field(default="", description="Optional relative path prefix filter.")
    case_insensitive: bool = True
    max_matches: int = Field(default=50, ge=1, le=500)


class WorkspaceSearchMatch(BaseModel):
    path: str
    line_number: int
    line: str


class WorkspaceSearchOutput(BaseModel):
    matches: list[WorkspaceSearchMatch] = Field(default_factory=list)
    match_count: int = 0
    workspace_id: str = ""
