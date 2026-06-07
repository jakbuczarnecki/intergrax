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
