# © Artur Czarnecki. All rights reserved.

"""HTTP schemas for managed workspace product API (LKW-PRODUCT-1)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CreateWorkspaceRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    description: str = ""
    tenant_id: str | None = None


class WorkspaceResponseV1(BaseModel):
    workspace_id: str
    tenant_id: str
    name: str
    description: str = ""
    status: str
    created_at: datetime
    updated_at: datetime


class WorkspaceListResponseV1(BaseModel):
    workspaces: list[WorkspaceResponseV1]


class RegisterSourceRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["local_folder"] = "local_folder"
    path: str = Field(..., min_length=1)
    recursive: bool = True


class SourceResponseV1(BaseModel):
    source_id: str
    workspace_id: str
    source_type: str
    path: str
    status: str
    recursive: bool = True
    created_at: datetime | None = None
    last_sync_at: datetime | None = None


class SourceListResponseV1(BaseModel):
    sources: list[SourceResponseV1]


class SyncOperationAcceptedV1(BaseModel):
    operation_id: str
    workspace_id: str
    source_id: str
    status: str


class OperationResponseV1(BaseModel):
    operation_id: str
    operation_type: str
    status: str
    workspace_id: str
    source_id: str
    files_discovered: int = 0
    files_processed: int = 0
    files_failed: int = 0
    documents_indexed: int = 0
    documents_unchanged: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class WorkspaceSearchRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)


class WorkspaceSearchHitV1(BaseModel):
    document_id: str
    source_id: str
    workspace_id: str
    source_path: str
    file_name: str
    score: float = 0.0
    snippet: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceSearchResponseV1(BaseModel):
    workspace_id: str
    query: str
    results: list[WorkspaceSearchHitV1]
