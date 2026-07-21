# © Artur Czarnecki. All rights reserved.

"""Managed workspace domain models (LKW-PRODUCT-1)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WorkspaceStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class WorkspaceSourceType(StrEnum):
    LOCAL_FOLDER = "local_folder"


class WorkspaceSourceStatus(StrEnum):
    REGISTERED = "registered"
    SYNCING = "syncing"
    READY = "ready"
    ERROR = "error"


class WorkspaceOperationType(StrEnum):
    SOURCE_SYNC = "source_sync"


class WorkspaceOperationStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Workspace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = ""
    status: WorkspaceStatus = WorkspaceStatus.ACTIVE
    created_at: datetime
    updated_at: datetime


class WorkspaceSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1)
    workspace_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    source_type: Literal[WorkspaceSourceType.LOCAL_FOLDER] = WorkspaceSourceType.LOCAL_FOLDER
    path: str = Field(..., min_length=1)
    recursive: bool = True
    status: WorkspaceSourceStatus = WorkspaceSourceStatus.REGISTERED
    created_at: datetime
    last_sync_at: datetime | None = None


class WorkspaceOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    workspace_id: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)
    operation_type: Literal[WorkspaceOperationType.SOURCE_SYNC] = (
        WorkspaceOperationType.SOURCE_SYNC
    )
    status: WorkspaceOperationStatus = WorkspaceOperationStatus.QUEUED
    files_discovered: int = 0
    files_processed: int = 0
    files_failed: int = 0
    documents_indexed: int = 0
    documents_unchanged: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class WorkspaceDocumentReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    workspace_id: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    file_name: str = Field(..., min_length=1)
    content_hash: str = Field(..., min_length=1)
    indexed_at: datetime
