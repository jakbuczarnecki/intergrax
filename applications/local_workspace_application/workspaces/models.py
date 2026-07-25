# © Artur Czarnecki. All rights reserved.

"""Managed workspace domain models (LKW-PRODUCT-1)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class WorkspaceStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class WorkspaceSourceType(StrEnum):
    LOCAL_FOLDER = "local_folder"
    MANAGED_UPLOAD = "managed_upload"
    UPLOADED_FOLDER_SNAPSHOT = "uploaded_folder_snapshot"
    CONNECTED_SOURCE = "connected_source"
    WEB_RESOURCE = "web_resource"


class WorkspaceSourceStatus(StrEnum):
    REGISTERED = "registered"
    SYNCING = "syncing"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class WorkspaceOperationType(StrEnum):
    SOURCE_SYNC = "source_sync"
    KNOWLEDGE_INGESTION = "knowledge_ingestion"


class WorkspaceOperationStatus(StrEnum):
    ACCEPTED = "accepted"
    QUEUED = "queued"
    RUNNING = "running"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class KnowledgeInputKind(StrEnum):
    MANAGED_FILE = "managed_file"
    UPLOADED_FOLDER_SNAPSHOT = "uploaded_folder_snapshot"
    SOURCE_CANDIDATE = "source_candidate"
    WEB_URL = "web_url"


class KnowledgeInputStatus(StrEnum):
    ACCEPTED = "accepted"
    RESOLVED = "resolved"
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
    source_type: WorkspaceSourceType = WorkspaceSourceType.LOCAL_FOLDER
    path: str = ""
    recursive: bool = False
    status: WorkspaceSourceStatus = WorkspaceSourceStatus.REGISTERED
    created_at: datetime
    last_sync_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_source_locator(self) -> Self:
        if self.source_type is WorkspaceSourceType.LOCAL_FOLDER:
            if not self.path.strip():
                raise ValueError("local_folder_path_required")
        else:
            if self.path != "":
                raise ValueError("non_local_path_must_be_empty")
            if self.recursive:
                raise ValueError("non_local_recursive_must_be_false")
        return self


class WorkspaceOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    workspace_id: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)
    operation_type: WorkspaceOperationType = WorkspaceOperationType.SOURCE_SYNC
    status: WorkspaceOperationStatus = WorkspaceOperationStatus.QUEUED
    files_discovered: int = 0
    files_processed: int = 0
    files_failed: int = 0
    documents_indexed: int = 0
    documents_unchanged: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    input_id: str | None = None
    queue_task_id: str | None = None
    queue_provider: str | None = None
    error_code: str | None = None
    created_at: datetime | None = None


class KnowledgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    workspace_id: str = Field(..., min_length=1)
    input_kind: KnowledgeInputKind
    idempotency_key: str = Field(..., min_length=1)
    operation_id: str = Field(..., min_length=1)
    source_id: str | None = None
    status: KnowledgeInputStatus
    submission_metadata: dict[str, str] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    error_code: str | None = None

    @field_validator("submission_metadata")
    @classmethod
    def _validate_submission_metadata(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            if not isinstance(key, str) or not isinstance(item, str):
                raise ValueError("submission_metadata_must_be_str_values")
            if "\n" in item or "\r" in item:
                raise ValueError("submission_metadata_multiline_forbidden")
        return value


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
