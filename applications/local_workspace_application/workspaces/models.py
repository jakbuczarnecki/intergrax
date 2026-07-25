# © Artur Czarnecki. All rights reserved.

"""Managed workspace domain models (LKW-PRODUCT-1)."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SUBMISSION_METADATA_MAX_ENTRIES = 16
_SUBMISSION_METADATA_MAX_KEY_LEN = 64
_SUBMISSION_METADATA_MAX_VALUE_LEN = 256
_SUBMISSION_METADATA_KEY_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SUBMISSION_METADATA_SENSITIVE_SEGMENTS = frozenset(
    {
        "token",
        "secret",
        "password",
        "passwd",
        "credential",
        "credentials",
        "authorization",
        "auth",
        "apikey",
        "url",
        "uri",
        "path",
        "filepath",
    }
)
_SUBMISSION_METADATA_SENSITIVE_TOKEN_SEQUENCES = frozenset(
    {
        ("api", "key"),
        ("file", "path"),
        ("local", "path"),
    }
)
_SUBMISSION_METADATA_SCHEME_RE = re.compile(r"(?i)[a-z][a-z0-9+.-]*://")
_SUBMISSION_METADATA_WINDOWS_PATH_RE = re.compile(r"(?i)^[a-z]:[\\/]")
_SUBMISSION_METADATA_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _contains_sensitive_token_sequence(tokens: tuple[str, ...]) -> bool:
    for sequence in _SUBMISSION_METADATA_SENSITIVE_TOKEN_SEQUENCES:
        width = len(sequence)
        for start in range(len(tokens) - width + 1):
            if tokens[start : start + width] == sequence:
                return True
    return False


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

    @field_validator("submission_metadata", mode="before")
    @classmethod
    def _validate_submission_metadata(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("submission_metadata_must_be_string_map")
        if len(value) > _SUBMISSION_METADATA_MAX_ENTRIES:
            raise ValueError("submission_metadata_too_many_entries")
        validated: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not isinstance(item, str):
                raise ValueError("submission_metadata_must_be_string_map")
            if len(key) > _SUBMISSION_METADATA_MAX_KEY_LEN:
                raise ValueError("submission_metadata_key_too_long")
            if len(item) > _SUBMISSION_METADATA_MAX_VALUE_LEN:
                raise ValueError("submission_metadata_value_too_long")
            if _SUBMISSION_METADATA_KEY_RE.fullmatch(key) is None:
                raise ValueError("submission_metadata_invalid_key")
            tokens = tuple(re.split(r"[._-]", key))
            if any(token in _SUBMISSION_METADATA_SENSITIVE_SEGMENTS for token in tokens):
                raise ValueError("submission_metadata_sensitive_key_forbidden")
            if _contains_sensitive_token_sequence(tokens):
                raise ValueError("submission_metadata_sensitive_key_forbidden")
            if _is_unsafe_submission_metadata_value(item):
                raise ValueError("submission_metadata_unsafe_value")
            validated[key] = item
        return validated


def _is_unsafe_submission_metadata_value(value: str) -> bool:
    if _SUBMISSION_METADATA_CONTROL_RE.search(value) is not None:
        return True
    if _SUBMISSION_METADATA_SCHEME_RE.search(value) is not None:
        return True
    if _SUBMISSION_METADATA_WINDOWS_PATH_RE.match(value) is not None:
        return True
    if value.startswith("\\\\"):
        return True
    if value.startswith("/"):
        return True
    if value.startswith("~/") or value.startswith("~\\"):
        return True
    if value.casefold().startswith("bearer "):
        return True
    return False


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
